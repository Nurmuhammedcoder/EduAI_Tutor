import sympy as sp
import google.generativeai as genai
import re
import hashlib
import sqlite3
import datetime
import streamlit as st
import time
import random

def get_cache_connection():
    try:
        return sqlite3.connect('ai_cache.db', check_same_thread=False)
    except Exception as e:
        st.error(f"Ошибка подключения к кэшу: {str(e)}")
        return None

def check_brackets(expr):
    stack = []
    for char in expr:
        if char == '(': stack.append(char)
        elif char == ')':
            if not stack: return False
            stack.pop()
    return len(stack) == 0

def generate_detailed_feedback(task, answer):
    common_mistakes = {
        "раскрытие скобок": "Не забывайте учитывать знаки при раскрытии скобок! Помните: -(a+b) = -a - b",
        "степени": "Правила степеней: a^m * a^n = a^(m+n), (a^m)^n = a^(m*n), (a*b)^n = a^n * b^n",
        "факторизация": "Используйте формулы сокращенного умножения: a^2 - b^2 = (a-b)(a+b), a^2 ± 2ab + b^2 = (a±b)^2",
        "упрощение": "Упрощайте выражение до конца: объединяйте подобные слагаемые, сокращайте дроби",
        "знаки": "Внимательно следите за знаками, особенно при переносе выражений через знак равенства",
        "дроби": "При работе с дробями находите общий знаменатель, сокращайте дроби",
        "синтаксис": "Используйте правильный синтаксис: ** для степеней, * для умножения"
    }
    
    feedback = ""
    if "(" in answer and not check_brackets(answer):
        feedback += "⚠️ **Ошибка в расстановке скобок!**\n" + common_mistakes["раскрытие скобок"] + "\n\n"
    if "^" in answer:
        feedback += "⚠️ **Используйте ** вместо ^ для степеней**\nНапример: x**2 вместо x^2\n" + common_mistakes["степени"] + "\n\n"
    if "factor" in task and "factor" not in answer:
        feedback += "⚠️ **Задание требует факторизации (разложения на множители)**\n" + common_mistakes["факторизация"] + "\n\n"
    if "/" in answer and "simplify" in task:
        feedback += "⚠️ **Требуется упрощение дробей**\n" + common_mistakes["дроби"] + "\n\n"
    if "=" in answer and "solve" not in task:
        feedback += "⚠️ **Возможно, вы пытались решить уравнение вместо упрощения выражения**\n"
        
    return feedback if feedback else "Проверьте правильность всех операций и знаков. Убедитесь, что выражение полностью упрощено."

def generate_math_explanation(task_expr, answer_expr):
    steps = []
    simplified = sp.simplify(task_expr)
    steps.append(f"1. **Упрощаем исходное выражение:**\n   `{sp.pretty(task_expr)}` → `{sp.pretty(simplified)}`")
    
    expanded = sp.expand(simplified)
    factored = sp.factor(simplified)
    
    if expanded != simplified:
        steps.append(f"2. **Раскрываем скобки:**\n   `{sp.pretty(expanded)}`")
    if factored != simplified:
        steps.append(f"3. **Факторизуем:**\n   `{sp.pretty(factored)}`")
    
    if not task_expr.equals(answer_expr):
        try:
            student_simplified = sp.simplify(answer_expr)
            steps.append(f"4. **Ваш ответ:**\n   `{sp.pretty(answer_expr)}`")
            steps.append(f"5. **После упрощения:**\n   `{sp.pretty(student_simplified)}`")
            steps.append("6. **🔍 Сравнение:** " + (
                "✅ Выражения идентичны после упрощения!" 
                if simplified.equals(student_simplified) 
                else "❌ Выражения отличаются"
            ))
        except:
            steps.append("4. **Ваш ответ содержит синтаксическую ошибку**")
    
    return "\n\n".join(steps)

def is_valid_math_expression(expr):
    allowed_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-*/^=()[]{},. ")
    if any(char not in allowed_chars for char in expr):
        return False
    
    forbidden = {'__', 'import', 'eval', 'exec', 'open', 'lambda', ';', 'os', 'sys', 'subprocess', 'shutil', 'glob'}
    if any(func in expr for func in forbidden):
        return False
    
    if re.search(r'\b[a-zA-Z_][a-zA-Z0-9_]*\s*\(', expr):
        return False
    
    if len(expr) > 500:
        return False
        
    return True

def check_math(task, answer):
    try:
        if not is_valid_math_expression(task) or not is_valid_math_expression(answer):
            return {
                'is_correct': False,
                'explanation': "Выражение содержит недопустимые символы или функции",
                'correct_answer': "",
                'latex_answer': "",
                'detailed_feedback': "Пожалуйста, используйте только математические выражения без вызовов функций."
            }
        
        expr_task = sp.sympify(task)
        expr_answer = sp.sympify(answer)
        
        is_correct = expr_task.equals(expr_answer)
        
        explanation = generate_math_explanation(expr_task, expr_answer)
        detailed_fb = generate_detailed_feedback(task, answer)
        
        return {
            'is_correct': is_correct,
            'explanation': explanation,
            'correct_answer': sp.pretty(expr_task),
            'latex_answer': sp.latex(expr_task),
            'detailed_feedback': detailed_fb
        }
    except Exception as e:
        return {
            'is_correct': False,
            'explanation': f"Ошибка разбора: {str(e)}",
            'correct_answer': "",
            'latex_answer': "",
            'detailed_feedback': "Проверьте правильность ввода выражения. Убедитесь, что используете правильный синтаксис."
        }

def get_math_ai_feedback(task, student_answer, correct_answer, api_key, model_name, prompt):
    try:
        if not api_key:
            return "API ключ не установлен"
        
        query_hash = hashlib.md5(f"{task}_{student_answer}_{prompt}_{model_name}".encode()).hexdigest()
        conn = get_cache_connection()
        if not conn:
            return "Ошибка подключения к кэшу"
        
        c = conn.cursor()
        c.execute("SELECT response FROM responses WHERE query_hash = ?", (query_hash,))
        cached_response = c.fetchone()
        
        if cached_response:
            return cached_response[0]
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        
        full_prompt = f"""
Ты опытный репетитор по математике. Объясни решение задачи ученику средней школы понятным языком.

Задание: {task}
Ответ ученика: {student_answer}
Правильный ответ: {correct_answer}

{prompt}

Объясни:
1. Как решать этот тип задач (понятными словами)
2. Пошаговое решение с комментариями
3. Где ученик допустил ошибку и почему
4. Как избежать подобных ошибок в будущем
5. Приведи аналогичный пример для закрепления

Ответь на русском языке, используя дружелюбный и поддерживающий тон.
"""
        response = model.generate_content(
            full_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.3,
                max_output_tokens=2000,
                top_p=0.95
            ),
            safety_settings={
                'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE',
                'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE',
                'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE',
                'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE'
            }
        )
        
        if response.candidates and response.candidates[0].content.parts:
            result = response.candidates[0].content.parts[0].text
        else:
            result = "Не удалось получить ответ от модели. Попробуйте еще раз."
        
        c.execute("INSERT OR REPLACE INTO responses (query_hash, response, timestamp) VALUES (?, ?, ?)",
                  (query_hash, result, datetime.datetime.now()))
        conn.commit()
        conn.close()
        
        return result
    
    except Exception as e:
        error_str = str(e)
        if "429" in error_str or "quota" in error_str.lower():
            return "⚠️ Достигнут лимит запросов. Подождите 60 секунд."
        elif "404" in error_str:
            return "⚠️ Модель не найдена."
        else:
            return f"Ошибка генерации: {error_str}"

def generate_personalized_task(student_id):
    try:
        conn = get_cache_connection()
        if not conn:
            return "Решите уравнение: 2*x + 5 = 15", "5"
        
        c = conn.cursor()
        
        c.execute("""
            SELECT error_type, COUNT(*) as count 
            FROM error_history 
            WHERE student_id = ? AND subject = 'math'
            GROUP BY error_type 
            ORDER BY count DESC 
            LIMIT 1
        """, (student_id,))
        error_row = c.fetchone()
        
        if error_row:
            error_type = error_row[0]
            if "brackets" in error_type:
                return "Раскройте скобки: (x + 2)*(x - 3)", "x**2 - x - 6"
            elif "factor" in error_type:
                return "Разложите на множители: x**2 - 4", "(x-2)(x+2)"
            elif "simplify" in error_type:
                return "Упростите выражение: (x**2 - 4)/(x-2)", "x+2"
        
        c.execute("""
            SELECT topic, AVG(score) as avg_score
            FROM learning_history
            WHERE student_id = ? AND subject = 'math'
            GROUP BY topic
            ORDER BY avg_score ASC
            LIMIT 1
        """, (student_id,))
        topic_row = c.fetchone()
        
        if topic_row:
            topic = topic_row[0]
            if "Алгебра" in topic:
                return "Решите уравнение: 2*x + 5 = 15", "5"
            elif "Геометрия" in topic:
                return "Найдите площадь треугольника с основанием 5 и высотой 3", "7.5"
            elif "Тригонометрия" in topic:
                return "Упростите: sin(x)**2 + cos(x)**2", "1"
        
        return "Решите уравнение: 3*x - 7 = 8", "5"
    except:
        return "Решите уравнение: 2*x + 5 = 15", "5"