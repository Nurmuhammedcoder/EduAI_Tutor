import os
import streamlit as st
from PIL import Image
import pytesseract
import plotly.express as px
import plotly.graph_objects as go
import datetime
import time
import google.generativeai as genai
import platform
import hashlib
import sqlite3
import re
import numpy as np
import pandas as pd
import requests
import json
from math_checker import check_math, get_math_ai_feedback, generate_personalized_task
from english_checker import check_english, get_grammar_explanation
import tempfile
import base64
import warnings
import traceback
from streamlit_lottie import st_lottie
from streamlit_player import st_player
import calendar
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from io import BytesIO
import fitz  # PyMuPDF
import pdfplumber
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from canvas_api import CanvasAPI
import logging
from contextlib import contextmanager
from functools import wraps

# Фильтрация предупреждений
warnings.filterwarnings("ignore", category=UserWarning, module="google")

# Настройка логирования - СНАЧАЛА!
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('eduai.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('EduAI')

# Настройки среды
if platform.system() == 'Windows':
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
elif platform.system() == 'Darwin':
    pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'
else:
    pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

os.environ['OMP_THREAD_LIMIT'] = '1'

# Импорт новой системы оценивания
try:
    from grading_system import GradingSystem, LearningAnalytics
    GRADING_SYSTEM_AVAILABLE = True
except ImportError:
    logger.warning("grading_system.py не найден. Используется базовая система оценивания.")
    GRADING_SYSTEM_AVAILABLE = False

# Декоратор для обработки ошибок
def handle_errors(operation_name):
    """Декоратор для единообразной обработки ошибок"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Ошибка в {operation_name}: {str(e)}", exc_info=True)
                st.error(f"Произошла ошибка в {operation_name}: {str(e)}")
                return None
        return wrapper
    return decorator

# Кэширование для производительности
@st.cache_resource
def load_language_tool():
    """Кэшируем инициализацию LanguageTool"""
    try:
        import language_tool_python
        return language_tool_python.LanguageTool('en-US')
    except Exception as e:
        logger.error(f"Ошибка загрузки LanguageTool: {str(e)}")
        return None

# Настройки интерфейса
st.set_page_config(
    page_title="EduAI Tutor Pro",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/eduai-tutor',
        'Report a bug': "https://github.com/eduai-tutor/issues",
        'About': "EduAI Tutor Pro - AI система проверки домашних заданий"
    }
)
def log_performance(func):
    """Декоратор для мониторинга производительности функций"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        
        if elapsed > 2:  # Логируем только медленные операции
            logger.warning(f"{func.__name__} выполнялась {elapsed:.2f}с")
        
        return result
    return wrapper
# Применяем к критическим функциям:

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('eduai.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('EduAI')

# Импорт новой системы оценивания
try:
    from grading_system import GradingSystem, LearningAnalytics
    GRADING_SYSTEM_AVAILABLE = True
except ImportError:
    logger.warning("grading_system.py не найден. Используется базовая система оценивания.")
    GRADING_SYSTEM_AVAILABLE = False
# Фильтрация предупреждений
warnings.filterwarnings("ignore", category=UserWarning, module="google")

# Настройки среды
if platform.system() == 'Windows':
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
elif platform.system() == 'Darwin':
    pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'
else:
    pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

os.environ['OMP_THREAD_LIMIT'] = '1'

# Настройки интерфейса
st.set_page_config(
    page_title="EduAI Tutor Pro",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/eduai-tutor',
        'Report a bug': "https://github.com/eduai-tutor/issues",
        'About': "EduAI Tutor Pro - AI система проверки домашних заданий"
    }
)

# Словарь переводов
translations = {
    "Русский": {
        "title": "🚀 EduAI Tutor Pro",
        "subject": "Выберите предмет:",
        "math": "Математика",
        "eng": "Английский",
        "test": "Тест (PDF)",
        "check_math": "Проверить математику",
        "upload": "Загрузите работу студента (PDF или изображение)",
        "example": "Пример:",
        "task": "Введите задание:",
        "answer": "Введите ответ:",
        "correct": "✅ Верно! Отличная работа!",
        "incorrect": "❌ Ответ неверный. Давайте разберем:",
        "solution": "📚 Подробное решение:",
        "original": "Исходное задание:",
        "explanation": "Пошаговое объяснение:",
        "correct_answer": "Правильный ответ:",
        "text_rep": "Текстовое представление:",
        "errors": "🔍 Анализ возможных ошибок",
        "progress": "📊 Ваш прогресс",
        "history": "История ваших ответов",
        "check_text": "Проверить текст",
        "no_errors": "✅ Ошибок не найдено!",
        "why_error": "📖 Почему это ошибка?",
        "gpt_feedback": "🤖 AI-рекомендация",
        "student_id": "ID студента:",
        "feedback": "Оставьте отзыв:",
        "send_feedback": "Отправить отзыв",
        "ai_math": "🤖 Получить AI-объяснение",
        "ai_math_prompt": "Объясните решение задачи как репетитор",
        "ai_loading": "Генерируем AI-объяснение...",
        "ai_error": "Ошибка AI, попробуйте позже",
        "settings": "⚙️ Настройки",
        "api_settings": "🔑 Настройки Google Gemini",
        "api_key": "Введите ваш Google AI API ключ:",
        "api_help": "Получите на aistudio.google.com/app/apikey",
        "key_saved": "Ключ сохранен!",
        "mobile_warning": "📱 Для лучшего опыта используйте горизонтальную ориентацию",
        "connection_success": "✅ Подключение к Google Gemini успешно!",
        "connection_fail": "❌ Ошибка подключения. Проверьте ключ API",
        "available_models": "Доступные модели:",
        "invalid_key": "Недействительный API ключ",
        "ai_retry": "🔄 Попробовать снова",
        "ai_clear": "❌ Очистить ответ",
        "rate_limit": "⚠️ Достигнут лимит запросов. Подождите 60 секунд",
        "model_not_found": "⚠️ Модель не найдена. Используйте другую модель.",
        "model_selection": "Выберите модель AI:",
        "default_model": "gemini-pro",
        "wait_message": "⏳ Ожидаем снятия ограничения API...",
        "retry_countdown": "Повторная попытка через:",
        "app_name": "EduAI Tutor Pro",
        "app_description": "Интеллектуальная система проверки домашних заданий",
        "features": "🌟 Премиум функции",
        "pro_tip": "💡 Профессиональный совет",
        "processing": "🔍 Обработка файла...",
        "extraction_success": "✅ Текст успешно извлечен!",
        "extraction_error": "⚠️ Не удалось извлечь текст из файла",
        "processing_error": "❌ Ошибка обработки файла",
        "pdf_warning": "⚠️ Для работы с PDF установите PyMuPDF: pip install PyMuPDF",
        "export_results": "📥 Экспорт результатов",
        "export_btn": "Скачать отчет",
        "pdf_processing": "📄 Обработка PDF...",
        "ocr_processing": "🖼️ Распознавание изображения...",
        "canvas_integration": "🎓 Интеграция с Canvas LMS",
        "canvas_url": "Canvas URL:",
        "canvas_key": "Canvas API Key:",
        "canvas_help": "Получите в настройках Canvas",
        "canvas_courses": "Выберите курс:",
        "canvas_assignments": "Выберите задание:",
        "canvas_load": "Загрузить из Canvas",
        "canvas_upload": "Отправить оценку в Canvas",
        "canvas_success": "✅ Оценка отправлена в Canvas!",
        "canvas_error": "❌ Ошибка интеграции с Canvas",
        "adaptive_learning": "🎯 Персонализированные задания",
        "generate_task": "Сгенерировать задание",
        "learning_analytics": "📈 Аналитика обучения",
        "topic_analysis": "📚 Анализ по темам",
        "activity_calendar": "🗓️ Календарь активности",
        "language_select": "🌐 Язык интерфейса:",
        "error_cloud": "☁️ Облако ошибок",
        "video_explanation": "🎥 Видео-объяснение",
        "file_type_error": "⚠️ Неподдерживаемый формат файла. Используйте PDF или изображения (PNG, JPG)",
        "test_mode": "📝 Тест с множественным выбором",
        "upload_test": "Загрузите PDF с тестом",
        "upload_answers": "Загрузите PDF с ответами студента",
        "process_test": "Обработать тест",
        "question": "Вопрос",
        "student_answer": "Ответ студента",
        "correct_answer": "Правильный ответ",
        "result": "Результат",
        "score": "Оценка",
        "correct_answers": "Правильных ответов",
        "test_results": "Результаты теста",
        "test_analysis": "📊 Анализ теста",
        "answer_key": "Ключ ответов",
        "extract_answers": "Извлечь ответы",
        "test_pdf_error": "Ошибка обработки теста PDF",
        "answers_pdf_error": "Ошибка обработки ответов PDF",
        "no_questions_found": "Вопросы не найдены",
        "no_answers_found": "Ответы не найдены",
        "test_processed": "✅ Тест успешно обработан",
        "answers_processed": "✅ Ответы успешно обработаны",
        "view_test": "Просмотр теста",
        "download_results": "Скачать результаты",
        "answer_format": "Формат ответов: A, B, C, D...",
        "question_format": "Формат вопросов: 1., 2., 3....",
        "test_instructions": "Инструкция по формату тестов",
        "total_questions": "Всего вопросов",
        "test_grade": "Оценка за тест",
        "grade_system": "Система оценок:",
        "5_point": "5-балльная",
        "100_point": "100% система",
        "auto_grade": "Автоматическая оценка",
        "grade_result": "Оценка:",
        "grade_scale": "Шкала оценки:",
        "grade_description": "Критерии оценивания",
        "grade_5": "Отлично (5)",
        "grade_4": "Хорошо (4)",
        "grade_3": "Удовлетворительно (3)",
        "grade_2": "Неудовлетворительно (2)",
        "grade_rules_math": "Математика: 5 - 0 ошибок, 4 - 1 ошибка, 3 - 2 ошибки, 2 - 3+ ошибок",
        "grade_rules_eng": "Английский: 5 - 0 ошибок, 4 - 1-2 ошибки, 3 - 3-4 ошибки, 2 - 5+ ошибок",
        "grade_rules_test": "Тесты: 5 - 90-100%, 4 - 75-89%, 3 - 60-74%, 2 - 0-59%",
        "grade_comment": "Комментарий к оценке:",
        "view_grades": "📝 Журнал оценок",
        "manual_correction": "✏️ Корректировка ответов",
        "recalculate_results": "🔄 Пересчитать результаты",
        "test_summary": "📊 Результаты тестирования"
    },
    "English": {
        "title": "🚀 EduAI Tutor Pro",
        "subject": "Select subject:",
        "math": "Mathematics",
        "eng": "English",
        "test": "Test (PDF)",
        "check_math": "Check Math",
        "upload": "Upload student work (PDF or image)",
        "example": "Example:",
        "task": "Enter task:",
        "answer": "Enter answer:",
        "correct": "✅ Correct! Excellent work!",
        "incorrect": "❌ Answer is incorrect. Let's analyze:",
        "solution": "📚 Detailed solution:",
        "original": "Original task:",
        "explanation": "Step-by-step explanation:",
        "correct_answer": "Correct answer:",
        "text_rep": "Text representation:",
        "errors": "🔍 Error analysis",
        "progress": "📊 Your progress",
        "history": "Answer history",
        "check_text": "Check text",
        "no_errors": "✅ No errors found!",
        "why_error": "📖 Why is this an error?",
        "gpt_feedback": "🤖 AI recommendation",
        "student_id": "Student ID:",
        "feedback": "Leave feedback:",
        "send_feedback": "Send feedback",
        "ai_math": "🤖 Get AI explanation",
        "ai_math_prompt": "Explain the solution as a tutor",
        "ai_loading": "Generating AI explanation...",
        "ai_error": "AI error, try again later",
        "settings": "⚙️ Settings",
        "api_settings": "🔑 Google Gemini Settings",
        "api_key": "Enter your Google AI API key:",
        "api_help": "Get at aistudio.google.com/app/apikey",
        "key_saved": "Key saved!",
        "mobile_warning": "📱 Use landscape for better experience",
        "connection_success": "✅ Connected to Google Gemini!",
        "connection_fail": "❌ Connection error. Check API key",
        "available_models": "Available models:",
        "invalid_key": "Invalid API key",
        "ai_retry": "🔄 Try again",
        "ai_clear": "❌ Clear response",
        "rate_limit": "⚠️ Rate limit reached. Wait 60 seconds",
        "model_not_found": "⚠️ Model not found. Use another model.",
        "model_selection": "Select AI model:",
        "default_model": "gemini-pro",
        "wait_message": "⏳ Waiting for API limit reset...",
        "retry_countdown": "Retry in:",
        "app_name": "EduAI Tutor Pro",
        "app_description": "AI-Powered Homework Grading System",
        "features": "🌟 Premium Features",
        "pro_tip": "💡 Pro Tip",
        "processing": "🔍 Processing file...",
        "extraction_success": "✅ Text extracted successfully!",
        "extraction_error": "⚠️ Failed to extract text from file",
        "processing_error": "❌ File processing error",
        "pdf_warning": "⚠️ Install PyMuPDF for PDF processing: pip install PyMuPDF",
        "export_results": "📥 Export results",
        "export_btn": "Download report",
        "pdf_processing": "📄 Processing PDF...",
        "ocr_processing": "🖼️ Image recognition...",
        "canvas_integration": "🎓 Canvas LMS Integration",
        "canvas_url": "Canvas URL:",
        "canvas_key": "Canvas API Key:",
        "canvas_help": "Get from Canvas settings",
        "canvas_courses": "Select course:",
        "canvas_assignments": "Select assignment:",
        "canvas_load": "Load from Canvas",
        "canvas_upload": "Send grade to Canvas",
        "canvas_success": "✅ Grade sent to Canvas!",
        "canvas_error": "❌ Canvas integration error",
        "adaptive_learning": "🎯 Personalized Tasks",
        "generate_task": "Generate Task",
        "learning_analytics": "📈 Learning Analytics",
        "topic_analysis": "📚 Topic Analysis",
        "activity_calendar": "🗓️ Activity Calendar",
        "language_select": "🌐 Interface Language:",
        "error_cloud": "☁️ Error Cloud",
        "video_explanation": "🎥 Video Explanation",
        "file_type_error": "⚠️ Unsupported file format. Use PDF or images (PNG, JPG)",
        "test_mode": "📝 Multiple Choice Test",
        "upload_test": "Upload Test PDF",
        "upload_answers": "Upload Student Answers PDF",
        "process_test": "Process Test",
        "question": "Question",
        "student_answer": "Student Answer",
        "correct_answer": "Correct Answer",
        "result": "Result",
        "score": "Score",
        "correct_answers": "Correct Answers",
        "test_results": "Test Results",
        "test_analysis": "📊 Test Analysis",
        "answer_key": "Answer Key",
        "extract_answers": "Extract Answers",
        "test_pdf_error": "Test PDF processing error",
        "answers_pdf_error": "Answers PDF processing error",
        "no_questions_found": "No questions found",
        "no_answers_found": "No answers found",
        "test_processed": "✅ Test processed successfully",
        "answers_processed": "✅ Answers processed successfully",
        "view_test": "View Test",
        "download_results": "Download Results",
        "answer_format": "Answer format: A, B, C, D...",
        "question_format": "Question format: 1., 2., 3....",
        "test_instructions": "Test format instructions",
        "total_questions": "Total Questions",
        "test_grade": "Test Grade",
        "grade_system": "Grading system:",
        "5_point": "5-point scale",
        "100_point": "100% system",
        "auto_grade": "Auto grading",
        "grade_result": "Grade:",
        "grade_scale": "Grading scale:",
        "grade_description": "Grading criteria",
        "grade_5": "Excellent (5)",
        "grade_4": "Good (4)",
        "grade_3": "Satisfactory (3)",
        "grade_2": "Unsatisfactory (2)",
        "grade_rules_math": "Math: 5 - 0 errors, 4 - 1 error, 3 - 2 errors, 2 - 3+ errors",
        "grade_rules_eng": "English: 5 - 0 errors, 4 - 1-2 errors, 3 - 3-4 errors, 2 - 5+ errors",
        "grade_rules_test": "Tests: 5 - 90-100%, 4 - 75-89%, 3 - 60-74%, 2 - 0-59%",
        "grade_comment": "Grade comment:",
        "view_grades": "📝 Gradebook",
        "manual_correction": "✏️ Manual Correction",
        "recalculate_results": "🔄 Recalculate Results",
        "test_summary": "📊 Test Summary"
    }
}

# Глобальная переменная для соединения с БД
cache_conn = None

@contextmanager
def get_db_connection():
    """Контекстный менеджер для безопасной работы с БД"""
    conn = None
    try:
        conn = sqlite3.connect('ai_cache.db', check_same_thread=False, timeout=10)
        conn.row_factory = sqlite3.Row  # Для удобного доступа к колонкам
        yield conn
        conn.commit()
    except sqlite3.Error as e:
        logger.error(f"Ошибка БД: {str(e)}")
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()

def init_cache_db():
    """Инициализация базы данных с улучшенной обработкой ошибок"""
    try:
        with get_db_connection() as conn:
            c = conn.cursor()
            
            # Создание таблиц с индексами для производительности
            c.execute('''CREATE TABLE IF NOT EXISTS responses
                         (query_hash TEXT PRIMARY KEY, 
                          response TEXT, 
                          timestamp DATETIME,
                          model TEXT)''')
            
            c.execute('''CREATE TABLE IF NOT EXISTS learning_history
                         (id INTEGER PRIMARY KEY AUTOINCREMENT,
                          student_id TEXT,
                          subject TEXT,
                          topic TEXT,
                          score REAL,
                          timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
            
            c.execute('''CREATE INDEX IF NOT EXISTS idx_learning_student 
                         ON learning_history(student_id, timestamp DESC)''')
            
            c.execute('''CREATE TABLE IF NOT EXISTS error_history
                         (id INTEGER PRIMARY KEY AUTOINCREMENT,
                          student_id TEXT,
                          subject TEXT,
                          error_type TEXT,
                          count INTEGER DEFAULT 1,
                          last_occurrence DATETIME DEFAULT CURRENT_TIMESTAMP,
                          UNIQUE(student_id, error_type))''')
            
            c.execute('''CREATE INDEX IF NOT EXISTS idx_error_student 
                         ON error_history(student_id)''')
            
            c.execute('''CREATE TABLE IF NOT EXISTS test_results
                         (id INTEGER PRIMARY KEY AUTOINCREMENT,
                          student_id TEXT,
                          test_name TEXT,
                          score REAL,
                          correct_count INTEGER,
                          total_count INTEGER,
                          timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
            
            c.execute('''CREATE TABLE IF NOT EXISTS grades
                         (id INTEGER PRIMARY KEY AUTOINCREMENT,
                          student_id TEXT,
                          subject TEXT,
                          grade INTEGER,
                          percentage REAL,
                          comment TEXT,
                          timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
            
            c.execute('''CREATE INDEX IF NOT EXISTS idx_grades_student 
                         ON grades(student_id, timestamp DESC)''')
            
            logger.info("База данных успешно инициализирована")
            return True
    except Exception as e:
        logger.error(f"Критическая ошибка инициализации БД: {str(e)}")
        st.error(f"Ошибка инициализации базы данных: {str(e)}")
        return False

# Инициализация при старте
if not init_cache_db():
    st.warning("⚠️ Работа с временной базой данных. Данные не сохраняются между сессиями.")

# Инициализация состояний
def init_session_state():
    keys = [
        'math_scores', 'uploaded_text', 'gemini_key', 
        'math_result', 'english_errors', 'ai_status', 'feedback_text',
        'last_request_time', 'available_models', 'selected_model',
        'rate_limit_timer', 'waiting_for_api', 'student_id',
        'api_usage_count', 'api_limit', 'last_api_call',
        'file_processed', 'file_uploaded', 'learning_level',
        'current_subject', 'feedback_sent', 'canvas_url', 
        'canvas_key', 'canvas_courses', 'canvas_assignments',
        'selected_course', 'selected_assignment', 'personalized_task',
        'language', 'activity_data', 'error_history',
        'test_pdf', 'answers_pdf', 'test_data', 'student_answers',
        'test_results', 'test_score', 'show_test_viewer',
        'test_name', 'test_processed', 'answers_processed',
        'grade_system', 'math_grade', 'eng_grade', 'test_grade', 
        'grade_history', 'grade_comment', 'grade_rules',
        'english_score', 'show_gradebook'
    ]
    defaults = {
        'math_scores': [],
        'uploaded_text': "",
        'gemini_key': "",
        'math_result': None,
        'english_errors': [],
        'feedback_text': "",
        'ai_status': {
            'math': {'generating': False, 'response': None, 'error': None},
            'eng': {}
        },
        'last_request_time': 0,
        'available_models': [],
        'selected_model': "gemini-pro",
        'rate_limit_timer': 0,
        'waiting_for_api': False,
        'student_id': "STUD-001",
        'api_usage_count': 0,
        'api_limit': 60,
        'last_api_call': time.time(),
        'file_processed': False,
        'file_uploaded': False,
        'learning_level': "Средний",
        'current_subject': "Математика",
        'feedback_sent': False,
        'canvas_url': "",
        'canvas_key': "",
        'canvas_courses': [],
        'canvas_assignments': [],
        'selected_course': None,
        'selected_assignment': None,
        'personalized_task': {"task": "", "answer": ""},
        'language': "Русский",
        'activity_data': pd.DataFrame(columns=['date', 'activity']),
        'error_history': [],
        'test_pdf': None,
        'answers_pdf': None,
        'test_data': [],
        'student_answers': {},
        'test_results': [],
        'test_score': 0,
        'show_test_viewer': False,
        'test_name': "Тест по математике",
        'test_processed': False,
        'answers_processed': False,
        'grade_system': "5_point",
        'math_grade': None,
        'eng_grade': None,
        'test_grade': None,
        'grade_history': [],
        'grade_comment': "",
        'grade_rules': {
            "math": {"5": 0, "4": 1, "3": 2, "2": 3},
            "eng": {"5": 0, "4": 2, "3": 4, "2": 5},
            "test": {"5": 90, "4": 75, "3": 60, "2": 0}
        },
        'english_score': None,
        'show_gradebook': False
    }
    
    for key in keys:
        if key not in st.session_state:
            st.session_state[key] = defaults.get(key, None)

init_session_state()

# Функция для управления лимитами API
def check_api_rate_limit():
    current_time = time.time()
    elapsed = current_time - st.session_state.last_api_call
    
    if elapsed > 60:
        st.session_state.api_usage_count = 0
        st.session_state.last_api_call = current_time
    
    if st.session_state.api_usage_count >= st.session_state.api_limit:
        remaining = 60 - int(elapsed)
        return False, remaining
    
    return True, 0

# Генерация отчета
def generate_report():
    try:
        report = f"Отчет студента: {st.session_state.student_id}\n"
        report += f"Уровень: {st.session_state.learning_level}\n"
        report += f"Предмет: {st.session_state.current_subject}\n"
        report += f"Дата: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
        
        if st.session_state.math_result:
            report += "=== Результаты по математике ===\n"
            report += f"Задание: {st.session_state.math_result.get('task', '')}\n"
            report += f"Ответ: {st.session_state.math_result.get('answer', '')}\n"
            report += f"Правильно: {'Да' if st.session_state.math_result['is_correct'] else 'Нет'}\n"
            report += f"Объяснение:\n{st.session_state.math_result['explanation']}\n\n"
        
        if st.session_state.english_errors:
            report += "=== Результаты по английскому ===\n"
            report += f"Найдено ошибок: {len(st.session_state.english_errors)}\n"
            for i, error in enumerate(st.session_state.english_errors):
                report += f"Ошибка {i+1}: {error['message']}\n"
                report += f"Рекомендация: {error.get('suggestion', '')}\n\n"
        
        if 'math_scores' in st.session_state and st.session_state.math_scores:
            report += "=== Прогресс ===\n"
            report += f"Всего заданий: {len(st.session_state.math_scores)}\n"
            report += f"Правильных ответов: {sum(st.session_state.math_scores)}\n"
            report += f"Точность: {sum(st.session_state.math_scores)/len(st.session_state.math_scores)*100:.1f}%\n"
        
        if st.session_state.test_results:
            report += "\n=== Результаты теста ===\n"
            report += f"Тест: {st.session_state.test_name}\n"
            report += f"Оценка: {st.session_state.test_score}%\n"
            report += f"Правильных ответов: {sum(1 for r in st.session_state.test_results if r['is_correct'])}/{len(st.session_state.test_results)}\n"
        
        return report
    except Exception as e:
        st.error(f"Ошибка генерации отчета: {str(e)}")
        return "Ошибка при формировании отчета"

@handle_errors("обработки загруженного файла")
def process_uploaded_file(uploaded_file):
    """Улучшенная обработка загруженных файлов с валидацией"""
    try:
        text_content = ""
        
        # Проверка размера файла (макс 10MB)
        file_size = len(uploaded_file.getvalue())
        if file_size > 10 * 1024 * 1024:
            st.error("Файл слишком большой (>10MB). Загрузите файл меньшего размера.")
            return None
        
        if uploaded_file.type == "application/pdf":
            with st.spinner(translations[st.session_state.language]["pdf_processing"]):
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file_path = tmp_file.name
                    
                    doc = fitz.open(tmp_file_path)
                    
                    if len(doc) > 50:
                        st.warning("PDF содержит более 50 страниц. Обрабатываются только первые 50.")
                    
                    for page_num, page in enumerate(doc):
                        if page_num >= 50:  # Ограничение
                            break
                        text_content += page.get_text()
                    
                    doc.close()
                    os.unlink(tmp_file_path)
                    
                    logger.info(f"PDF обработан: {len(doc)} страниц, {len(text_content)} символов")
                    
                except Exception as e:
                    logger.error(f"Ошибка обработки PDF: {str(e)}")
                    st.error(f"Не удалось обработать PDF: {str(e)}")
                    return None
                    
        elif uploaded_file.type in ["image/png", "image/jpeg", "image/jpg"]:
            with st.spinner(translations[st.session_state.language]["ocr_processing"]):
                try:
                    image = Image.open(uploaded_file)
                    
                    # Проверка размера изображения
                    if image.size[0] * image.size[1] > 10000000:  # 10 мегапикселей
                        st.warning("Изображение слишком большое, изменяем размер...")
                        image.thumbnail((4000, 4000))
                    
                    image = image.convert('L')
                    image = image.point(lambda x: 0 if x < 128 else 255, '1')
                    text_content = pytesseract.image_to_string(image, lang='rus+eng')
                    
                    logger.info(f"Изображение обработано: {len(text_content)} символов")
                    
                except Exception as e:
                    logger.error(f"Ошибка OCR: {str(e)}")
                    st.error(f"Не удалось распознать текст: {str(e)}")
                    return None
        else:
            st.warning(translations[st.session_state.language]["file_type_error"])
            return None
        
        if not text_content.strip():
            st.warning("Файл не содержит извлекаемого текста")
            return None
            
        return text_content.strip()
        
    except Exception as e:
        logger.error(f"Критическая ошибка обработки файла: {str(e)}", exc_info=True)
        st.error(f"Произошла ошибка: {str(e)}")
        return None
# Генерация календаря активности
def generate_activity_calendar():
    try:
        if st.session_state.activity_data.empty:
            dates = pd.date_range(end=datetime.datetime.today(), periods=30, freq='D')
            activity = np.random.randint(0, 10, size=30)
            st.session_state.activity_data = pd.DataFrame({'date': dates, 'activity': activity})
        
        df = st.session_state.activity_data.copy()
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        
        current_year = datetime.datetime.now().year
        current_month = datetime.datetime.now().month
        
        month_days = calendar.monthrange(current_year, current_month)[1]
        days = [datetime.date(current_year, current_month, day) for day in range(1, month_days+1)]
        
        activity_map = {}
        for date, act in zip(df['date'], df['activity']):
            if date.year == current_year and date.month == current_month:
                activity_map[date.day] = act
        
        fig = go.Figure()
        
        for day in days:
            day_activity = activity_map.get(day.day, 0)
            color = f'rgba(106, 17, 203, {min(1.0, day_activity/10)})'
            
            fig.add_trace(go.Scatter(
                x=[day.weekday()],
                y=[5 - day.isocalendar()[1] + day.isocalendar()[1] - days[0].isocalendar()[1]],
                mode='markers',
                marker=dict(
                    size=30,
                    color=color,
                    line=dict(width=1, color='rgba(0,0,0,0.2)')
                ),
                text=f"{day.strftime('%d %b')}: {day_activity} задач",
                hoverinfo='text',
                showlegend=False
            ))
        
        fig.update_xaxes(
            tickvals=[0, 1, 2, 3, 4, 5, 6],
            ticktext=['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Вс'],
            range=[-0.5, 6.5]
        )
        
        fig.update_yaxes(
            visible=False, 
            range=[-1, 6],
            scaleanchor="x", 
            scaleratio=1
        )
        
        fig.update_layout(
            title=f'Активность за {calendar.month_name[current_month]} {current_year}',
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=40, b=0),
            xaxis=dict(fixedrange=True),
            yaxis=dict(fixedrange=True)
        )
        
        return fig
    except Exception as e:
        st.error(f"Ошибка создания календаря: {str(e)}")
        return None

# Генерация облака ошибок
def generate_error_cloud():
    try:
        if not st.session_state.error_history:
            return None
            
        error_text = " ".join([f"{error['error_type']} " * error['count'] 
                              for error in st.session_state.error_history])
        
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white',
            colormap='viridis',
            max_words=50
        ).generate(error_text)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        return fig
    except Exception as e:
        st.error(f"Ошибка создания облака ошибок: {str(e)}")
        return None
if GRADING_SYSTEM_AVAILABLE:  # Проверяем, есть ли новая система
    grading_system = GradingSystem()  # Создаем объект
    grade_info = grading_system.grade_test(st.session_state.test_results)
    
    # Получаем не просто оценку, а детальный анализ:
    grade_value = grade_info['grade']  # Оценка: 5, 4, 3, 2
    feedback = grade_info['feedback']  # "Отличный результат! Демонстрирует..."
# Загрузка Lottie анимации
def load_lottie(url):
    try:
        if url.startswith('http'):
            r = requests.get(url)
            if r.status_code == 200:
                return r.json()
        return None
    except:
        return None
# ============ ДОБАВЬТЕ ЭТУ ФУНКЦИЮ ПЕРЕД extract_questions_from_pdf ============
def diagnose_pdf(pdf_file):
    """Диагностика PDF для отладки"""
    try:
        with pdfplumber.open(pdf_file) as pdf:
            st.write(f"📄 **Всего страниц:** {len(pdf.pages)}")
            
            for i, page in enumerate(pdf.pages[:2], 1):
                text = page.extract_text()
                st.write(f"\n**📃 Страница {i}:**")
                st.code(text if text else "⚠️ Текст не найден", language="text")
    except Exception as e:
        st.error(f"Ошибка диагностики: {str(e)}")

# Здесь идёт ваша функция extract_questions_from_pdf...
# ========== УЛУЧШЕННЫЕ ФУНКЦИИ ДЛЯ ТЕСТИРОВАНИЯ ==========

def extract_questions_from_pdf(pdf_file):
    """
    Улучшенное извлечение вопросов из PDF с множественными паттернами
    """
    questions = []
    logger.info(f"Начало обработки PDF теста")
    
    try:
        with pdfplumber.open(pdf_file) as pdf:
            full_text = ""
            
            # Собираем весь текст
            for page_num, page in enumerate(pdf.pages, 1):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        full_text += f"\n--- PAGE {page_num} ---\n{page_text}"
                except Exception as e:
                    logger.warning(f"Ошибка чтения страницы {page_num}: {e}")
                    continue
            
            if not full_text.strip():
                st.error("PDF файл не содержит извлекаемого текста")
                return []
            
            # Паттерны для поиска вопросов (от более специфичных к общим)
            question_patterns = [
                # Паттерн 1: "Вопрос 1: текст"
                r'Вопрос\s*(\d+)[:\s]+(.*?)(?=\nВопрос\s*\d+|\n[A-D][.)]\s|$)',
                # Паттерн 2: "1. текст вопроса"
                r'(?:^|\n)(\d+)\.\s+(.*?)(?=\n\d+\.|\n[A-D][.)]\s|$)',
                # Паттерн 3: "1) текст вопроса"
                r'(?:^|\n)(\d+)\)\s+(.*?)(?=\n\d+\)|\n[A-D][.)]\s|$)',
                # Паттерн 4: просто номер и текст
                r'(?:^|\n)(\d+)\s+(.*?)(?=\n\d+\s|\n[A-D][.)]\s|$)'
            ]
            
            found_questions = {}
            
            for pattern in question_patterns:
                matches = re.finditer(pattern, full_text, re.DOTALL | re.MULTILINE)
                
                for match in matches:
                    q_num = int(match.group(1))
                    question_text = match.group(2).strip()
                    
                    # Пропускаем слишком короткие вопросы
                    if len(question_text) < 10:
                        continue
                    
                    # Ищем варианты ответов после вопроса
                    options = {}
                    options_pattern = r'([A-D])[.)]\s+([^\n]+?)(?=\n[A-D][.)]|\n\d+[.)]|\nОтвет|\nПравильный|\n*$)'
                    
                    # Ищем в тексте после текущего вопроса
                    search_start = match.end()
                    search_text = full_text[search_start:search_start+1000]
                    
                    for opt_letter, opt_text in re.findall(options_pattern, search_text, re.MULTILINE):
                        options[opt_letter.upper()] = opt_text.strip()
                    
                    # Ищем правильный ответ
                    correct_answer = ""
                    answer_patterns = [
                        r'Ответ[:\s]*([A-D])',
                        r'Правильный\s+ответ[:\s]*([A-D])',
                        r'Верный\s+ответ[:\s]*([A-D])',
                        r'Correct[:\s]*([A-D])'
                    ]
                    
                    for ans_pattern in answer_patterns:
                        ans_match = re.search(ans_pattern, search_text, re.IGNORECASE)
                        if ans_match:
                            correct_answer = ans_match.group(1).upper()
                            break
                    
                    # Сохраняем вопрос (избегаем дубликатов)
                    if options and q_num not in found_questions:
                        found_questions[q_num] = {
                            'number': q_num,
                            'question': question_text[:500],  # Ограничиваем длину
                            'options': options,
                            'correct_answer': correct_answer,
                            'page': full_text[:match.start()].count('--- PAGE')
                        }
                
                # Если нашли достаточно вопросов, останавливаемся
                if len(found_questions) >= 5:
                    break
            
            questions = sorted(found_questions.values(), key=lambda x: x['number'])
            
            logger.info(f"Извлечено {len(questions)} вопросов из PDF")
            
            if not questions:
                st.warning("⚠️ Вопросы не найдены. Проверьте формат PDF файла.")
                st.info("""
                **Требования к формату:**
                - Вопросы должны быть пронумерованы: 1., 2., 3. или Вопрос 1:
                - Варианты ответов: A. B. C. D.
                - Правильный ответ: "Ответ: A" или "Правильный ответ: A"
                """)
            
            return questions
            
    except Exception as e:
        logger.error(f"Ошибка обработки PDF: {str(e)}", exc_info=True)
        st.error(f"Ошибка обработки теста: {str(e)}")
        return []
    

    
    # ========= ДОБАВЬТЕ ЭТУ КНОПКУ =========
    if st.button("🔬 Показать что внутри PDF", key="diagnose_btn"):
        diagnose_pdf(st.session_state.test_pdf)
    # ========================================
st.divider()

def extract_answers_from_pdf(pdf_file):
    """Извлечение ответов с проверкой"""
    answers = {}
    
    try:
        # Проверяем что файл не None
        if pdf_file is None:
            st.error("Файл не загружен")
            return {}
        
        # Сбрасываем указатель (на всякий случай)
        try:
            pdf_file.seek(0)
        except:
            pass
        
        with pdfplumber.open(pdf_file) as pdf:
            full_text = ""
            
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    full_text += text + "\n"
            
            st.write("**📄 Текст из PDF:**")
            st.code(full_text[:500] if full_text else "❌ Пусто", language="text")
            
            if not full_text.strip():
                st.error("PDF пустой или сканированный. Используйте ручной ввод ниже.")
                return {}
            
            # Заменяем русские буквы
            replacements = {
                'А': 'A', 'а': 'A',
                'В': 'B', 'в': 'B',
                'С': 'C', 'с': 'C',
                'Д': 'D', 'д': 'D'
            }
            
            for rus, lat in replacements.items():
                full_text = full_text.replace(rus, lat)
            
            # Ищем ответы
            pattern = r'(\d+)\s*[.\):;\-–—]*\s*([A-D])\b'
            matches = re.findall(pattern, full_text, re.IGNORECASE)
            
            for num_str, ans in matches:
                answers[int(num_str)] = ans.upper()
            
            if answers:
                st.success(f"✅ Найдено {len(answers)} ответов")
                st.json(answers)
            else:
                st.warning("⚠️ Ответы не распознаны. Используйте ручной ввод.")
            
            return answers
            
    except Exception as e:
        st.error(f"Ошибка: {e}")
        return {}


def grade_test(test_questions, student_answers):
    """Проверка теста и подсчёт результатов"""
    results = []
    correct_count = 0
    
    for question in test_questions:
        q_num = question['number']
        correct_answer = question['correct_answer']
        student_answer = student_answers.get(q_num, None)
        
        is_correct = (student_answer == correct_answer) if student_answer else False
        
        if is_correct:
            correct_count += 1
        
        results.append({
            'number': q_num,
            'question': question['question'],
            'student_answer': student_answer or '—',
            'correct_answer': correct_answer,
            'is_correct': is_correct,
            'options': question['options']
        })
    
    total = len(test_questions)
    score = (correct_count / total * 100) if total > 0 else 0
    
    return results, score, correct_count

# После extract_answers_from_pdf добавьте:

st.markdown("---")
st.subheader("✏️ Ручной ввод (если PDF не работает)")

manual_input = st.text_area(
    "Введите ответы студента",
    placeholder="Примеры форматов:\n1.A\n2.B\n3.C\n\nили\n1. А\n2. В\n3. С",
    height=150,
    key="manual_answer_input"
)

if st.button("✅ Проверить эти ответы", key="check_manual"):
    if not manual_input.strip():
        st.error("Введите ответы!")
    else:
        # Заменяем русские буквы
        text = manual_input
        replacements = {'А':'A', 'а':'A', 'В':'B', 'в':'B', 'С':'C', 'с':'C', 'Д':'D', 'д':'D'}
        for rus, lat in replacements.items():
            text = text.replace(rus, lat)
        
        # Извлекаем ответы
        manual_ans = {}
        pattern = r'(\d+)\s*[.\):;\-–—\s]*([A-D])\b'
        matches = re.findall(pattern, text, re.IGNORECASE)
        
        for num_str, ans in matches:
            manual_ans[int(num_str)] = ans.upper()
        
        if manual_ans:
            st.session_state.student_answers = manual_ans
            st.session_state.answers_processed = True
            
            st.success(f"✅ Распознано ответов: {len(manual_ans)}")
            st.json(manual_ans)
            
            # ПРОВЕРЯЕМ ТЕСТ
            if st.session_state.get('test_data'):
                results, score, correct = grade_test(
                    st.session_state.test_data,
                    manual_ans
                )
                
                st.session_state.test_results = results
                st.session_state.test_score = score
                
                # Показываем результат
                st.markdown("---")
                st.header(f"📊 Результат: {score:.1f}%")
                st.progress(score/100)
                st.write(f"**Правильных ответов:** {correct} из {len(st.session_state.test_data)}")
                
                # Детали по вопросам
                for res in results:
                    icon = "✅" if res['is_correct'] else "❌"
                    st.write(f"{icon} **Вопрос {res['number']}:** {res['question'][:50]}...")
                    st.write(f"   Ответ студента: **{res['student_answer']}** | Правильный: **{res['correct_answer']}**")
                
                st.balloons()
            else:
                st.warning("Сначала загрузите тест с вопросами!")
        else:
            st.error("Не удалось распознать ответы. Проверьте формат.")

# === РУЧНОЙ ВВОД ОТВЕТОВ ===
st.markdown("---")
st.subheader("✏️ Ручной ввод ответов студента")

with st.expander("📝 Введите ответы вручную"):
    manual_input = st.text_area(
        "Вставьте ответы (каждый с новой строки)",
        placeholder="1.A\n2.B\n3.C\n4.D\n\nИли:\n1. A\n2. B\n3. C",
        height=150,
        key="manual_answers_area"
    )
    
    if st.button("✅ Проверить эти ответы", key="submit_manual_answers"):
        # Преобразуем русские в латинские
        replacements = {'А':'A', 'а':'A', 'В':'B', 'в':'B', 'С':'C', 'с':'C', 'Д':'D', 'д':'D'}
        for rus, lat in replacements.items():
            manual_input = manual_input.replace(rus, lat)
        
        manual_ans = {}
        
        # Парсим разные форматы
        for line in manual_input.strip().split('\n'):
            # Ищем: число + любой разделитель + буква A-D
            match = re.search(r'(\d+)\s*[.):;\-–—]*\s*([A-D])', line, re.IGNORECASE)
            if match:
                q_num = int(match.group(1))
                answer = match.group(2).upper()
                manual_ans[q_num] = answer
        
        if manual_ans:
            st.session_state.student_answers = manual_ans
            st.session_state.answers_processed = True
            
            # Проверяем тест
            if st.session_state.test_data:
                results, score, correct = grade_test(
                    st.session_state.test_data,
                    st.session_state.student_answers
                )
                st.session_state.test_results = results
                st.session_state.test_score = score
                
                st.success(f"✅ Проверено! Правильных ответов: {correct}/{len(st.session_state.test_data)}")
                st.balloons()
                st.rerun()
        else:
            st.error("❌ Не удалось распознать ответы. Проверьте формат.")


def save_test_results_to_db(student_id, test_name, score, correct_count, total_count):
    """Сохранение результатов теста в БД"""
    try:
        with get_db_connection() as conn:
            c = conn.cursor()
            c.execute('''INSERT INTO test_results 
                         (student_id, test_name, score, correct_count, total_count) 
                         VALUES (?, ?, ?, ?, ?)''',
                      (student_id, test_name, score, correct_count, total_count))
            logger.info(f"Результаты теста сохранены для {student_id}")
            return True
    except Exception as e:
        logger.error(f"Ошибка сохранения результатов: {str(e)}")
        return False


def grade_test(test_data, student_answers):
    results = []
    correct_count = 0
    
    for question in test_data:
        q_num = question['number']
        student_answer = student_answers.get(q_num, "").upper()
        correct_answer = question.get('correct_answer', '').upper()
        
        # Проверка на валидность ответа
        if student_answer and student_answer in ['A', 'B', 'C', 'D']:
            is_correct = student_answer == correct_answer
        else:
            is_correct = False
        
        if is_correct:
            correct_count += 1
        
        results.append({
            'number': q_num,
            'question': question['question'],
            'options': question['options'],
            'student_answer': student_answer,
            'correct_answer': correct_answer,
            'is_correct': is_correct,
            'page': question['page']
        })
    
    score = round(100.0 * correct_count / len(test_data)) if test_data else 0
    return results, score, correct_count

# Функция для создания PDF с результатами
def create_results_pdf(results, score, correct_count, total_questions):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Создаем пользовательские стили с уникальными именами
    custom_styles = {
        'Custom_Title': ParagraphStyle(
            name='Custom_Title',
            parent=styles['Heading1'],
            fontSize=18,
            alignment=TA_CENTER,
            spaceAfter=20
        ),
        'Custom_Header': ParagraphStyle(
            name='Custom_Header',
            parent=styles['Heading2'],
            fontSize=14,
            alignment=TA_LEFT,
            spaceAfter=10
        ),
        'Custom_QuestionText': ParagraphStyle(
            name='Custom_QuestionText',
            parent=styles['BodyText'],
            fontSize=12,
            spaceAfter=5
        ),
        'Custom_OptionText': ParagraphStyle(
            name='Custom_OptionText',
            parent=styles['BodyText'],
            fontSize=10,
            leftIndent=20,
            spaceAfter=3
        )
    }
    
    # Содержимое PDF
    content = []
    
    # Заголовок
    title = Paragraph(f"Результаты теста: {score}%", custom_styles['Custom_Title'])
    content.append(title)
    
    # Сводка
    summary = Paragraph(
        f"<b>Студент:</b> {st.session_state.student_id}<br/>"
        f"<b>Тест:</b> {st.session_state.test_name}<br/>"
        f"<b>Правильных ответов:</b> {correct_count}/{total_questions}<br/>"
        f"<b>Оценка:</b> {score}%<br/><br/>",
        styles['BodyText']
    )
    content.append(summary)
    
    # Результаты по вопросам
    header = Paragraph("Детализация ответов:", custom_styles['Custom_Header'])
    content.append(header)
    
    # Таблица с результатами
    result_data = [["№", "Вопрос", "Ответ студента", "Правильный ответ", "Результат"]]
    
    for res in results:
        status = "✅ Верно" if res['is_correct'] else "❌ Неверно"
        result_data.append([
            str(res['number']),
            res['question'][:100] + "..." if len(res['question']) > 100 else res['question'],
            res['student_answer'],
            res['correct_answer'],
            status
        ])
    
    table = Table(result_data, colWidths=[30, 200, 60, 80, 60])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
    ]))
    
    content.append(table)
    content.append(Spacer(1, 20))
    
    # Подробные ответы
    for res in results:
        content.append(Paragraph(f"<b>Вопрос {res['number']}:</b> {res['question']}", custom_styles['Custom_QuestionText']))
        
        for option, text in res['options'].items():
            prefix = ""
            if option == res['correct_answer']:
                prefix = "<b>✓ </b>"
            elif option == res['student_answer']:
                prefix = "<b>✗ </b>"
                
            content.append(Paragraph(f"{prefix}{option}. {text}", custom_styles['Custom_OptionText']))
        
        content.append(Paragraph(
            f"<b>Ответ студента:</b> {res['student_answer']} &nbsp; "
            f"<b>Правильный ответ:</b> {res['correct_answer']} &nbsp; "
            f"<b>Результат:</b> {'✅ Верно' if res['is_correct'] else '❌ Неверно'}",
            custom_styles['Custom_OptionText']
        ))
        
        content.append(Spacer(1, 15))
    
    # Создаем PDF
    doc.build(content)
    buffer.seek(0)
    return buffer

def calculate_grade(subject, result):
    """
    Улучшенная функция оценивания с использованием новой системы
    """
    if GRADING_SYSTEM_AVAILABLE:
        grading_system = GradingSystem()
        
        if subject == "math":
            grade_result = grading_system.grade_math(result)
            return grade_result['grade'], grade_result.get('percentage', 0)
        
        elif subject == "eng":
            word_count = result.get('word_count', len(str(result).split()))
            errors = result if isinstance(result, list) else []
            grade_result = grading_system.grade_english(errors, word_count)
            return grade_result['grade'], grade_result.get('percentage', 0)
        
        elif subject == "test":
            grade_result = grading_system.grade_test(result.get('results', []))
            return grade_result['grade'], grade_result.get('percentage', 0)
    
    # Fallback к старой системе
    rules = st.session_state.grade_rules.get(subject, {})
    
    if subject == "math":
        error_count = 0 if result.get('is_correct', False) else 1
        for grade, max_errors in rules.items():
            if error_count <= max_errors:
                return int(grade), 100 if error_count == 0 else 75
    
    elif subject == "eng":
        error_count = len(result) if isinstance(result, list) else 0
        for grade, max_errors in rules.items():
            if error_count <= max_errors:
                return int(grade), max(0, 100 - error_count * 5)
    
    elif subject == "test":
        score = result.get('score', 0)
        for grade, min_score in rules.items():
            if score >= min_score:
                return int(grade), score
    
    return 2, 0


def add_to_grade_history(subject, grade, comment, percentage=None):
    """Сохранение оценки в историю с улучшенной обработкой"""
    entry = {
        "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "subject": subject,
        "grade": grade,
        "percentage": percentage,
        "comment": comment,
        "student": st.session_state.student_id
    }
    st.session_state.grade_history.append(entry)
    
    try:
        with get_db_connection() as conn:
            c = conn.cursor()
            c.execute('''INSERT INTO grades 
                         (student_id, subject, grade, percentage, comment) 
                         VALUES (?, ?, ?, ?, ?)''',
                      (st.session_state.student_id, subject, grade, percentage, comment))
            logger.info(f"Оценка сохранена: {subject} -{grade} баллов")
        return True
    except Exception as e:
        logger.error(f"Ошибка сохранения оценки: {str(e)}")
        st.warning("Оценка не сохранена в базу данных")
        return False
    


# Сайдбар
with st.sidebar:
    trans = translations[st.session_state.language]
    
    st.image("https://cdn-icons-png.flaticon.com/512/201/201623.png", width=80)
    st.markdown(f"### {trans['app_name']}")
    st.caption(trans["app_description"])
    
    st.session_state.language = st.selectbox(
        trans["language_select"], 
        ["Русский", "English"],
        index=0 if st.session_state.language == "Русский" else 1
    )
    trans = translations[st.session_state.language]
    
    st.divider()
    
    st.subheader("👤 Профиль студента")
    student_id = st.text_input(trans["student_id"], st.session_state.student_id)
    st.session_state.student_id = student_id
    
    level = st.selectbox("Уровень обучения", ["Начальный", "Средний", "Продвинутый"], 
                         index=["Начальный", "Средний", "Продвинутый"].index(st.session_state.learning_level))
    st.session_state.learning_level = level
    
    st.divider()
    
    st.subheader(f"**{trans['canvas_integration']}**")
    canvas_url = st.text_input(
        trans["canvas_url"], 
        value=st.session_state.canvas_url,
        help="Пример: https://yourinstitution.instructure.com"
    )
    canvas_key = st.text_input(
        trans["canvas_key"], 
        type="password",
        value=st.session_state.canvas_key
    )
    
    if canvas_url and canvas_key and (canvas_url != st.session_state.canvas_url or canvas_key != st.session_state.canvas_key):
        st.session_state.canvas_url = canvas_url
        st.session_state.canvas_key = canvas_key
        canvas_api = CanvasAPI(st.session_state.canvas_url, st.session_state.canvas_key)
        success, user_name = canvas_api.test_connection()
        if success:
            st.success(f"✅ Подключено к Canvas: {user_name}")
            success, courses = canvas_api.get_courses()
            if success:
                st.session_state.canvas_courses = courses
                st.session_state.selected_course = None
                st.session_state.canvas_assignments = []
            else:
                st.error(trans["canvas_error"])
        else:
            st.error(trans["canvas_error"])
    
    if st.session_state.canvas_courses:
        course_names = [course['name'] for course in st.session_state.canvas_courses]
        selected_course_name = st.selectbox(
            trans["canvas_courses"], 
            course_names,
            index=0
        )
        selected_course = next(course for course in st.session_state.canvas_courses if course['name'] == selected_course_name)
        
        if st.session_state.selected_course != selected_course['id']:
            st.session_state.selected_course = selected_course['id']
            canvas_api = CanvasAPI(st.session_state.canvas_url, st.session_state.canvas_key)
            success, assignments = canvas_api.get_assignments(selected_course['id'])
            if success:
                st.session_state.canvas_assignments = assignments
                st.session_state.selected_assignment = None
            else:
                st.session_state.canvas_assignments = []
        
        if st.session_state.canvas_assignments:
            assignment_names = [assignment['name'] for assignment in st.session_state.canvas_assignments]
            selected_assignment_name = st.selectbox(
                trans["canvas_assignments"], 
                assignment_names,
                index=0
            )
            st.session_state.selected_assignment = next(
                assignment for assignment in st.session_state.canvas_assignments 
                if assignment['name'] == selected_assignment_name
            )
            
            if st.button(trans["canvas_load"]):
                st.session_state.uploaded_text = st.session_state.selected_assignment['description']
                st.session_state.file_processed = True
                st.success(f"Задание '{selected_assignment_name}' загружено!")
    
    st.divider()
    
    st.subheader(f"**{trans['api_settings']}**")
    
    api_key = st.text_input(
        trans["api_key"], 
        type="password",
        help=trans["api_help"],
        value=st.session_state.get('gemini_key', '')
    )
    
    if api_key and api_key != st.session_state.gemini_key:
        st.session_state.gemini_key = api_key
        st.success(trans["key_saved"])
        try:
            genai.configure(api_key=api_key)
            models = genai.list_models()
            model_names = [model.name.split('/')[-1] for model in models 
                          if 'generateContent' in model.supported_generation_methods]
            st.session_state.available_models = model_names
            
            if 'gemini-pro' in model_names:
                st.session_state.selected_model = 'gemini-pro'
            elif model_names:
                st.session_state.selected_model = model_names[0]
            
            st.session_state.ai_status['math']['error'] = None
        except Exception as e:
            st.error(f"Ошибка: {str(e)}")
    
    if st.session_state.available_models:
        selected_model = st.selectbox(
            trans["model_selection"],
            options=st.session_state.available_models,
            index=st.session_state.available_models.index(
                st.session_state.selected_model) if st.session_state.selected_model in st.session_state.available_models else 0
        )
        st.session_state.selected_model = selected_model
        st.info(f"**Выбрана модель:** {selected_model}")
    
    usage_percent = min(100, st.session_state.api_usage_count / st.session_state.api_limit * 100)
    st.progress(usage_percent / 100, 
               text=f"Использовано API: {st.session_state.api_usage_count}/{st.session_state.api_limit}")
    
    if st.button("🔍 Проверить подключение", use_container_width=True, key="test_connection"):
        if st.session_state.gemini_key:
            try:
                models = genai.list_models()
                model_names = [model.name.split('/')[-1] for model in models 
                              if 'generateContent' in model.supported_generation_methods]
                st.session_state.available_models = model_names
                st.success(trans["connection_success"])
                st.info(f"{trans['available_models']} {', '.join(model_names[:3])}{'...' if len(model_names) > 3 else ''}")
            except Exception as e:
                error_msg = str(e)
                if "API_KEY" in error_msg or "401" in error_msg:
                    st.error(trans["invalid_key"])
                elif "429" in error_msg:
                    st.error(trans["rate_limit"])
                elif "404" in error_msg:
                    st.error(trans["model_not_found"])
                else:
                    st.error(f"{trans['connection_fail']}: {error_msg}")
        else:
            st.warning("Введите API ключ сначала")
    
    st.divider()
    st.subheader(trans["features"])
    st.markdown("""
    - 🧠 AI-объяснения
    - 📈 Система прогресса
    - 🏆 Награды за достижения
    - 🎯 Персональные рекомендации
    - 🔒 Безопасная обработка данных
    - ⚡ Оптимизация лимитов API
    - 📥 Экспорт результатов
    - 📊 Аналитика успеваемости
    - 🎓 Интеграция с Canvas LMS
    - 🗓️ Календарь активности
    - 📝 Проверка тестов PDF
    """)

    st.divider()
    
    st.subheader(f"**{trans['grade_system']}**")
    grade_system = st.radio(
        trans["grade_system"],
        [trans["5_point"], trans["100_point"]],
        index=0 if st.session_state.grade_system == "5_point" else 1
    )
    st.session_state.grade_system = "5_point" if grade_system == trans["5_point"] else "100_point"
    
    st.checkbox(trans["auto_grade"], True, help=trans["grade_description"])
    
    st.divider()
    
    if st.button(trans["view_grades"], use_container_width=True):
        st.session_state.show_gradebook = not st.session_state.get('show_gradebook', False)
    
    if st.session_state.get('show_gradebook', False):
        st.subheader("📝 Журнал оценок")
        if st.session_state.grade_history:
            grade_df = pd.DataFrame(st.session_state.grade_history)
            st.dataframe(grade_df, hide_index=True, use_container_width=True)
            
            fig = px.pie(grade_df, names='grade', title='Распределение оценок')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("История оценок пуста")
st.divider()

if st.button("📊 Аналитика прогресса", use_container_width=True):
    st.session_state.show_analytics = not st.session_state.get('show_analytics', False)

if st.session_state.get('show_analytics', False) and GRADING_SYSTEM_AVAILABLE:
    st.subheader("📊 Аналитика обучения")
    
    if st.session_state.grade_history:
        df_grades = pd.DataFrame(st.session_state.grade_history)
        
        # Анализ тренда
        analytics = LearningAnalytics()
        trend_info = analytics.calculate_progress_trend(df_grades)
        
        # Отображение тренда
        trend_emoji = {
            'improving': '📈',
            'declining': '📉',
            'stable': '➡️',
            'insufficient_data': '❓'
        }
        
        st.metric(
            "Тренд обучения",
            f"{trend_emoji.get(trend_info['trend'], '❓')} {trend_info['trend']}",
            f"Средняя оценка: {trend_info.get('overall_avg', 0)}"
        )
        
        if trend_info['trend'] != 'insufficient_data':
            col1, col2 = st.columns(2)
            col1.metric("Недавние результаты", f"{trend_info.get('recent_avg', 0):.1f}")
            col2.metric("Лучшая оценка", trend_info.get('best_grade', 0))
        
        # График прогресса
        fig = px.line(
            df_grades,
            x='date',
            y='grade',
            title='Прогресс оценок',
            markers=True
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    st.markdown("""
    <div style="text-align: center;">
        <p>Версия 6.0.0</p>
        <p>© 2024 EduAI Tutor Pro</p>
    </div>
    """, unsafe_allow_html=True)

# Основной интерфейс
st.title(f"✨ {trans['title']}")
st.markdown(f"**{trans['app_description']}**")

lottie_hello = load_lottie("https://assets9.lottiefiles.com/packages/lf20_vybwn7df.json")
if lottie_hello:
    st_lottie(lottie_hello, height=200, key="hello")

if st.session_state.waiting_for_api:
    current_time = time.time()
    remaining = int(60 - (current_time - st.session_state.rate_limit_timer))
    
    if remaining > 0:
        with st.empty():
            st.warning(f"{trans['wait_message']} {trans['retry_countdown']} {remaining} сек")
            time.sleep(1)
            st.rerun()
    else:
        st.session_state.waiting_for_api = False

with st.expander("📁 Загрузить работу", expanded=True):
    uploaded_file = st.file_uploader(trans["upload"], type=["pdf", "png", "jpg", "jpeg"], 
                                    key="file_uploader")
    
    if uploaded_file and not st.session_state.file_processed:
        text_content = process_uploaded_file(uploaded_file)
        if text_content:
            st.session_state.uploaded_text = text_content
            st.session_state.file_processed = True
            st.session_state.file_uploaded = True
            st.success(trans["extraction_success"])
    
    if st.session_state.file_processed and st.session_state.file_uploaded:
        st.text_area("Извлеченный текст:", value=st.session_state.uploaded_text, height=150)
        if st.button("Очистить загруженный файл", type="secondary"):
            st.session_state.file_processed = False
            st.session_state.file_uploaded = False
            st.session_state.uploaded_text = ""
            st.rerun()

st.subheader("📚 Выберите предмет")
subject_options = [
    trans["math"], 
    trans["eng"],
    trans["test"]
]
subject = st.radio(
    trans["subject"], 
    subject_options, 
    horizontal=True,
    index=0 if st.session_state.current_subject == trans["math"] else 
           1 if st.session_state.current_subject == trans["eng"] else 2,
    label_visibility="collapsed"
)
st.session_state.current_subject = subject

if subject == trans["test"]:
    trans = translations[st.session_state.language]
    
    st.subheader("📝 Система автоматической проверки тестов")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📄 Тестовый документ")
        st.session_state.test_name = st.text_input("Название теста", st.session_state.test_name)
        uploaded_test = st.file_uploader(
            trans["upload_test"], 
            type=["pdf"],
            key="test_uploader"
        )
        
        if uploaded_test:
            st.session_state.test_pdf = uploaded_test
            
    # ========= ДОБАВЬТЕ ЭТУ КНОПКУ =========
            if st.button("🔬 Показать что внутри PDF", key="diagnose_btn"):
                diagnose_pdf(st.session_state.test_pdf)
    # ========================================
        if st.button("📥 Обработать тест", key="process_test_btn", disabled=not st.session_state.test_pdf):
            with st.spinner("🔍 Анализ теста..."):
                # ИЗМЕНЕНИЕ 1: Используем улучшенную функцию
                st.session_state.test_data = extract_questions_from_pdf(st.session_state.test_pdf)
                
                if st.session_state.test_data:
                    st.success(f"✅ Найдено {len(st.session_state.test_data)} вопросов")
                    st.session_state.show_test_viewer = True
                    st.session_state.test_processed = True
                    
                    # ДОБАВЛЕНИЕ: Показываем превью найденных вопросов
                    with st.expander("👁️ Превью найденных вопросов", expanded=False):
                        for q in st.session_state.test_data[:3]:  # Показываем первые 3
                            st.write(f"**Вопрос {q['number']}:** {q['question'][:100]}...")
                            st.write(f"Вариантов: {len(q['options'])}, Правильный ответ: {q['correct_answer'] or '❌ Не найден'}")
                            st.divider()
                else:
                    st.warning(trans["no_questions_found"])
    
    with col2:
        # === ЗАГРУЗКА ОТВЕТОВ СТУДЕНТА ===
        st.subheader("📝 Шаг 2: Загрузите ответы студента")

uploaded_answers = st.file_uploader(
    "Загрузите PDF с ответами студента", 
    type=['pdf'],
    key="student_answers_uploader"
)

if uploaded_answers is not None:
    # Сохраняем в session_state
    st.session_state.answers_pdf = uploaded_answers
    
    st.info("📄 Файл загружен. Извлекаем ответы...")
    
    # Извлекаем ответы
    answers = extract_answers_from_pdf(uploaded_answers)
    
    if answers:
        st.session_state.student_answers = answers
        st.session_state.answers_processed = True
        st.success(f"✅ Найдено {len(answers)} ответов")
    else:
        st.warning("⚠️ Ответы не распознаны. Используйте ручной ввод ниже.")
# === РУЧНОЙ ВВОД (если PDF не работает) ===
st.markdown("---")
st.subheader("✏️ Или введите ответы вручную")

manual_input = st.text_area(
    "Введите ответы студента (каждый с новой строки)",
    placeholder="1.A\n2.B\n3.C\n4.D\n\nили\n1. А\n2. В\n3. С",
    height=150,
    key="manual_answers_text"
)

if st.button("✅ Проверить ответы", key="submit_answers_btn"):
    if not manual_input.strip():
        st.error("❌ Введите ответы!")
    else:
        # Заменяем русские буквы на латинские
        text = manual_input
        replacements = {'А':'A', 'а':'A', 'В':'B', 'в':'B', 'С':'C', 'с':'C', 'Д':'D', 'д':'D'}
        for rus, lat in replacements.items():
            text = text.replace(rus, lat)
        
        # Извлекаем ответы
        manual_ans = {}
        pattern = r'(\d+)\s*[.\):;\-–—]*\s*([A-D])\b'
        matches = re.findall(pattern, text, re.IGNORECASE)
        
        for num_str, ans in matches:
            manual_ans[int(num_str)] = ans.upper()
        
        if manual_ans:
            st.session_state.student_answers = manual_ans
            st.session_state.answers_processed = True
            
            st.success(f"✅ Распознано {len(manual_ans)} ответов")
            st.json(manual_ans)
            
            # ПРОВЕРЯЕМ ТЕСТ
            if st.session_state.get('test_data'):
                results, score, correct = grade_test(
                    st.session_state.test_data,
                    manual_ans
                )
                
                st.session_state.test_results = results
                st.session_state.test_score = score
                
                st.balloons()
                st.rerun()
            else:
                st.warning("⚠️ Сначала загрузите тест с вопросами!")
        else:
            st.error("❌ Не удалось распознать ответы. Проверьте формат.")  
            # === ДОБАВЬТЕ ЭТУ ДИАГНОСТИКУ ===
    st.subheader("🔬 Диагностика PDF с ответами")
    
    try:
        import pdfplumber
        with pdfplumber.open(uploaded_answers) as pdf:
            full_text = ""
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    full_text += text + "\n"
            
            st.write(f"**Страниц в PDF:** {len(pdf.pages)}")
            st.text_area("**Весь текст из PDF:**", full_text, height=300, key="full_pdf_text")
            
            if not full_text.strip():
                st.error("⚠️ PDF пустой! Это сканированное изображение. Используйте ручной ввод.")
            else:
                st.info("✅ Текст найден! Смотрите выше что извлеклось.")
                
    except Exception as e:
        st.error(f"Ошибка чтения PDF: {e}")

        if st.button("📊 Проверить ответы", key="extract_answers_btn", disabled=not st.session_state.answers_pdf):
            with st.spinner("🔍 Проверка ответов..."):
                # ИЗМЕНЕНИЕ 2: Используем улучшенную функцию
                st.session_state.student_answers = extract_answers_from_pdf(st.session_state.answers_pdf)
                
                if st.session_state.student_answers:
                    st.success(f"✅ Найдено {len(st.session_state.student_answers)} ответов")
                    st.session_state.answers_processed = True
                    
                    # ДОБАВЛЕНИЕ: Показываем превью ответов
                    with st.expander("👁️ Превью ответов студента", expanded=False):
                        for num, ans in list(st.session_state.student_answers.items())[:5]:
                            st.write(f"Вопрос {num}: **{ans}**")
                    
                    if st.session_state.test_data:
                        st.session_state.test_results, st.session_state.test_score, correct_count = grade_test(
                            st.session_state.test_data, 
                            st.session_state.student_answers
                        )
                        st.balloons()
                    else:
                        st.warning(trans["no_questions_found"])
                else:
                    st.warning(trans["no_answers_found"])
    
    st.divider()
    
    # Ручная корректировка ответов
    if st.session_state.test_data and st.session_state.student_answers:
        with st.expander(trans["manual_correction"], expanded=False):  # ИЗМЕНЕНИЕ 3: Сделали сворачиваемым
            st.warning("Проверьте автоматическое распознавание ответов и при необходимости скорректируйте")
            
            cols = st.columns([1, 3, 2, 2])
            with cols[0]:
                st.markdown("**№**")
            with cols[1]:
                st.markdown("**Ответ студента**")
            with cols[2]:
                st.markdown("**Правильный ответ**")
            with cols[3]:
                st.markdown("**Статус**")
            
            for i, q in enumerate(st.session_state.test_data):
                cols = st.columns([1, 3, 2, 2])
                with cols[0]:
                    st.markdown(f"**{q['number']}**")
                with cols[1]:
                    new_answer = st.text_input(
                        f"Ответ на вопрос {q['number']}",
                        value=st.session_state.student_answers.get(q['number'], ""),
                        key=f"answer_{q['number']}",
                        label_visibility="collapsed"
                    )
                    st.session_state.student_answers[q['number']] = new_answer.upper()
                with cols[2]:
                    new_correct = st.text_input(
                        f"Правильный ответ на вопрос {q['number']}",
                        value=q['correct_answer'],
                        key=f"correct_{q['number']}",
                        label_visibility="collapsed"
                    )
                    st.session_state.test_data[i]['correct_answer'] = new_correct.upper()
                with cols[3]:
                    correct = new_answer.upper() == new_correct.upper()
                    st.markdown(f"{'✅' if correct else '❌'}")
            
            if st.button(trans["recalculate_results"], key="recalculate_results", use_container_width=True):
                st.session_state.test_results, st.session_state.test_score, correct_count = grade_test(
                    st.session_state.test_data, 
                    st.session_state.student_answers
                )
                st.success("Результаты обновлены!")
                st.rerun()  # ДОБАВЛЕНИЕ: Перезагружаем страницу
    
    # Просмотр теста
    if st.session_state.test_data and st.session_state.show_test_viewer:
        with st.expander("📝 Просмотр теста", expanded=False):  # ИЗМЕНЕНИЕ 4: Сделали сворачиваемым
            for question in st.session_state.test_data:
                with st.expander(f"Вопрос {question['number']}: {question['question'][:80]}...", expanded=False):
                    st.markdown(f"**Полный текст:** {question['question']}")
                    st.caption(f"Страница: {question['page']}")
                    
                    for option, text in question['options'].items():
                        is_correct = option == question['correct_answer']
                        st.markdown(f"{'✅' if is_correct else '⚪'} **{option}.** {text}")
                    
                    if question['correct_answer']:
                        st.success(f"**Правильный ответ:** {question['correct_answer']}")
    
    # ИЗМЕНЕНИЕ 5: Полностью переработанное отображение результатов
    if st.session_state.test_results:
        st.divider()
        st.subheader(trans["test_summary"])
        
        correct_count = sum(1 for r in st.session_state.test_results if r['is_correct'])
        total_questions = len(st.session_state.test_results)
        score_percent = st.session_state.test_score
        
        # ДОБАВЛЕНИЕ: Используем новую систему оценивания если доступна
        if GRADING_SYSTEM_AVAILABLE:
            grading_system = GradingSystem()
            grade_info = grading_system.grade_test(st.session_state.test_results)
            
            grade_value = grade_info['grade']
            feedback = grade_info['feedback']
            
            st.success(f"**🎓 Оценка: {grade_value}/5**")
            st.info(f"**💬 Обратная связь:** {feedback}")
        else:
            # Fallback
            grade_value, _ = calculate_grade("test", {"score": score_percent, "results": st.session_state.test_results})
            grade_text = trans[f"grade_{grade_value}"]
            st.success(f"**🎓 Оценка: {grade_text} ({grade_value}/5)**")
        
        # Метрики
        col1, col2, col3, col4 = st.columns(4)  # ИЗМЕНЕНИЕ 6: Добавили 4-ю колонку
        col1.metric("Общий балл", f"{score_percent}%")
        col2.metric("Правильные", f"{correct_count}/{total_questions}")
        col3.metric("Неправильные", total_questions - correct_count)
        col4.metric("Оценка", f"{grade_value}/5")  # ДОБАВЛЕНИЕ
        
        # Визуализация результатов
        fig = go.Figure()
        fig.add_trace(go.Pie(
            labels=['Правильные', 'Неправильные'],
            values=[correct_count, total_questions - correct_count],
            hole=0.5,
            marker_colors=['#4CAF50', '#F44336'],
            textinfo='label+percent',
            textfont_size=14
        ))
        fig.update_layout(
            title='Соотношение ответов',
            height=300,
            showlegend=True,
            margin=dict(t=50, b=0, l=0, r=0)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # ДОБАВЛЕНИЕ: Сохранение в БД
        if save_test_results_to_db(
            st.session_state.student_id,
            st.session_state.test_name,
            score_percent,
            correct_count,
            total_questions
        ):
            logger.info(f"Результаты теста сохранены для {st.session_state.student_id}")
        
        # Детализация по вопросам
        st.subheader("🔍 Детализация ответов")
        
        # ДОБАВЛЕНИЕ: Фильтр для отображения
        filter_option = st.radio(
            "Показать:",
            ["Все вопросы", "Только неправильные", "Только правильные"],
            horizontal=True
        )
        
        filtered_results = st.session_state.test_results
        if filter_option == "Только неправильные":
            filtered_results = [r for r in st.session_state.test_results if not r['is_correct']]
        elif filter_option == "Только правильные":
            filtered_results = [r for r in st.session_state.test_results if r['is_correct']]
        
        for res in filtered_results:
            status = "✅" if res['is_correct'] else "❌"
            with st.expander(f"{status} Вопрос {res['number']}", expanded=False):
                st.markdown(f"**Вопрос:** {res['question']}")
                
                cols = st.columns(2)
                with cols[0]:
                    st.markdown("**Варианты ответа:**")
                    for option, text in res['options'].items():
                        if option == res['correct_answer']:
                            st.success(f"✅ **{option}.** {text}")
                        elif option == res['student_answer'] and not res['is_correct']:
                            st.error(f"❌ **{option}.** {text}")
                        else:
                            st.markdown(f"**{option}.** {text}")
                
                with cols[1]:
                    st.markdown(f"**Ответ студента:** `{res['student_answer']}`")
                    st.markdown(f"**Правильный ответ:** `{res['correct_answer']}`")
                    if res['is_correct']:
                        st.success("✅ Ответ верный")
                    else:
                        st.error("❌ Ответ неверный")
                        st.caption(f"Страница в тесте: {res['page']}")
        
        # Экспорт результатов
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="📥 Скачать отчет PDF",
                data=create_results_pdf(
                    st.session_state.test_results,
                    st.session_state.test_score,
                    correct_count,
                    total_questions
                ),
                file_name=f"Результаты_{st.session_state.student_id}_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        
        with col2:
            # ДОБАВЛЕНИЕ: Экспорт в CSV
            csv_data = pd.DataFrame(st.session_state.test_results)
            st.download_button(
                label="📊 Экспорт в CSV",
                data=csv_data.to_csv(index=False).encode('utf-8'),
                file_name=f"Результаты_{st.session_state.student_id}_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        # ИЗМЕНЕНИЕ 7: Улучшенная система комментариев к оценке
        st.divider()
        st.subheader("📝 Комментарий к оценке")
        
        # Автоматический комментарий
        if GRADING_SYSTEM_AVAILABLE and grade_info:
            default_comment = grade_info.get('feedback', '')
        else:
            if score_percent >= 90:
                default_comment = "Отличный результат! Демонстрирует глубокое понимание материала."
            elif score_percent >= 75:
                default_comment = "Хороший результат. Имеются незначительные пробелы в знаниях."
            elif score_percent >= 60:
                default_comment = "Удовлетворительно. Рекомендуется повторить ключевые темы."
            else:
                default_comment = "Неудовлетворительно. Требуется дополнительное изучение материала."
        
        st.session_state.grade_comment = st.text_area(
            "Комментарий преподавателя:", 
            value=default_comment,
            height=120,
            help="Этот комментарий будет сохранен вместе с оценкой"
        )
        
        # ИЗМЕНЕНИЕ 8: Улучшенная кнопка сохранения
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button("💾 Сохранить оценку в журнал", key="save_test_grade", use_container_width=True, type="primary"):
                # Сохраняем с процентами
                if add_to_grade_history(
                    st.session_state.test_name, 
                    grade_value, 
                    st.session_state.grade_comment,
                    percentage=score_percent
                ):
                    st.success("✅ Оценка сохранена в журнал успеваемости!")
                    st.balloons()
                else:
                    st.warning("⚠️ Оценка сохранена в сессии, но не в базу данных")
        
        with col2:
            if st.button("🔄 Очистить", key="clear_test", use_container_width=True):
                st.session_state.test_results = []
                st.session_state.test_data = []
                st.session_state.student_answers = {}
                st.session_state.test_pdf = None
                st.session_state.answers_pdf = None
                st.rerun()
    
    # Инструкция по форматированию - оставляем как есть
    else:  # ДОБАВЛЕНИЕ: Показываем инструкцию только если нет результатов
        st.info(f"""
        **📝 Рекомендации по оформлению тестов:**
        
        1. {trans['question_format']}
        2. {trans['answer_format']}
        3. Варианты ответов оформляйте как:  
           **A.** Первый вариант  
           **B.** Второй вариант
        4. Правильный ответ указывайте как:  
           **Ответ: A**
        5. Ответы студента должны быть в формате:  
           **1. B**  
           **2. C**
        """)
# ============================================================================
# ПОЛНАЯ ЗАМЕНА СЕКЦИИ МАТЕМАТИКИ
# Найдите строку: elif subject == trans["math"]:
# Замените весь блок до elif subject == trans["eng"]: на этот код
# ============================================================================

elif subject == trans["math"]:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.write(f"**{trans['example']}** `x**2 + 2*x + 1`, {trans['answer']}: `(x+1)**2`")
        
        default_text = st.session_state.get('uploaded_text', 'x**2 + 2*x + 1')
        task = st.text_input(trans["task"], value=default_text)
        
        answer = st.text_input(trans["answer"], "(x+1)**2")
        
        if st.button(trans["check_math"], use_container_width=True, type="primary", key="math_check_btn"):
            with st.spinner("Проверяем решение..."):
                try:
                    math_result = check_math(task, answer)
                    st.session_state.math_result = math_result
                    math_result['task'] = task
                    math_result['answer'] = answer
                    
                    # Сохранение в БД через контекстный менеджер
                    try:
                        with get_db_connection() as conn:
                            c = conn.cursor()
                            c.execute("INSERT INTO learning_history (student_id, subject, topic, score) VALUES (?, ?, ?, ?)",
                                      (st.session_state.student_id, "math", "Алгебра", 1 if math_result['is_correct'] else 0))
                            logger.info(f"Результат математики сохранен для {st.session_state.student_id}")
                    except Exception as db_error:
                        logger.error(f"Ошибка БД: {str(db_error)}")
                    
                    if math_result['is_correct']:
                        st.success(trans["correct"])
                        st.balloons()
                    else:
                        st.error(trans["incorrect"])
                except Exception as e:
                    logger.error(f"Ошибка проверки математики: {str(e)}")
                    st.error(f"Ошибка при проверке: {str(e)}")
                    st.session_state.math_result = None
        
        if st.session_state.math_result:
            math_result = st.session_state.math_result
            
            with st.expander(trans["solution"], expanded=True):
                st.write(f"**{trans['original']}**")
                st.latex(rf"\text{{}} {task} \quad \rightarrow \quad {math_result.get('latex_answer', '')}")
                
                st.write(f"**{trans['explanation']}**")
                st.code(math_result.get('explanation', ''), language='python')
                
                st.write(f"**{trans['correct_answer']}**")
                st.latex(math_result.get('latex_answer', ''))
                
                st.write(f"**{trans['text_rep']}**")
                st.code(math_result.get('correct_answer', ''))
            
            if math_result.get('detailed_feedback'):
                with st.expander(trans["errors"], expanded=True):
                    st.write(math_result['detailed_feedback'])
            
            if 'math_scores' not in st.session_state:
                st.session_state.math_scores = []
            
            score = 1 if math_result['is_correct'] else 0
            st.session_state.math_scores.append(score)
            
            # AI объяснение
            if st.session_state.gemini_key and st.session_state.selected_model:
                ai_state = st.session_state.ai_status['math']
                
                if ai_state['generating']:
                    with st.spinner(trans["ai_loading"]):
                        try:
                            api_available, remaining = check_api_rate_limit()
                            if not api_available:
                                st.error(f"Лимит API достигнут. Повторите через {remaining} сек.")
                                ai_state['generating'] = False
                                st.session_state.waiting_for_api = True
                                st.session_state.rate_limit_timer = time.time()
                                time.sleep(1)
                                st.rerun()
                            
                            query_hash = hashlib.md5(f"{task}_{answer}_{st.session_state.selected_model}".encode()).hexdigest()
                            
                            try:
                                with get_db_connection() as conn:
                                    c = conn.cursor()
                                    c.execute("SELECT response FROM responses WHERE query_hash = ?", (query_hash,))
                                    cached_response = c.fetchone()
                                    
                                    if cached_response:
                                        ai_explanation = cached_response[0]
                                        logger.info("Использован кэшированный AI ответ")
                                    else:
                                        current_time = time.time()
                                        if current_time - st.session_state.last_request_time < 2:
                                            time.sleep(2 - (current_time - st.session_state.last_request_time))
                                        
                                        ai_explanation = get_math_ai_feedback(
                                            task, 
                                            answer, 
                                            math_result.get('correct_answer', ''), 
                                            st.session_state.gemini_key,
                                            st.session_state.selected_model,
                                            trans["ai_math_prompt"]
                                        )
                                        st.session_state.last_request_time = time.time()
                                        st.session_state.api_usage_count += 1
                                        
                                        c.execute("INSERT OR REPLACE INTO responses (query_hash, response, timestamp) VALUES (?, ?, ?)",
                                                  (query_hash, ai_explanation, datetime.datetime.now()))
                                        logger.info("AI ответ сохранен в кэш")
                            except Exception as db_error:
                                logger.warning(f"Кэш недоступен, запрашиваем напрямую: {str(db_error)}")
                                ai_explanation = get_math_ai_feedback(
                                    task, 
                                    answer, 
                                    math_result.get('correct_answer', ''), 
                                    st.session_state.gemini_key,
                                    st.session_state.selected_model,
                                    trans["ai_math_prompt"]
                                )
                            
                            ai_state['response'] = ai_explanation
                            ai_state['generating'] = False
                        except Exception as e:
                            error_msg = str(e)
                            logger.error(f"Ошибка AI: {error_msg}")
                            if "429" in error_msg or "quota" in error_msg.lower():
                                ai_state['error'] = trans["rate_limit"]
                                st.session_state.rate_limit_timer = time.time()
                                st.session_state.waiting_for_api = True
                            elif "API_KEY" in error_msg or "401" in error_msg:
                                ai_state['error'] = trans["invalid_key"]
                            elif "404" in error_msg:
                                ai_state['error'] = trans["model_not_found"]
                            else:
                                ai_state['error'] = f"Ошибка: {error_msg}"
                            ai_state['generating'] = False
                
                if ai_state['error']:
                    st.error(f"**{trans['ai_error']}:** {ai_state['error']}")
                    if st.button(trans["ai_retry"], key="math_ai_retry", use_container_width=True):
                        ai_state['error'] = None
                        ai_state['generating'] = True
                        st.rerun()
                
                elif ai_state['response']:
                    st.info(f"**AI Tutor:** {ai_state['response']}")
                    if st.button(trans["ai_clear"], key="math_ai_clear", use_container_width=True):
                        ai_state['response'] = None
                        st.rerun()
                
                elif st.button(trans["ai_math"], use_container_width=True):
                    ai_state['generating'] = True
                    st.rerun()
            
            st.subheader(trans["video_explanation"])
            st_player("https://www.youtube.com/watch?v=KG6ILNOiMgM", height=350)
    
    with col2:
        st.subheader(trans["adaptive_learning"])
        
        if st.button(trans["generate_task"], use_container_width=True):
            task_new, answer_new = generate_personalized_task(st.session_state.student_id)
            st.session_state.personalized_task = {"task": task_new, "answer": answer_new}
        
        if st.session_state.personalized_task.get("task"):
            st.info("**Персональное задание на основе ваших ошибок:**")
            st.code(st.session_state.personalized_task["task"])
            st.write("**Правильный ответ:**")
            st.code(st.session_state.personalized_task["answer"])
            
            if st.button("Использовать это задание", key="use_personalized_task"):
                st.session_state.uploaded_text = st.session_state.personalized_task["task"]
                st.rerun()
    
    # Блок оценивания
    if st.session_state.math_result:
        st.divider()
        st.subheader("Оценивание")
        
        # Используем новую систему оценивания если доступна
        if GRADING_SYSTEM_AVAILABLE:
            grading_system = GradingSystem()
            grade_info = grading_system.grade_math(st.session_state.math_result)
            
            grade_value = grade_info['grade']
            st.session_state.math_grade = grade_value
            
            grade_text = trans[f"grade_{grade_value}"]
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.metric("Оценка", f"{grade_value}/5", grade_text)
            with col2:
                st.info(f"**{grade_info['feedback']}**")
            
            # Показываем детали
            if grade_info.get('strengths'):
                with st.expander("Сильные стороны", expanded=False):
                    for strength in grade_info['strengths']:
                        st.write(f"✓ {strength}")
            
            if grade_info.get('areas_for_improvement'):
                with st.expander("Рекомендации", expanded=False):
                    for area in grade_info['areas_for_improvement']:
                        st.write(f"→ {area}")
            
            comment = grade_info['feedback']
            percentage = grade_info.get('percentage', 100 if st.session_state.math_result['is_correct'] else 0)
        else:
            # Fallback к старой системе
            grade_value, percentage = calculate_grade("math", st.session_state.math_result)
            st.session_state.math_grade = grade_value
            
            grade_text = trans[f"grade_{grade_value}"]
            st.metric("Оценка", f"{grade_value}/5", grade_text)
            
            comment = "Отличная работа!" if st.session_state.math_result['is_correct'] else "Есть ошибки. Изучите объяснение."
        
        st.session_state.grade_comment = st.text_area(
            trans["grade_comment"], 
            value=comment,
            height=100,
            help="Комментарий сохраняется вместе с оценкой"
        )
        
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button("Сохранить оценку в журнал", key="save_math_grade", use_container_width=True, type="primary"):
                if add_to_grade_history("Математика", grade_value, st.session_state.grade_comment, percentage):
                    st.success("Оценка сохранена!")
                    st.balloons()
                else:
                    st.warning("Оценка сохранена в сессии")
        
        with col2:
            if st.button("Новая задача", key="clear_math", use_container_width=True):
                st.session_state.math_result = None
                st.rerun()

# ============================================================================
# ПОЛНАЯ ЗАМЕНА СЕКЦИИ АНГЛИЙСКОГО
# Найдите строку: elif subject == trans["eng"]:
# Замените весь блок до конца секции на этот код
# ============================================================================

elif subject == trans["eng"]:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        default_text = st.session_state.get('uploaded_text', 'He go to school.')
        text = st.text_area("Введите текст:", value=default_text, height=300)
        
        if st.button(trans["check_text"], use_container_width=True, type="primary", key="eng_check_btn"):
            with st.spinner('Проверяем текст...'):
                try:
                    errors = check_english(text)
                    st.session_state.english_errors = errors
                    
                    word_count = len(text.split())
                    error_count = len(errors)
                    
                    if word_count > 0:
                        score = max(1, 5 - (error_count / (word_count / 10)))
                        score = round(score, 1)
                    else:
                        score = 0
                    
                    st.session_state.english_score = score
                    
                    # Сохранение в БД через контекстный менеджер
                    try:
                        with get_db_connection() as conn:
                            c = conn.cursor()
                            
                            # Сохраняем в историю ошибок
                            for error in errors:
                                c.execute('''INSERT INTO error_history (student_id, subject, error_type) 
                                             VALUES (?, ?, ?)
                                             ON CONFLICT(student_id, error_type) 
                                             DO UPDATE SET count = count + 1, last_occurrence = CURRENT_TIMESTAMP''',
                                          (st.session_state.student_id, "english", error.get('ruleId', 'unknown')))
                            
                            # Получаем историю ошибок
                            c.execute("SELECT error_type, count FROM error_history WHERE student_id = ?", 
                                      (st.session_state.student_id,))
                            st.session_state.error_history = [{"error_type": row[0], "count": row[1]} for row in c.fetchall()]
                            
                            logger.info(f"История ошибок обновлена для {st.session_state.student_id}")
                    except Exception as db_error:
                        logger.error(f"Ошибка БД: {str(db_error)}")
                        
                except Exception as e:
                    logger.error(f"Ошибка проверки английского: {str(e)}")
                    st.error(f"Ошибка при проверке текста: {str(e)}")
                    st.session_state.english_errors = []
                    st.session_state.english_score = None
        
        if 'english_score' in st.session_state and st.session_state.english_score is not None:
            st.metric("Текущая оценка", st.session_state.english_score, 
                      help="Рассчитана по формуле: 5 - (кол-во ошибок / (кол-во слов / 10))")
        
        if st.session_state.english_errors is not None:
            errors = st.session_state.english_errors
            
            if not errors:
                st.success(trans["no_errors"])
                st.balloons()
            else:
                st.subheader(f"Найдено {len(errors)} ошибок")
                for i, error in enumerate(errors):
                    if isinstance(error, dict) and 'message' in error:
                        with st.expander(f"Ошибка #{i+1}: {error['message']}", expanded=True):
                            st.error(f"**{error['message']}**")
                            
                            if 'suggestion' in error and error['suggestion']:
                                st.info(f"**Совет:** {error['suggestion']}")
                            
                            if 'context' in error and error['context']:
                                st.code(f"Контекст: {error['context']}")
                            
                            rule_id = error.get('ruleId', '')
                            explanation = get_grammar_explanation(rule_id)
                            with st.expander(trans["why_error"], expanded=False):
                                st.write(explanation)
    
    with col2:
        if hasattr(st.session_state, 'error_history') and st.session_state.error_history:
            st.subheader("Аналитика ошибок")
            df_errors = pd.DataFrame(st.session_state.error_history)
            df_errors = df_errors.sort_values('count', ascending=False).head(5)
            
            fig, ax = plt.subplots()
            ax.barh(df_errors['error_type'], df_errors['count'], color='#ff6b6b')
            ax.set_xlabel('Количество повторений')
            ax.set_title('Топ-5 частых ошибок')
            st.pyplot(fig)
            
            st.write("**Рекомендации по улучшению:**")
            for error_type in df_errors['error_type'].head(3):
                st.write(f"- {get_grammar_explanation(error_type)}")
    
    # Блок оценивания английского
    if st.session_state.english_errors is not None:
        st.divider()
        st.subheader("Оценивание")
        
        errors = st.session_state.english_errors
        word_count = len(text.split())
        
        # Используем новую систему оценивания
        if GRADING_SYSTEM_AVAILABLE:
            grading_system = GradingSystem()
            grade_info = grading_system.grade_english(errors, word_count)
            
            grade_value = grade_info['grade']
            st.session_state.eng_grade = grade_value
            
            grade_text = trans[f"grade_{grade_value}"]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Оценка", f"{grade_value}/5", grade_text)
            with col2:
                st.metric("Ошибок", grade_info['error_count'])
            with col3:
                st.metric("Точность", f"{grade_info.get('percentage', 0):.1f}%")
            
            st.info(f"**{grade_info['feedback']}**")
            
            # Категории ошибок
            if grade_info.get('error_categories'):
                with st.expander("Детализация по категориям", expanded=False):
                    for category, count in grade_info['error_categories'].items():
                        st.write(f"**{category.title()}:** {count} {'ошибка' if count == 1 else 'ошибок'}")
            
            comment = grade_info['feedback']
            percentage = grade_info.get('percentage', 0)
        else:
            # Fallback
            error_count = len(errors)
            if word_count > 0:
                score = max(1, 5 - (error_count / (word_count / 10)))
                grade_value = int(round(score))
                percentage = max(0, 100 - error_count * 5)
            else:
                grade_value = 2
                percentage = 0
            
            grade_text = trans[f"grade_{grade_value}"]
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Оценка", f"{grade_value}/5", grade_text)
            with col2:
                st.metric("Найдено ошибок", error_count)
            
            comment = f"Найдено {error_count} {'ошибка' if error_count == 1 else 'ошибок'}. {'Отличная работа!' if error_count == 0 else 'Обратите внимание на рекомендации.'}"
        
        st.session_state.grade_comment = st.text_area(
            trans["grade_comment"],
            value=comment,
            height=100,
            help="Комментарий сохраняется с оценкой"
        )
        
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button("Сохранить оценку в журнал", key="save_eng_grade", use_container_width=True, type="primary"):
                if add_to_grade_history("Английский", grade_value, st.session_state.grade_comment, percentage):
                    st.success("Оценка сохранена!")
                    st.balloons()
                else:
                    st.warning("Оценка сохранена в сессии")
        
        with col2:
            if st.button("Новый текст", key="clear_eng", use_container_width=True):
                st.session_state.english_errors = None
                st.session_state.english_score = None
                st.rerun()

if st.session_state.math_result or st.session_state.english_errors or st.session_state.test_results:
    st.divider()
    with st.expander(trans["export_results"], expanded=False):
        report = generate_report()
        st.download_button(
            label=trans["export_btn"],
            data=report,
            file_name=f"report_{st.session_state.student_id}_{datetime.datetime.now().strftime('%Y%m%d')}.txt",
            mime="text/plain"
        )

st.subheader("✏️ Ручной ввод (если PDF не работает)")

with st.expander("Ввести вопросы вручную"):
    num_questions = st.number_input("Сколько вопросов?", 1, 20, 3, key="manual_num_q")
    
    if st.button("Создать поля для ввода", key="create_manual_fields"):
        st.session_state.manual_mode = True
    
    if st.session_state.get('manual_mode'):
        manual_q = []
        
        for i in range(num_questions):
            st.markdown(f"### Вопрос {i+1}")
            q = st.text_input(f"Текст вопроса", key=f"mq_text_{i}")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                a = st.text_input("A)", key=f"mopt_a_{i}")
            with col2:
                b = st.text_input("B)", key=f"mopt_b_{i}")
            with col3:
                c = st.text_input("C)", key=f"mopt_c_{i}")
            with col4:
                d = st.text_input("D)", key=f"mopt_d_{i}")
            
            correct = st.radio("Правильный:", ['A','B','C','D'], key=f"mcorrect_{i}", horizontal=True)
            
            if q and a and b:
                manual_q.append({
                    'number': i+1,
                    'question': q,
                    'options': {'A':a, 'B':b, 'C':c, 'D':d},
                    'correct_answer': correct,
                    'page': 1
                })
        
        if st.button("✅ Использовать эти вопросы", key="use_manual_q"):
            st.session_state.test_data = manual_q
            st.session_state.test_processed = True
            st.success(f"✅ Добавлено {len(manual_q)} вопросов!")
            st.rerun()

st.divider()
# здесь продолжается ваш код...
with st.expander("💬 Обратная связь", expanded=True):
    feedback_dir = "feedback"
    os.makedirs(feedback_dir, exist_ok=True)

    feedback_text = st.text_area(trans["feedback"], height=100, key="feedback_text", 
                                value=st.session_state.feedback_text, 
                                placeholder="Поделитесь вашими впечатлениями...")
    
    rating = st.slider("Оцените приложение (1-5 звёзд)", 1, 5, 5)
    
    if st.button(trans["send_feedback"], use_container_width=True, type="primary") and not st.session_state.feedback_sent:
        if feedback_text.strip():
            now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            feedback_hash = hashlib.md5(feedback_text.encode()).hexdigest()[:6]
            filename = f"{feedback_dir}/{st.session_state.student_id}_{now}_{feedback_hash}.txt"
            
            try:
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(f"Студент: {st.session_state.student_id}\n")
                    f.write(f"Уровень: {st.session_state.learning_level}\n")
                    f.write(f"Рейтинг: {'⭐' * rating}\n")
                    f.write(f"Дата: {now}\n\n")
                    f.write(feedback_text)
                    
                st.success("✅ Спасибо за ваш отзыв! Он сохранен.")
                st.balloons()
                st.session_state.feedback_text = ""
                st.session_state.feedback_sent = True
            except Exception as e:
                st.error(f"Ошибка сохранения отзыва: {str(e)}")
        else:
            st.warning("Пожалуйста, введите ваш отзыв")

tips = [
    "💡 Для сложных выражений используйте скобки для группировки",
    "💡 Проверяйте орфографию перед отправкой текста на проверку",
    "💡 Для длинных текстов используйте загрузку файлов PDF или изображений",
    "💡 Сохраняйте API ключ в безопасном месте",
    "💡 Регулярно проверяйте свой прогресс для отслеживания улучшений",
    "💡 Используйте AI-объяснения для лучшего понимания ошибок",
    "💡 Персональные задания помогают закрепить проблемные темы",
    "💡 Календарь активности показывает вашу продуктивность",
    "💡 Для тестов используйте четкое форматирование вопросов и ответов"
]

st.info(f"**{trans['pro_tip']}:** {tips[datetime.datetime.now().second % len(tips)]}")

st.divider()
st.markdown("""
<div style="text-align: center; padding: 20px; color: #7f8c8d;">
    <p>EduAI Tutor Pro © 2024 | AI-репетитор нового поколения</p>
    <p>Сделано с ❤️ для будущего образования</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<style>
    :root {
        --primary: #6a11cb;
        --secondary: #2575fc;
        --success: #00b09b;
        --danger: #ff416c;
        --warning: #ffc107;
        --info: #17a2b8;
        --light: #f8f9fa;
        --dark: #343a40;
        --gradient: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
    }
    
    body {
        background: linear-gradient(to right, #f5f7fa, #e4e7f0);
        color: #333;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        line-height: 1.6;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: var(--dark);
        font-weight: 700;
        margin-bottom: 1rem;
    }
    
    .stButton>button {
        border-radius: 12px;
        padding: 12px 28px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        background: var(--gradient);
        color: white;
        border: none;
        font-size: 16px;
    }

    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.2);
        background: linear-gradient(135deg, #5a0db9 0%, #1c65e0 100%);
    }
    
    .stButton>button:active {
        transform: translateY(1px);
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
    }
    
    .stButton>button:disabled {
        background: #cccccc;
        cursor: not-allowed;
    }
    
    .stExpander {
        background-color: white;
        border-radius: 15px;
        padding: 25px;
        margin-bottom: 25px;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.08);
        border: none;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .stExpander:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 25px rgba(0, 0, 0, 0.15);
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2c3e50 0%, #1a2530 100%);
        color: white;
        padding: 25px;
        box-shadow: 5px 0 15px rgba(0, 0, 0, 0.1);
    }
    
    [data-testid="stSidebar"] .stTextInput input {
        background-color: rgba(255, 255, 255, 0.12);
        color: white;
        border-radius: 10px;
        padding: 12px;
        border: 1px solid rgba(255,255,255,0.3);
        transition: all 0.3s;
    }
    
    [data-testid="stSidebar"] .stTextInput input:focus {
        background-color: rgba(255, 255, 255, 0.2);
        border-color: rgba(255,255,255,0.5);
    }
    
    [data-testid="stSidebar"] .stSelectbox select {
        background-color: rgba(255, 255, 255, 0.12);
        color: white;
        border-radius: 10px;
        padding: 10px;
        border: 1px solid rgba(255,255,255,0.3);
    }
    
    [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"]>div {
        background-color: rgba(255, 255, 255, 0.12);
        color: white;
    }
    
    [data-testid="stSidebar"] .stRadio > label {
        color: white;
        font-weight: 500;
    }
    
    [data-testid="stSidebar"] .stMarkdown h1, 
    [data-testid="stSidebar"] .stMarkdown h2, 
    [data-testid="stSidebar"] .stMarkdown h3, 
    [data-testid="stSidebar"] .stMarkdown h4, 
    [data-testid="stSidebar"] .stMarkdown h5, 
    [data-testid="stSidebar"] .stMarkdown h6 {
        color: white;
    }
    
    [data-testid="stSidebar"] .stProgress > div > div > div {
        background: var(--gradient);
    }
    
    .stDataFrame {
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        overflow: hidden;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
    
    .stAlert, .stSuccess, .stError, .stWarning, .stInfo {
        animation: fadeIn 0.6s ease-out;
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0;
        font-size: 16px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    }
    
    .stSuccess {
        background: linear-gradient(135deg, var(--success) 0%, #96c93d 100%);
        color: white;
        border: none;
    }
    
    .stError {
        background: linear-gradient(135deg, var(--danger) 0%, #ff4b2b 100%);
        color: white;
        border: none;
    }
    
    .stInfo {
        background: linear-gradient(135deg, var(--info) 0%, #6dd5ed 100%);
        color: white;
        border: none;
    }
    
    .stWarning {
        background: linear-gradient(135deg, var(--warning) 0%, #ffcc80 100%);
        color: #212529;
        border: none;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .stSpinner > div {
        border: 5px solid rgba(106, 17, 203, 0.2);
        border-top: 5px solid var(--primary);
        border-radius: 50%;
        width: 45px;
        height: 45px;
        animation: spin 1s linear infinite;
        margin: 0 auto;
    }
    
    @media (max-width: 768px) {
        .main .block-container { 
            padding: 1.2rem; 
        }
        
        .stTextInput>div>div>input, .stTextArea>div>div>textarea {
            font-size: 16px !important; 
            padding: 14px !important;
        }
        
        .stButton>button { 
            width: 100%; 
            padding: 16px; 
            font-size: 17px; 
            margin: 10px 0; 
        }
        
        .sidebar .sidebar-content { 
            padding: 1.5rem; 
        }
        
        .stRadio>div { 
            flex-direction: column; 
            gap: 12px; 
        }
        
        .stRadio>div>label { 
            margin: 8px 0; 
            padding: 14px; 
            border-radius: 12px; 
            background: rgba(255, 255, 255, 0.12); 
            width: 100%;
        }
        
        .stTextArea>div>div>textarea { 
            min-height: 200px; 
        }
        
        h1 { 
            font-size: 2.2rem !important; 
        }
        
        h2 { 
            font-size: 1.8rem !important; 
        }
        
        h3 { 
            font-size: 1.6rem !important; 
        }
    }
    
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(120deg, #e0f7fa 0%, #f5f5f5 100%);
    }
    
    .custom-card {
        background: white;
        border-radius: 15px;
        padding: 25px;
        margin: 20px 0;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.08);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        border-left: 5px solid var(--primary);
        animation: fadeIn 0.8s ease-out;
    }
    
    .custom-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 30px rgba(0, 0, 0, 0.15);
    }
    
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        border-radius: 12px;
        padding: 14px 18px;
        border: 1px solid #ddd;
        transition: all 0.3s;
        background-color: white;
        font-size: 16px;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .stTextInput>div>div>input:focus, .stTextArea>div>div>textarea:focus {
        border-color: var(--primary);
        box-shadow: 0 0 0 3px rgba(106, 17, 203, 0.2);
    }
    
    .stRadio>div>label {
        background: white;
        border-radius: 12px;
        padding: 14px 22px;
        margin: 10px 0;
        box-shadow: 0 4px 10px rgba(0,0,0,0.08);
        transition: all 0.3s;
        font-weight: 500;
        border: 2px solid transparent;
    }
    
    .stRadio>div>label:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 15px rgba(0,0,0,0.1);
        border-color: rgba(106, 17, 203, 0.2);
    }
    
    .stRadio>div>div:first-child {
        background: var(--gradient);
        border-radius: 12px;
    }
    
    .stDivider {
        border-top: 2px solid var(--primary);
        border-radius: 2px;
        margin: 30px 0;
        opacity: 0.3;
    }
    
    @keyframes fadeInMain {
        from { opacity: 0; transform: translateX(-20px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    .main .block-container {
        animation: fadeInMain 0.8s ease-out;
    }
    
    @keyframes fadeInSidebar {
        from { opacity: 0; transform: translateX(20px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    [data-testid="stSidebar"] {
        animation: fadeInSidebar 0.8s ease-out;
    }
    
    @keyframes pulseButton {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    .stButton>button:focus {
        animation: pulseButton 0.6s ease;
    }
    
    .stProgress > div > div > div {
        background: var(--gradient) !important;
    }
    
    .stMarkdown p, .stMarkdown li {
        font-size: 16px;
        line-height: 1.8;
    }
    
    .stTooltip {
        background: rgba(255,255,255,0.9) !important;
        color: #333 !important;
        border-radius: 10px !important;
        box-shadow: 0 5px 15px rgba(0,0,0,0.15) !important;
        border: none !important;
        padding: 15px !important;
        font-size: 14px !important;
    }
    
    .lottie-container {
        margin: 0 auto;
        display: flex;
        justify-content: center;
    }
</style>
""", unsafe_allow_html=True)