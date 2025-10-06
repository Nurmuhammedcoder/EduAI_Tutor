import streamlit as st
import logging
import re

logger = logging.getLogger('EduAI')

def check_english(text):
    """Базовая проверка без LanguageTool"""
    if not text or len(text.strip()) < 3:
        return []
    
    errors = []
    
    # Базовые проверки
    sentences = text.split('.')
    for i, sentence in enumerate(sentences):
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # Проверка заглавной буквы
        if sentence and sentence[0].islower():
            errors.append({
                'message': 'Предложение должно начинаться с заглавной буквы',
                'suggestion': sentence.capitalize(),
                'context': sentence[:50],
                'ruleId': 'UPPERCASE_SENTENCE_START'
            })
        
        # Проверка a/an
        if re.search(r'\ba\s+[aeiouAEIOU]', sentence):
            errors.append({
                'message': "Используйте 'an' перед гласными",
                'suggestion': "Замените 'a' на 'an'",
                'context': sentence[:50],
                'ruleId': 'EN_A_VS_AN'
            })
    
    logger.info(f"Базовая проверка: найдено {len(errors)} ошибок")
    return errors

def get_grammar_explanation(rule_id):
    explanations = {
        "UPPERCASE_SENTENCE_START": "Предложение должно начинаться с заглавной буквы",
        "EN_A_VS_AN": "Используйте 'a' перед согласными, 'an' - перед гласными",
    }
    return explanations.get(rule_id, "Грамматическое правило английского языка")