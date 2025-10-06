"""
Улучшенная обработка PDF с лучшим распознаванием
"""
import re
import pdfplumber
import fitz
import streamlit as st
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class PDFProcessor:
    @staticmethod
    def extract_text_from_pdf(pdf_file) -> Optional[str]:
        """Извлечение текста из PDF"""
        try:
            text_content = ""
            doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
            
            for page in doc:
                text_content += page.get_text()
            
            doc.close()
            return text_content.strip()
        except Exception as e:
            logger.error(f"PDF text extraction error: {str(e)}")
            return None
    
    @staticmethod
    def extract_questions_from_pdf(pdf_file) -> List[Dict]:
        """Улучшенное извлечение вопросов из PDF"""
        questions = []
        
        try:
            with pdfplumber.open(pdf_file) as pdf:
                full_text = "\n".join([page.extract_text() for page in pdf.pages])
                
                # Паттерны для поиска вопросов
                question_pattern = r'(\d+)[.)]\s+(.*?)(?=\n\d+[.)]|\n[A-D][.)]|$)'
                question_matches = re.finditer(question_pattern, full_text, re.DOTALL)
                
                for match in question_matches:
                    q_num = int(match.group(1))
                    question_text = match.group(2).strip()
                    
                    # Извлечение вариантов ответов
                    options = {}
                    start_pos = match.end()
                    remaining_text = full_text[start_pos:]
                    
                    for letter in ['A', 'B', 'C', 'D']:
                        opt_pattern = rf'{letter}[.)]\s*(.*?)(?=\n[A-D][.)]|\n\d+[.)]|$)'
                        opt_match = re.search(opt_pattern, remaining_text)
                        
                        if opt_match:
                            options[letter] = opt_match.group(1).strip()
                    
                    # Поиск правильного ответа
                    answer_pattern = r'(?:Ответ|Answer|Correct)[:\s]*([A-D])'
                    answer_match = re.search(answer_pattern, remaining_text[:500], re.IGNORECASE)
                    correct_answer = answer_match.group(1).upper() if answer_match else ""
                    
                    if question_text and len(options) >= 2:
                        questions.append({
                            'number': q_num,
                            'question': question_text,
                            'options': options,
                            'correct_answer': correct_answer
                        })
        
        except Exception as e:
            logger.error(f"Question extraction error: {str(e)}")
            st.error(f"Ошибка обработки PDF: {str(e)}")
        
        return sorted(questions, key=lambda x: x['number'])
    
    @staticmethod
    def extract_answers_from_pdf(pdf_file) -> Dict[int, str]:
        """Улучшенное извлечение ответов из PDF"""
        answers = {}
        
        try:
            with pdfplumber.open(pdf_file) as pdf:
                full_text = "\n".join([page.extract_text() for page in pdf.pages])
                
                # Множественные паттерны для надежности
                patterns = [
                    r'(\d+)[.)]\s*([A-D])',
                    r'(\d+)\s*[-:]\s*([A-D])',
                    r'Question\s*(\d+)[:\s]*([A-D])',
                    r'Q(\d+)[:\s]*([A-D])'
                ]
                
                for pattern in patterns:
                    matches = re.findall(pattern, full_text, re.IGNORECASE)
                    for num, ans in matches:
                        answers[int(num)] = ans.strip().upper()
        
        except Exception as e:
            logger.error(f"Answer extraction error: {str(e)}")
        
        return answers