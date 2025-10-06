"""
Безопасное управление базой данных
"""
import sqlite3
from contextlib import contextmanager
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, db_path: str = 'ai_cache.db'):
        self.db_path = db_path
        self.init_database()
    
    @contextmanager
    def get_connection(self):
        """Контекстный менеджер для безопасной работы с БД"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            yield conn
            conn.commit()
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {str(e)}")
            raise
        finally:
            if conn:
                conn.close()
    
    def init_database(self):
        """Инициализация всех таблиц"""
        with self.get_connection() as conn:
            c = conn.cursor()
            
            # Таблица кэша ответов
            c.execute('''CREATE TABLE IF NOT EXISTS responses
                         (query_hash TEXT PRIMARY KEY, 
                          response TEXT, 
                          timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
            
            # Таблица истории обучения
            c.execute('''CREATE TABLE IF NOT EXISTS learning_history
                         (id INTEGER PRIMARY KEY AUTOINCREMENT,
                          student_id TEXT,
                          subject TEXT,
                          topic TEXT,
                          score REAL,
                          timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
            
            # Таблица ошибок
            c.execute('''CREATE TABLE IF NOT EXISTS error_history
                         (id INTEGER PRIMARY KEY AUTOINCREMENT,
                          student_id TEXT,
                          subject TEXT,
                          error_type TEXT,
                          count INTEGER DEFAULT 1,
                          last_occurrence DATETIME DEFAULT CURRENT_TIMESTAMP,
                          UNIQUE(student_id, error_type))''')
            
            # Таблица оценок
            c.execute('''CREATE TABLE IF NOT EXISTS grades
                         (id INTEGER PRIMARY KEY AUTOINCREMENT,
                          student_id TEXT,
                          subject TEXT,
                          grade INTEGER,
                          percentage REAL,
                          comment TEXT,
                          timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
            
            # Индексы для оптимизации
            c.execute('''CREATE INDEX IF NOT EXISTS idx_student_grades 
                         ON grades(student_id, timestamp DESC)''')
            c.execute('''CREATE INDEX IF NOT EXISTS idx_student_errors 
                         ON error_history(student_id, last_occurrence DESC)''')
    
    def save_grade(self, student_id: str, subject: str, grade: int, 
                   percentage: float, comment: str) -> bool:
        """Сохранение оценки"""
        try:
            with self.get_connection() as conn:
                c = conn.cursor()
                c.execute('''INSERT INTO grades 
                            (student_id, subject, grade, percentage, comment) 
                            VALUES (?, ?, ?, ?, ?)''',
                         (student_id, subject, grade, percentage, comment))
                return True
        except Exception as e:
            logger.error(f"Error saving grade: {str(e)}")
            return False
    
    def get_student_grades(self, student_id: str, limit: int = 50):
        """Получение оценок студента"""
        try:
            with self.get_connection() as conn:
                c = conn.cursor()
                c.execute('''SELECT * FROM grades 
                            WHERE student_id = ? 
                            ORDER BY timestamp DESC 
                            LIMIT ?''', (student_id, limit))
                return c.fetchall()
        except Exception as e:
            logger.error(f"Error fetching grades: {str(e)}")
            return []
    
    def save_error(self, student_id: str, subject: str, error_type: str):
        """Сохранение ошибки"""
        try:
            with self.get_connection() as conn:
                c = conn.cursor()
                c.execute('''INSERT INTO error_history 
                            (student_id, subject, error_type) 
                            VALUES (?, ?, ?)
                            ON CONFLICT(student_id, error_type) 
                            DO UPDATE SET 
                                count = count + 1,
                                last_occurrence = CURRENT_TIMESTAMP''',
                         (student_id, subject, error_type))
                return True
        except Exception as e:
            logger.error(f"Error saving error: {str(e)}")
            return False