"""
Enhanced grading system with configurable rubrics and detailed analytics
Система оценивания с настраиваемыми рубриками и детальной аналитикой
"""
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum

class GradeScale(Enum):
    FIVE_POINT = "5_point"
    PERCENTAGE = "percentage"
    LETTER = "letter"

@dataclass
class GradingRubric:
    """Configurable grading rubric / Настраиваемая система оценивания"""
    scale: GradeScale
    thresholds: Dict[str, float]
    weights: Dict[str, float] = None
    
    def calculate_grade(self, score: float, max_score: float = 100) -> tuple:
        """Calculate grade based on rubric"""
        percentage = (score / max_score) * 100 if max_score > 0 else 0
        
        if self.scale == GradeScale.FIVE_POINT:
            for grade, threshold in sorted(
                self.thresholds.items(), 
                key=lambda x: x[1], 
                reverse=True
            ):
                if percentage >= threshold:
                    return int(grade), percentage
            return 2, percentage
        
        elif self.scale == GradeScale.PERCENTAGE:
            return percentage, percentage
        
        elif self.scale == GradeScale.LETTER:
            for grade, threshold in sorted(
                self.thresholds.items(),
                key=lambda x: x[1],
                reverse=True
            ):
                if percentage >= threshold:
                    return grade, percentage
            return 'F', percentage

class GradingSystem:
    """Comprehensive grading system / Комплексная система оценивания"""
    
    def __init__(self):
        self.rubrics = {
            'math': GradingRubric(
                scale=GradeScale.FIVE_POINT,
                thresholds={'5': 90, '4': 75, '3': 60, '2': 0}
            ),
            'english': GradingRubric(
                scale=GradeScale.FIVE_POINT,
                thresholds={'5': 95, '4': 85, '3': 70, '2': 0}
            ),
            'test': GradingRubric(
                scale=GradeScale.FIVE_POINT,
                thresholds={'5': 90, '4': 75, '3': 60, '2': 0}
            )
        }
    
    def grade_math(self, result: Dict) -> Dict:
        """Grade math assignment / Оценить задание по математике"""
        score = 100 if result.get('is_correct', False) else 0
        grade, percentage = self.rubrics['math'].calculate_grade(score)
        
        feedback = self._generate_math_feedback(result, grade)
        
        return {
            'grade': grade,
            'percentage': percentage,
            'feedback': feedback,
            'strengths': self._identify_strengths(result),
            'areas_for_improvement': self._identify_weaknesses(result)
        }
    
    def grade_english(self, errors: List[Dict], word_count: int) -> Dict:
        """Grade English assignment / Оценить задание по английскому"""
        error_count = len(errors)
        
        if word_count > 0:
            error_rate = (error_count / word_count) * 100
            score = max(0, 100 - (error_rate * 10))
        else:
            score = 0
        
        grade, percentage = self.rubrics['english'].calculate_grade(score)
        
        error_categories = self._categorize_errors(errors)
        
        return {
            'grade': grade,
            'percentage': percentage,
            'error_count': error_count,
            'error_rate': error_rate if word_count > 0 else 0,
            'error_categories': error_categories,
            'feedback': self._generate_english_feedback(error_categories, grade)
        }
    
    def grade_test(self, results: List[Dict]) -> Dict:
        """Grade multiple choice test / Оценить тест с множественным выбором"""
        if not results:
            return {'grade': 0, 'percentage': 0, 'feedback': 'No results'}
        
        correct = sum(1 for r in results if r.get('is_correct', False))
        total = len(results)
        percentage = (correct / total) * 100 if total > 0 else 0
        
        grade, _ = self.rubrics['test'].calculate_grade(percentage)
        
        topic_analysis = self._analyze_by_topic(results)
        
        return {
            'grade': grade,
            'percentage': percentage,
            'correct': correct,
            'total': total,
            'topic_analysis': topic_analysis,
            'feedback': self._generate_test_feedback(percentage, topic_analysis)
        }
    
    def _generate_math_feedback(self, result: Dict, grade: int) -> str:
        """Generate personalized math feedback"""
        if grade == 5:
            return "Отличная работа! Ваше решение демонстрирует глубокое понимание математики."
        elif grade == 4:
            return "Хорошая работа. Есть небольшие ошибки, но общий подход правильный."
        elif grade == 3:
            return "Удовлетворительно. Рекомендуется повторить решение для улучшения понимания."
        else:
            return "Требуется улучшение. Сосредоточьтесь на фундаментальных концепциях."
    
    def _generate_english_feedback(self, error_cats: Dict, grade: int) -> str:
        """Generate personalized English feedback"""
        feedback = []
        
        if grade >= 4:
            feedback.append("Сильное письмо с минимальными ошибками.")
        
        if error_cats.get('grammar', 0) > 3:
            feedback.append("Обратите внимание на грамматические основы.")
        
        if error_cats.get('spelling', 0) > 2:
            feedback.append("Будьте внимательнее к правописанию.")
        
        if error_cats.get('punctuation', 0) > 2:
            feedback.append("Повторите правила пунктуации.")
        
        return " ".join(feedback) if feedback else "Хорошая работа в целом."
    
    def _generate_test_feedback(self, percentage: float, topic_analysis: Dict) -> str:
        """Generate test feedback with topic insights"""
        feedback = []
        
        if percentage >= 90:
            feedback.append("Выдающийся результат!")
        elif percentage >= 75:
            feedback.append("Хорошее понимание материала.")
        elif percentage >= 60:
            feedback.append("Удовлетворительно, но требуется повторение ключевых концепций.")
        else:
            feedback.append("Необходим значительный пересмотр материала курса.")
        
        weak_topics = [
            topic for topic, data in topic_analysis.items()
            if isinstance(data, dict) and data.get('percentage', 100) < 60
        ]
        
        if weak_topics:
            feedback.append(f"Сосредоточьтесь на: {', '.join(weak_topics)}")
        
        return " ".join(feedback)
    
    def _categorize_errors(self, errors: List[Dict]) -> Dict[str, int]:
        """Categorize English errors / Категоризировать ошибки английского"""
        categories = {}
        
        for error in errors:
            rule_id = error.get('ruleId', 'unknown')
            
            if 'SPELL' in rule_id or 'MORFOLOGIK' in rule_id:
                cat = 'spelling'
            elif 'VERB' in rule_id or 'AGREEMENT' in rule_id:
                cat = 'grammar'
            elif 'PUNCT' in rule_id or 'COMMA' in rule_id:
                cat = 'punctuation'
            elif 'UPPERCASE' in rule_id:
                cat = 'capitalization'
            else:
                cat = 'other'
            
            categories[cat] = categories.get(cat, 0) + 1
        
        return categories
    
    def _analyze_by_topic(self, results: List[Dict]) -> Dict:
        """Analyze test results by topic"""
        return {
            'overall': {
                'correct': sum(1 for r in results if r.get('is_correct', False)),
                'total': len(results),
                'percentage': (sum(1 for r in results if r.get('is_correct', False)) / len(results)) * 100 if results else 0
            }
        }
    
    def _identify_strengths(self, result: Dict) -> List[str]:
        """Identify student strengths"""
        strengths = []
        
        if result.get('is_correct'):
            strengths.append("Правильный подход к решению")
            strengths.append("Сильные навыки решения задач")
        
        return strengths
    
    def _identify_weaknesses(self, result: Dict) -> List[str]:
        """Identify areas for improvement"""
        weaknesses = []
        
        if not result.get('is_correct'):
            weaknesses.append("Повторите фундаментальные концепции")
            weaknesses.append("Практикуйте подобные задачи")
        
        return weaknesses

class LearningAnalytics:
    """Advanced learning analytics / Продвинутая аналитика обучения"""
    
    @staticmethod
    def calculate_progress_trend(grades: pd.DataFrame) -> Dict:
        """Calculate learning progress over time"""
        if grades.empty or len(grades) < 2:
            return {'trend': 'insufficient_data'}
        
        grades = grades.copy()
        grades['date'] = pd.to_datetime(grades.get('timestamp', grades.get('date', pd.Timestamp.now())))
        grades = grades.sort_values('date')
        
        grades['moving_avg'] = grades['grade'].rolling(window=min(5, len(grades)), min_periods=1).mean()
        
        recent = grades.tail(min(5, len(grades)))['grade'].mean()
        earlier = grades.head(min(5, len(grades)))['grade'].mean()
        
        if recent > earlier + 0.5:
            trend = 'improving'
        elif recent < earlier - 0.5:
            trend = 'declining'
        else:
            trend = 'stable'
        
        return {
            'trend': trend,
            'recent_avg': round(recent, 2),
            'overall_avg': round(grades['grade'].mean(), 2),
            'best_grade': int(grades['grade'].max()),
            'improvement_rate': round((recent - earlier) / len(grades), 3) if len(grades) > 0 else 0
        }
    
    @staticmethod
    def identify_struggle_areas(error_history: pd.DataFrame) -> List[Dict]:
        """Identify topics where student struggles most"""
        if error_history.empty:
            return []
        
        struggle_areas = error_history.groupby('error_type').agg({
            'count': 'sum',
            'last_occurrence': 'max'
        }).reset_index()
        
        struggle_areas = struggle_areas.sort_values('count', ascending=False)
        
        return struggle_areas.head(5).to_dict('records')