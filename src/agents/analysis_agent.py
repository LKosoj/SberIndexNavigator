"""
Агент для интеллектуального анализа данных и генерации рекомендаций.
Проводит статистический анализ, выявляет паттерны и предлагает business insights.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from langchain_openai import ChatOpenAI
from langchain.schema import BaseMessage, HumanMessage, SystemMessage
import json

from src.config.settings import (
    OPENAI_API_KEY, 
    OPENAI_BASE_URL,
    OPENAI_MODEL, 
    OPENAI_TEMPERATURE
)

logger = logging.getLogger(__name__)


class AnalysisAgent:
    """Агент для проведения интеллектуального анализа данных и генерации рекомендаций."""
    
    def __init__(self):
        """Инициализация агента анализа."""
        self.llm = self._initialize_llm()
        
        # Шаблоны промптов для разных типов анализа
        self.prompts = {
            "descriptive": """
Ты - эксперт по анализу данных индексов Сбербанка. Проведи описательный анализ данных.

ЗАДАЧА: Проанализируй данные и предоставь:
1. Ключевые статистические показатели 
2. Основные тренды и паттерны
3. Выбросы и аномалии
4. Распределение данных
5. Практические рекомендации

ДАННЫЕ:
{data_summary}

ВОПРОС ПОЛЬЗОВАТЕЛЯ: {question}

Ответь в структурированном формате JSON:
{{
    "analysis_type": "descriptive",
    "key_insights": ["инсайт1", "инсайт2", "инсайт3"],
    "statistical_summary": {{
        "mean": "среднее значение и его интерпретация",
        "median": "медиана и что она показывает", 
        "std": "стандартное отклонение и что это означает",
        "min_max": "минимум и максимум с их контекстом"
    }},
    "trends": ["тренд1", "тренд2"],
    "anomalies": ["аномалия1", "аномалия2"],
    "business_implications": ["импликация1", "импликация2"],
    "recommendations": ["практическая рекомендация1", "практическая рекомендация2", "практическая рекомендация3"]
}}
""",

            "comparative": """
Ты - эксперт по анализу данных индексов Сбербанка. Проведи сравнительный анализ.

ЗАДАЧА: Сравни данные между регионами/периодами и найди:
1. Лидеров и аутсайдеров
2. Существенные различия
3. Причины различий
4. Рекомендации по улучшению

ДАННЫЕ:
{data_summary}

ВОПРОС ПОЛЬЗОВАТЕЛЯ: {question}

Ответь в структурированном формате JSON:
{{
    "analysis_type": "comparative", 
    "leaders": ["лидер1", "лидер2"],
    "outsiders": ["аутсайдер1", "аутсайдер2"],
    "key_differences": ["различие1", "различие2"],
    "success_factors": ["фактор1", "фактор2"],
    "improvement_areas": ["область1", "область2"],
    "recommendations": ["рекомендация1", "рекомендация2"]
}}
""",

            "correlation": """
Ты - эксперт по анализу данных индексов Сбербанка. Проведи корреляционный анализ.

ЗАДАЧА: Найди взаимосвязи между показателями:
1. Сильные корреляции (>0.7 или <-0.7)
2. Причинно-следственные связи
3. Неожиданные зависимости
4. Практические выводы

ДАННЫЕ:
{data_summary}

КОРРЕЛЯЦИОННАЯ МАТРИЦА:
{correlation_matrix}

ВОПРОС ПОЛЬЗОВАТЕЛЯ: {question}

Ответь в структурированном формате JSON:
{{
    "analysis_type": "correlation",
    "strong_correlations": [
        {{"variables": ["переменная1", "переменная2"], "correlation": 0.85, "interpretation": "интерпретация"}}
    ],
    "causal_relationships": ["связь1", "связь2"],
    "unexpected_findings": ["находка1", "находка2"],
    "actionable_insights": ["инсайт1", "инсайт2"],
    "investment_recommendations": ["рекомендация1", "рекомендация2"]
}}
""",

            "forecasting": """
Ты - эксперт по анализу данных индексов Сбербанка. Проведи прогнозный анализ.

ЗАДАЧА: На основе исторических данных предскажи:
1. Будущие тенденции
2. Потенциальные риски
3. Возможности роста
4. Стратегические рекомендации

ДАННЫЕ:
{data_summary}

ТРЕНДЫ:
{trends}

ВОПРОС ПОЛЬЗОВАТЕЛЯ: {question}

Ответь в структурированном формате JSON:
{{
    "analysis_type": "forecasting",
    "future_trends": ["тренд1", "тренд2"],
    "growth_opportunities": ["возможность1", "возможность2"],
    "potential_risks": ["риск1", "риск2"],
    "strategic_recommendations": ["рекомендация1", "рекомендация2"],
    "timeline": "временные рамки прогноза",
    "confidence_level": "уровень уверенности в прогнозе"
}}
"""
        }
    
    def _initialize_llm(self) -> ChatOpenAI:
        """Инициализация LLM."""
        llm_kwargs = {
            "model": OPENAI_MODEL,
            "temperature": OPENAI_TEMPERATURE,
            "openai_api_key": OPENAI_API_KEY
        }
        
        if OPENAI_BASE_URL:
            llm_kwargs["base_url"] = OPENAI_BASE_URL
        
        return ChatOpenAI(**llm_kwargs)
    
    def analyze_data(self, data: pd.DataFrame, question: str) -> Dict[str, Any]:
        """
        Проведение комплексного анализа данных.
        
        Args:
            data: DataFrame с данными для анализа
            question: Вопрос пользователя для контекста
            
        Returns:
            Результаты анализа с рекомендациями
        """
        try:
            # Определяем тип анализа на основе вопроса
            analysis_type = self._determine_analysis_type(question)
            
            # Проводим статистический анализ
            stats_analysis = self._perform_statistical_analysis(data)
            
            # Проводим анализ на основе типа
            if analysis_type == "descriptive":
                ai_analysis = self._descriptive_analysis(data, question, stats_analysis)
            elif analysis_type == "comparative":
                ai_analysis = self._comparative_analysis(data, question, stats_analysis)
            elif analysis_type == "correlation":
                ai_analysis = self._correlation_analysis(data, question, stats_analysis)
            elif analysis_type == "forecasting":
                ai_analysis = self._forecasting_analysis(data, question, stats_analysis)
            else:
                ai_analysis = self._descriptive_analysis(data, question, stats_analysis)
            
            # Оценка качества данных
            data_quality = self._assess_data_quality(data)
            
            # Генерация рекомендаций
            recommendations = self._generate_recommendations(ai_analysis, stats_analysis)
            
            # Объединяем результаты
            result = {
                "analysis_type": analysis_type,
                "statistical_analysis": stats_analysis,
                "ai_insights": ai_analysis,
                "data_quality": data_quality,
                "recommendations": recommendations,
                "success": True,
                "error": None
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Ошибка анализа данных: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                "analysis_type": "error",
                "error": str(e),
                "success": False
            }
    
    def _determine_analysis_type(self, question: str) -> str:
        """Определение типа анализа на основе вопроса пользователя."""
        question_lower = question.lower()
        
        correlation_keywords = ['корреляция', 'связь', 'зависимость', 'влияние', 'relationship', 'correlation']
        comparative_keywords = ['сравнить', 'лучший', 'худший', 'различия', 'compare', 'difference', 'лидер', 'аутсайдер']
        forecasting_keywords = ['прогноз', 'будущее', 'тренд', 'предсказать', 'forecast', 'predict', 'trend']
        recommendation_keywords = ['рекомендации', 'рекомендация', 'советы', 'что делать', 'как улучшить', 'инвестиции', 'recommendations', 'advice']
        
        if any(keyword in question_lower for keyword in recommendation_keywords):
            return "comparative"  # Comparative analysis лучше подходит для рекомендаций
        elif any(keyword in question_lower for keyword in correlation_keywords):
            return "correlation"
        elif any(keyword in question_lower for keyword in comparative_keywords):
            return "comparative"
        elif any(keyword in question_lower for keyword in forecasting_keywords):
            return "forecasting"
        else:
            return "descriptive"
    
    def _perform_statistical_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Проведение статистического анализа данных."""
        try:
            numeric_data = data.select_dtypes(include=[np.number])
            
            if numeric_data.empty:
                return {"error": "Нет числовых данных для анализа"}
            
            stats_summary = {
                "basic_stats": numeric_data.describe().to_dict(),
                "correlation_matrix": numeric_data.corr().to_dict() if len(numeric_data.columns) > 1 else {},
                "missing_values": data.isnull().sum().to_dict(),
                "data_types": data.dtypes.astype(str).to_dict(),
                "outliers": self._detect_outliers(numeric_data),
                "distribution_analysis": self._analyze_distributions(numeric_data)
            }
            
            # Кластерный анализ если данных достаточно
            if len(numeric_data) > 10 and len(numeric_data.columns) > 1:
                stats_summary["clustering"] = self._perform_clustering(numeric_data)
            
            return stats_summary
            
        except Exception as e:
            logger.error(f"Ошибка статистического анализа: {e}")
            return {"error": str(e)}
    
    def _detect_outliers(self, data: pd.DataFrame) -> Dict[str, List]:
        """Обнаружение выбросов в данных."""
        outliers = {}
        
        for column in data.columns:
            Q1 = data[column].quantile(0.25)
            Q3 = data[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            column_outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)][column].tolist()
            if column_outliers:
                outliers[column] = column_outliers
        
        return outliers
    
    def _analyze_distributions(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Анализ распределения данных."""
        distributions = {}
        
        for column in data.columns:
            if data[column].dtype in ['int64', 'float64']:
                # Тест на нормальность
                _, p_value = stats.normaltest(data[column].dropna())
                
                distributions[column] = {
                    "skewness": float(data[column].skew()),
                    "kurtosis": float(data[column].kurtosis()),
                    "normality_p_value": float(p_value),
                    "is_normal": p_value > 0.05
                }
        
        return distributions
    
    def _perform_clustering(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Кластерный анализ данных."""
        try:
            # Стандартизация данных
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(data.fillna(data.mean()))
            
            # K-means кластеризация
            n_clusters = min(4, len(data) // 3)  # Разумное количество кластеров
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(scaled_data)
            
            # Анализ кластеров
            cluster_analysis = {}
            for i in range(n_clusters):
                cluster_data = data[clusters == i]
                cluster_analysis[f"cluster_{i}"] = {
                    "size": len(cluster_data),
                    "mean_values": cluster_data.mean().to_dict(),
                    "characteristics": self._describe_cluster(cluster_data, data)
                }
            
            return {
                "n_clusters": n_clusters,
                "cluster_labels": clusters.tolist(),
                "cluster_analysis": cluster_analysis,
                "inertia": float(kmeans.inertia_)
            }
            
        except Exception as e:
            logger.error(f"Ошибка кластерного анализа: {e}")
            return {"error": str(e)}
    
    def _describe_cluster(self, cluster_data: pd.DataFrame, full_data: pd.DataFrame) -> str:
        """Описание характеристик кластера."""
        if len(cluster_data) == 0:
            return "Пустой кластер"
        
        # Найдем колонки, где кластер существенно отличается от общего среднего
        differences = []
        for col in cluster_data.select_dtypes(include=[np.number]).columns:
            cluster_mean = cluster_data[col].mean()
            full_mean = full_data[col].mean()
            diff_percent = ((cluster_mean - full_mean) / full_mean) * 100
            
            if abs(diff_percent) > 20:  # Существенное отличие
                direction = "выше" if diff_percent > 0 else "ниже"
                differences.append(f"{col} {direction} среднего на {abs(diff_percent):.1f}%")
        
        if differences:
            return f"Группа с {'; '.join(differences[:3])}"
        else:
            return "Группа со средними показателями"
    
    def _descriptive_analysis(self, data: pd.DataFrame, question: str, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Описательный анализ данных с использованием ИИ."""
        try:
            data_summary = self._create_data_summary(data, stats)
            
            prompt = self.prompts["descriptive"].format(
                data_summary=data_summary,
                question=question
            )
            
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return self._parse_ai_response(response.content)
            
        except Exception as e:
            logger.error(f"Ошибка описательного анализа: {e}")
            return {"error": str(e)}
    
    def _comparative_analysis(self, data: pd.DataFrame, question: str, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Сравнительный анализ данных с использованием ИИ."""
        try:
            data_summary = self._create_data_summary(data, stats)
            
            prompt = self.prompts["comparative"].format(
                data_summary=data_summary,
                question=question
            )
            
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return self._parse_ai_response(response.content)
            
        except Exception as e:
            logger.error(f"Ошибка сравнительного анализа: {e}")
            return {"error": str(e)}
    
    def _correlation_analysis(self, data: pd.DataFrame, question: str, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Корреляционный анализ данных с использованием ИИ."""
        try:
            data_summary = self._create_data_summary(data, stats)
            correlation_matrix = json.dumps(stats.get("correlation_matrix", {}), indent=2)
            
            prompt = self.prompts["correlation"].format(
                data_summary=data_summary,
                correlation_matrix=correlation_matrix,
                question=question
            )
            
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return self._parse_ai_response(response.content)
            
        except Exception as e:
            logger.error(f"Ошибка корреляционного анализа: {e}")
            return {"error": str(e)}
    
    def _forecasting_analysis(self, data: pd.DataFrame, question: str, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Прогнозный анализ данных с использованием ИИ."""
        try:
            data_summary = self._create_data_summary(data, stats)
            trends = self._identify_trends(data)
            
            prompt = self.prompts["forecasting"].format(
                data_summary=data_summary,
                trends=json.dumps(trends, indent=2),
                question=question
            )
            
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return self._parse_ai_response(response.content)
            
        except Exception as e:
            logger.error(f"Ошибка прогнозного анализа: {e}")
            return {"error": str(e)}
    
    def _identify_trends(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Идентификация трендов в данных."""
        trends = {}
        
        # Ищем временные колонки
        time_columns = [col for col in data.columns if 'month' in col.lower() or 'date' in col.lower() or 'time' in col.lower()]
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        if time_columns and len(numeric_columns) > 0:
            time_col = time_columns[0]
            
            for num_col in numeric_columns:
                if num_col != time_col:
                    # Простая линейная регрессия для определения тренда
                    x = np.arange(len(data))
                    y = data[num_col].fillna(data[num_col].mean())
                    
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                    
                    trend_direction = "возрастающий" if slope > 0 else "убывающий"
                    trend_strength = "сильный" if abs(r_value) > 0.7 else "слабый" if abs(r_value) > 0.3 else "отсутствует"
                    
                    trends[num_col] = {
                        "direction": trend_direction,
                        "strength": trend_strength,
                        "slope": float(slope),
                        "r_squared": float(r_value ** 2),
                        "p_value": float(p_value)
                    }
        
        return trends
    
    def _create_data_summary(self, data: pd.DataFrame, stats: Dict[str, Any]) -> str:
        """Создание краткого описания данных для ИИ."""
        summary_parts = [
            f"Данные содержат {len(data)} записей и {len(data.columns)} колонок.",
            f"Колонки: {', '.join(data.columns)}",
        ]
        
        # Добавляем информацию о числовых колонках
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            summary_parts.append(f"Числовые показатели: {', '.join(numeric_cols)}")
            
            # Топ и низ значений
            for col in numeric_cols[:3]:  # Первые 3 колонки
                max_val = data[col].max()
                min_val = data[col].min()
                mean_val = data[col].mean()
                summary_parts.append(f"{col}: мин={min_val:.2f}, среднее={mean_val:.2f}, макс={max_val:.2f}")
        
        # Добавляем информацию о категориальных колонках
        text_cols = data.select_dtypes(include=['object']).columns
        if len(text_cols) > 0:
            summary_parts.append(f"Категориальные: {', '.join(text_cols)}")
            
            for col in text_cols[:2]:  # Первые 2 колонки
                unique_vals = data[col].unique()
                summary_parts.append(f"{col}: {len(unique_vals)} уникальных значений ({', '.join(map(str, unique_vals[:5]))})")
        
        return " ".join(summary_parts)
    
    def _parse_ai_response(self, response: str) -> Dict[str, Any]:
        """Парсинг ответа ИИ в JSON."""
        try:
            # Убираем возможные markdown блоки
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]
            
            return json.loads(response.strip())
            
        except json.JSONDecodeError as e:
            logger.error(f"Ошибка парсинга JSON ответа: {e}")
            return {
                "error": "Ошибка парсинга ответа ИИ",
                "raw_response": response
            }
    
    def _assess_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Оценка качества данных."""
        total_cells = len(data) * len(data.columns)
        missing_cells = data.isnull().sum().sum()
        missing_percentage = (missing_cells / total_cells) * 100
        
        quality_score = 100 - missing_percentage
        if quality_score >= 95:
            quality_level = "Отличное"
        elif quality_score >= 80:
            quality_level = "Хорошее"
        elif quality_score >= 60:
            quality_level = "Удовлетворительное"
        else:
            quality_level = "Требует внимания"
        
        return {
            "quality_score": round(quality_score, 2),
            "quality_level": quality_level,
            "missing_percentage": round(missing_percentage, 2),
            "total_records": len(data),
            "issues": self._identify_data_issues(data)
        }
    
    def _identify_data_issues(self, data: pd.DataFrame) -> List[str]:
        """Идентификация проблем в данных."""
        issues = []
        
        # Проверка пропущенных значений
        missing_cols = data.columns[data.isnull().any()].tolist()
        if missing_cols:
            issues.append(f"Пропущенные значения в колонках: {', '.join(missing_cols)}")
        
        # Проверка дубликатов
        duplicates = data.duplicated().sum()
        if duplicates > 0:
            issues.append(f"Обнаружено {duplicates} дублирующихся записей")
        
        # Проверка выбросов
        numeric_data = data.select_dtypes(include=[np.number])
        outlier_cols = []
        for col in numeric_data.columns:
            Q1 = numeric_data[col].quantile(0.25)
            Q3 = numeric_data[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((numeric_data[col] < Q1 - 1.5 * IQR) | (numeric_data[col] > Q3 + 1.5 * IQR)).sum()
            if outliers > len(numeric_data) * 0.05:  # Более 5% выбросов
                outlier_cols.append(col)
        
        if outlier_cols:
            issues.append(f"Значительное количество выбросов в: {', '.join(outlier_cols)}")
        
        return issues
    
    def _generate_recommendations(self, ai_analysis: Dict[str, Any], stats_analysis: Dict[str, Any]) -> List[str]:
        """Генерация итоговых рекомендаций."""
        recommendations = []
        
        # Извлекаем рекомендации из AI анализа
        recommendation_fields = [
            "recommendations", 
            "investment_recommendations", 
            "strategic_recommendations"
        ]
        
        for field in recommendation_fields:
            if field in ai_analysis and isinstance(ai_analysis[field], list):
                recommendations.extend(ai_analysis[field])
        
        # Дополнительные рекомендации на основе статистического анализа
        if "outliers" in stats_analysis and stats_analysis["outliers"]:
            recommendations.append("Исследовать выявленные выбросы для понимания их природы и причин")
        
        if "clustering" in stats_analysis and "cluster_analysis" in stats_analysis["clustering"]:
            recommendations.append("Использовать кластерный анализ для сегментации регионов и персонализации подходов")
        
        # Рекомендации на основе качества данных
        if "data_quality" in stats_analysis:
            quality_score = stats_analysis.get("data_quality", {}).get("quality_score", 100)
            if quality_score < 80:
                recommendations.append("Улучшить качество сбора и обработки данных для повышения точности анализа")
        
        # Убираем дубликаты и ограничиваем количество
        unique_recommendations = list(dict.fromkeys(recommendations))  # Сохраняет порядок
        final_recommendations = unique_recommendations[:5]  # Топ-5 рекомендаций
        
        return final_recommendations


# Singleton instance
analysis_agent: Optional[AnalysisAgent] = None


def get_analysis_agent() -> AnalysisAgent:
    """
    Получение единственного экземпляра AnalysisAgent.
    
    Returns:
        Экземпляр AnalysisAgent
    """
    global analysis_agent
    if analysis_agent is None:
        analysis_agent = AnalysisAgent()
    return analysis_agent 