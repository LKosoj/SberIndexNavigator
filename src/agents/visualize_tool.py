"""
Инструмент для автоматического выбора типа визуализации данных.
Анализирует структуру данных и определяет оптимальный способ отображения.
"""

import logging
from typing import Dict, Any, Optional, List
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from src.config.settings import (
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENAI_MODEL,
    OPENAI_TEMPERATURE,
    VISUALIZATION_TOOL_PROMPT
)

logger = logging.getLogger(__name__)


class VisualizationInput(BaseModel):
    """Входные данные для инструмента визуализации."""
    data: str = Field(description="Данные в формате CSV или описание структуры данных")
    question: str = Field(description="Вопрос пользователя для контекста")


class VisualizeTool(BaseTool):
    """Инструмент для определения оптимального типа визуализации."""
    
    name: str = "visualize_tool"
    description: str = "Определяет оптимальный тип визуализации для данных"
    args_schema: type[BaseModel] = VisualizationInput
    llm: Optional[ChatOpenAI] = None
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self):
        """Инициализация инструмента визуализации."""
        super().__init__()
        
        llm_kwargs = {
            "model": OPENAI_MODEL,
            "temperature": OPENAI_TEMPERATURE,
            "openai_api_key": OPENAI_API_KEY
        }
        
        # Добавляем base_url если он задан
        if OPENAI_BASE_URL:
            llm_kwargs["base_url"] = OPENAI_BASE_URL
        
        self.llm = ChatOpenAI(**llm_kwargs)
    
    def _run(self, data: str, question: str) -> str:
        """
        Определение типа визуализации для данных.
        
        Args:
            data: Данные в формате CSV или описание
            question: Вопрос пользователя
            
        Returns:
            Тип визуализации: line, bar, scatter, map, table
        """
        try:
            # Формируем промпт для определения типа визуализации
            prompt = f"""
            {VISUALIZATION_TOOL_PROMPT}
            
            Вопрос пользователя: {question}
            
            Данные:
            {data}
            
            Определи оптимальный тип визуализации и ответь одним словом.
            """
            
            response = self.llm.invoke(prompt)
            chart_type = response.content.strip().lower()
            
            # Валидация типа визуализации
            valid_types = ["line", "bar", "scatter", "map", "table"]
            if chart_type not in valid_types:
                chart_type = "table"  # fallback
            
            logger.info(f"Определен тип визуализации: {chart_type}")
            return chart_type
            
        except Exception as e:
            logger.error(f"Ошибка определения типа визуализации: {e}")
            return "table"  # fallback
    
    async def _arun(self, data: str, question: str) -> str:
        """Асинхронная версия _run."""
        return self._run(data, question)


class VisualizationAnalyzer:
    """Анализатор для определения типа визуализации данных."""
    
    def __init__(self):
        """Инициализация анализатора."""
        self.tool = VisualizeTool()
    
    def analyze_data_structure(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Анализ структуры данных для определения характеристик.
        
        Args:
            df: DataFrame для анализа
            
        Returns:
            Словарь с характеристиками данных
        """
        try:
            analysis = {
                "row_count": len(df),
                "column_count": len(df.columns),
                "columns": list(df.columns),
                "numeric_columns": list(df.select_dtypes(include=['number']).columns),
                "text_columns": list(df.select_dtypes(include=['object']).columns),
                "has_time_data": False,
                "has_geographic_data": False,
                "has_multiple_numeric": len(df.select_dtypes(include=['number']).columns) > 1
            }
            
            # Проверка на временные данные
            time_keywords = ['date', 'time', 'month', 'year', 'день', 'месяц', 'год']
            analysis["has_time_data"] = any(
                any(keyword in col.lower() for keyword in time_keywords)
                for col in df.columns
            )
            
            # Проверка на географические данные
            geo_keywords = ['region', 'city', 'lat', 'lon', 'регион', 'город', 'координат']
            analysis["has_geographic_data"] = any(
                any(keyword in col.lower() for keyword in geo_keywords)
                for col in df.columns
            )
            
            return analysis
            
        except Exception as e:
            logger.error(f"Ошибка анализа структуры данных: {e}")
            return {}
    
    def determine_chart_type(self, df: pd.DataFrame, question: str) -> str:
        """
        Определение оптимального типа визуализации.
        
        Args:
            df: DataFrame с данными
            question: Вопрос пользователя
            
        Returns:
            Тип визуализации
        """
        try:
            # Анализируем структуру данных
            structure = self.analyze_data_structure(df)
            
            # Правила для автоматического определения
            if structure.get("has_time_data") and len(structure.get("numeric_columns", [])) > 0:
                return "line"
            
            if structure.get("has_geographic_data"):
                return "map"
            
            if len(structure.get("numeric_columns", [])) >= 2:
                # Проверяем на корреляцию в вопросе
                correlation_keywords = ['корреляция', 'зависимость', 'связь', 'correlation']
                if any(keyword in question.lower() for keyword in correlation_keywords):
                    return "scatter"
            
            if len(structure.get("text_columns", [])) > 0 and len(structure.get("numeric_columns", [])) > 0:
                return "bar"
            
            # Используем LLM для сложных случаев
            data_sample = df.head(3).to_string()
            return self.tool._run(data_sample, question)
            
        except Exception as e:
            logger.error(f"Ошибка определения типа визуализации: {e}")
            return "table"
    
    def get_visualization_config(self, chart_type: str, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Получение конфигурации для визуализации.
        
        Args:
            chart_type: Тип визуализации
            df: DataFrame с данными
            
        Returns:
            Конфигурация для создания визуализации
        """
        try:
            config = {
                "type": chart_type,
                "data": df,
                "title": "Анализ данных индексов Сбербанка",
                "x_column": None,
                "y_column": None,
                "color_column": None,
                "size_column": None
            }
            
            numeric_cols = list(df.select_dtypes(include=['number']).columns)
            text_cols = list(df.select_dtypes(include=['object']).columns)
            
            if chart_type == "line":
                # Для линейного графика ищем временную колонку
                time_cols = [col for col in df.columns if any(
                    keyword in col.lower() for keyword in ['month', 'date', 'time', 'месяц', 'год']
                )]
                config["x_column"] = time_cols[0] if time_cols else text_cols[0] if text_cols else None
                
                # Для y_column исключаем id, year и другие служебные колонки
                value_cols = [col for col in numeric_cols if not any(
                    keyword in col.lower() for keyword in ['year', 'id', 'code', '_id', 'год']
                )]
                config["y_column"] = value_cols[0] if value_cols else numeric_cols[0] if numeric_cols else None
                
            elif chart_type == "bar":
                config["x_column"] = text_cols[0] if text_cols else None
                
                # Для bar chart также исключаем служебные колонки
                value_cols = [col for col in numeric_cols if not any(
                    keyword in col.lower() for keyword in ['year', 'id', 'code', '_id', 'год']
                )]
                config["y_column"] = value_cols[0] if value_cols else numeric_cols[0] if numeric_cols else None
                
            elif chart_type == "scatter":
                # Для scatter исключаем служебные колонки
                value_cols = [col for col in numeric_cols if not any(
                    keyword in col.lower() for keyword in ['year', 'id', 'code', '_id', 'год']
                )]
                config["x_column"] = value_cols[0] if len(value_cols) > 0 else numeric_cols[0] if len(numeric_cols) > 0 else None
                config["y_column"] = value_cols[1] if len(value_cols) > 1 else numeric_cols[1] if len(numeric_cols) > 1 else None
                config["color_column"] = text_cols[0] if text_cols else None
                
            elif chart_type == "map":
                # Для карты ищем географические колонки
                geo_cols = [col for col in df.columns if any(
                    keyword in col.lower() for keyword in ['region', 'city', 'регион', 'город']
                )]
                config["location_column"] = geo_cols[0] if geo_cols else None
                
                # Для значений также исключаем служебные колонки
                value_cols = [col for col in numeric_cols if not any(
                    keyword in col.lower() for keyword in ['year', 'id', 'code', '_id', 'год']
                )]
                config["value_column"] = value_cols[0] if value_cols else numeric_cols[0] if numeric_cols else None
            
            return config
            
        except Exception as e:
            logger.error(f"Ошибка создания конфигурации визуализации: {e}")
            return {"type": "table", "data": df}


# Singleton instance
visualization_analyzer: Optional[VisualizationAnalyzer] = None


def get_visualization_analyzer() -> VisualizationAnalyzer:
    """
    Получение единственного экземпляра VisualizationAnalyzer.
    
    Returns:
        Экземпляр VisualizationAnalyzer
    """
    global visualization_analyzer
    if visualization_analyzer is None:
        visualization_analyzer = VisualizationAnalyzer()
    return visualization_analyzer 