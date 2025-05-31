"""
SQL-агент для генерации и выполнения SQL-запросов к базе данных индексов Сбербанка.
Использует LangChain для интеллектуального анализа пользовательских запросов.
"""

import logging
from typing import Dict, Any, Optional, List
import pandas as pd
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain.agents.agent_types import AgentType
from langchain.schema import AgentAction, AgentFinish
import duckdb
import tempfile
import shutil
import os

from src.config.settings import (
    OPENAI_API_KEY, 
    OPENAI_BASE_URL,
    OPENAI_MODEL, 
    OPENAI_TEMPERATURE,
    SQL_AGENT_SYSTEM_PROMPT,
    MAX_RETRIES
)
from src.data.database import get_database_manager

logger = logging.getLogger(__name__)


class SqlAgent:
    """SQL-агент для интеллектуального анализа данных индексов Сбербанка."""
    
    def __init__(self):
        """Инициализация SQL-агента."""
        self.db_manager = get_database_manager()
        self.llm = None
        self.agent = None
        self._initialize_agent()
    
    def _initialize_agent(self) -> None:
        """Инициализация LangChain агента."""
        try:
            if not OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY не установлен")
            
            # Инициализация LLM
            llm_kwargs = {
                "model": OPENAI_MODEL,
                "temperature": OPENAI_TEMPERATURE,
                "openai_api_key": OPENAI_API_KEY
            }
            
            # Добавляем base_url если он задан
            if OPENAI_BASE_URL:
                llm_kwargs["base_url"] = OPENAI_BASE_URL
            
            self.llm = ChatOpenAI(**llm_kwargs)
            
            # Создание SQLDatabase объекта для LangChain
            # Используем in-memory DuckDB для избежания конфликтов
            # Скопируем данные из основной базы
            
            # Создаем временную копию базы данных для LangChain
            temp_db = tempfile.mktemp(suffix='.db')
            shutil.copy2(self.db_manager.database_path, temp_db)
            
            # Создаем подключение для LangChain
            db_uri = f"duckdb:///{temp_db}"
            self.sql_db = SQLDatabase.from_uri(db_uri)
            self.temp_db_path = temp_db
            
            # Создание toolkit
            toolkit = SQLDatabaseToolkit(db=self.sql_db, llm=self.llm)
            
            # Создание агента
            self.agent = create_sql_agent(
                llm=self.llm,
                toolkit=toolkit,
                agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True,
                max_iterations=3,
                max_execution_time=30,
                early_stopping_method="generate"
            )
            
            logger.info("SQL-агент инициализирован успешно")
            
        except Exception as e:
            logger.error(f"Ошибка инициализации SQL-агента: {e}")
            raise
    
    def generate_sql_query(self, user_question: str) -> str:
        """
        Генерация SQL-запроса на основе вопроса пользователя.
        
        Args:
            user_question: Вопрос пользователя на естественном языке
            
        Returns:
            SQL-запрос в виде строки
        """
        try:
            # Формируем промпт для генерации SQL
            prompt = f"""
            {SQL_AGENT_SYSTEM_PROMPT}
            
            Вопрос пользователя: {user_question}
            
            Сгенерируй SQL-запрос для ответа на этот вопрос. 
            Верни *ТОЛЬКО* SQL-запрос без дополнительных объяснений.
            """
            
            # Используем LLM для генерации SQL
            response = self.llm.invoke(prompt)
            sql_query = response.content.strip()
            
            # Очищаем SQL от markdown форматирования
            if sql_query.startswith("```sql"):
                sql_query = sql_query[6:]
            if sql_query.startswith("```"):
                sql_query = sql_query[3:]
            if sql_query.endswith("```"):
                sql_query = sql_query[:-3]
            
            sql_query = sql_query.strip()
            
            logger.info(f"Сгенерирован SQL-запрос: {sql_query}")
            return sql_query
            
        except Exception as e:
            logger.error(f"Ошибка генерации SQL-запроса: {e}")
            raise
    
    def execute_query(self, sql_query: str) -> pd.DataFrame:
        """
        Выполнение SQL-запроса и возврат результата.
        
        Args:
            sql_query: SQL-запрос для выполнения
            
        Returns:
            DataFrame с результатами запроса
        """
        try:
            # Валидация запроса
            if not self.db_manager.validate_query(sql_query):
                raise ValueError("Невалидный SQL-запрос")
            
            # Выполнение запроса
            result = self.db_manager.execute_query(sql_query)
            
            logger.info(f"SQL-запрос выполнен успешно, получено {len(result)} строк")
            return result
            
        except Exception as e:
            logger.error(f"Ошибка выполнения SQL-запроса: {e}")
            raise
    
    def analyze_question(self, user_question: str) -> Dict[str, Any]:
        """
        Полный анализ вопроса пользователя: генерация SQL и выполнение.
        
        Args:
            user_question: Вопрос пользователя на естественном языке
            
        Returns:
            Словарь с результатами анализа
        """
        try:
            # Генерируем SQL-запрос
            sql_query = self.generate_sql_query(user_question)
            
            # Выполняем запрос
            data = self.execute_query(sql_query)
            
            # Формируем результат
            result = {
                "question": user_question,
                "sql_query": sql_query,
                "data": data,
                "success": True,
                "error": None
            }
            
            logger.info(f"Анализ вопроса завершен успешно")
            return result
            
        except Exception as e:
            logger.error(f"Ошибка анализа вопроса: {e}")
            return {
                "question": user_question,
                "sql_query": None,
                "data": pd.DataFrame(),
                "success": False,
                "error": str(e)
            }
    
    def get_table_schema(self) -> Dict[str, List[str]]:
        """
        Получение схемы таблиц базы данных.
        
        Returns:
            Словарь с информацией о таблицах и колонках
        """
        return self.db_manager.get_table_info()
    
    def suggest_questions(self) -> List[str]:
        """
        Предложение примеров вопросов на основе доступных данных.
        
        Returns:
            Список примеров вопросов
        """
        from src.config.settings import DEMO_QUESTIONS
        return DEMO_QUESTIONS

    def __del__(self):
        """Очистка временных файлов при удалении объекта."""
        if hasattr(self, 'temp_db_path') and os.path.exists(self.temp_db_path):
            try:
                os.unlink(self.temp_db_path)
            except:
                pass


# Singleton instance для использования в приложении
sql_agent: Optional[SqlAgent] = None


def get_sql_agent() -> SqlAgent:
    """
    Получение единственного экземпляра SqlAgent.
    
    Returns:
        Экземпляр SqlAgent
    """
    global sql_agent
    if sql_agent is None:
        sql_agent = SqlAgent()
    return sql_agent 