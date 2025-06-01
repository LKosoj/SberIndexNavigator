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
import re

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
            
            # Инициализация LLM с отключенным streaming для агентов
            llm_kwargs = {
                "model": OPENAI_MODEL,
                "temperature": OPENAI_TEMPERATURE,
                "openai_api_key": OPENAI_API_KEY,
                "streaming": False,  # ВАЖНО: отключаем streaming для агентов
                "verbose": False     # Уменьшаем вербозность для стабильности
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
            
            # Создание агента с оптимизированными настройками
            self.agent = create_sql_agent(
                llm=self.llm,
                toolkit=toolkit,
                agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True,
                max_iterations=5,  # Увеличиваем количество итераций для более сложных запросов
                max_execution_time=60,  # Увеличиваем время для обработки
                early_stopping_method="force",  # Используем поддерживаемый метод
                return_intermediate_steps=True,  # Возвращаем промежуточные шаги для отладки
                handle_parsing_errors=True,  # Обрабатываем ошибки парсинга
                agent_executor_kwargs={
                    "handle_parsing_errors": True,
                    "return_intermediate_steps": True
                }
            )
            
            logger.info("SQL-агент инициализирован успешно")
            
        except Exception as e:
            logger.error(f"Ошибка инициализации SQL-агента: {e}")
            raise
    
    def generate_sql_query(self, user_question: str, attempt: int = 1) -> str:
        """
        Генерация SQL-запроса на основе вопроса пользователя с retry логикой.
        
        Args:
            user_question: Вопрос пользователя на естественном языке
            attempt: Номер попытки генерации (для логирования)
            
        Returns:
            SQL-запрос в виде строки
        """
        try:
            # Получаем актуальную схему базы данных
            table_info = self.db_manager.get_table_info()
            schema_info = "\n".join([
                f"Таблица {table}: {', '.join(columns)}" 
                for table, columns in table_info.items()
            ])
            
            # Формируем промпт для генерации SQL
            base_prompt = f"""
            {SQL_AGENT_SYSTEM_PROMPT}
            
            АКТУАЛЬНАЯ СХЕМА БАЗЫ ДАННЫХ:
            {schema_info}
            
            Вопрос пользователя: {user_question}
            
            Сгенерируй SQL-запрос для ответа на этот вопрос. 
            ИСПОЛЬЗУЙ ТОЛЬКО СУЩЕСТВУЮЩИЕ ТАБЛИЦЫ И КОЛОНКИ ИЗ СХЕМЫ ВЫШЕ!
            Верни *ТОЛЬКО* SQL-запрос без дополнительных объяснений, комментариев и markdown форматирования.
            Не используй тройные кавычки или другие декораторы.
            """
            
            # Добавляем дополнительные инструкции при повторных попытках
            if attempt > 1:
                base_prompt += f"""
                
                ВАЖНО: Это попытка номер {attempt}. Предыдущие попытки были неуспешными.
                Убедись что SQL-запрос:
                - Использует только существующие таблицы и колонки ИЗ СХЕМЫ ВЫШЕ
                - Имеет правильный синтаксис
                - Не содержит markdown форматирования
                - Не содержит объяснений или комментариев
                """
            
            # Используем LLM для генерации SQL
            response = self.llm.invoke(base_prompt)
            sql_query = response.content.strip()
            
            # Очищаем SQL от markdown форматирования
            sql_query = self.db_manager.clean_sql_query(sql_query)
            
            logger.info(f"Сгенерирован SQL-запрос (попытка {attempt}): {sql_query}")
            return sql_query
            
        except Exception as e:
            logger.error(f"Ошибка генерации SQL-запроса (попытка {attempt}): {e}")
            raise
    
    def execute_query_with_retry(self, sql_query: str) -> pd.DataFrame:
        """
        Выполнение SQL-запроса с обработкой ошибок и retry логикой.
        
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
            logger.error(f"Проблемный запрос: {sql_query}")
            raise
    
    def execute_query(self, sql_query: str) -> pd.DataFrame:
        """
        Выполнение SQL-запроса и возврат результата.
        
        Args:
            sql_query: SQL-запрос для выполнения
            
        Returns:
            DataFrame с результатами запроса
        """
        return self.execute_query_with_retry(sql_query)
    
    def analyze_question(self, user_question: str) -> Dict[str, Any]:
        """
        Полный анализ вопроса пользователя: генерация SQL и выполнение с retry логикой.
        
        Args:
            user_question: Вопрос пользователя на естественном языке
            
        Returns:
            Словарь с результатами анализа
        """
        last_error = None
        last_sql_query = None
        
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                logger.info(f"Анализ вопроса, попытка {attempt}/{MAX_RETRIES}")
                
                # Генерируем SQL-запрос
                sql_query = self.generate_sql_query(user_question, attempt)
                last_sql_query = sql_query
                
                # Выполняем запрос
                data = self.execute_query_with_retry(sql_query)
                
                # Формируем успешный результат
                result = {
                    "question": user_question,
                    "sql_query": sql_query,
                    "data": data,
                    "success": True,
                    "error": None,
                    "attempts": attempt
                }
                
                logger.info(f"Анализ вопроса завершен успешно с {attempt} попытки")
                return result
                
            except Exception as e:
                last_error = e
                error_msg = str(e)
                
                logger.warning(f"Попытка {attempt}/{MAX_RETRIES} неуспешна: {error_msg}")
                
                # Если это последняя попытка, возвращаем ошибку
                if attempt == MAX_RETRIES:
                    logger.error(f"Все {MAX_RETRIES} попытки исчерпаны для вопроса: {user_question}")
                    break
                
                # Проверяем тип ошибки для определения стратегии retry
                if "syntax error" in error_msg.lower() or "parser error" in error_msg.lower():
                    logger.info(f"Обнаружена ошибка синтаксиса, перегенерируем запрос...")
                    continue
                elif "невалидный" in error_msg.lower() or "invalid" in error_msg.lower():
                    logger.info(f"Обнаружена ошибка валидации, перегенерируем запрос...")
                    continue
                else:
                    # Для других ошибок тоже пробуем еще раз
                    logger.info(f"Обнаружена ошибка: {error_msg}, перегенерируем запрос...")
                    continue
        
        # Если все попытки неуспешны
        logger.error(f"Не удалось проанализировать вопрос после {MAX_RETRIES} попыток")
        return {
            "question": user_question,
            "sql_query": last_sql_query,
            "data": pd.DataFrame(),
            "success": False,
            "error": f"Не удалось сгенерировать корректный SQL после {MAX_RETRIES} попыток. Последняя ошибка: {str(last_error)}",
            "attempts": MAX_RETRIES
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

    def analyze_question_with_agent(self, user_question: str) -> Dict[str, Any]:
        """
        Анализ вопроса пользователя с использованием SQLDatabaseToolkit агента.
        
        Args:
            user_question: Вопрос пользователя на естественном языке
            
        Returns:
            Словарь с результатами анализа
        """
        try:
            logger.info(f"Анализ вопроса с помощью SQL агента: {user_question}")
            
            # Получаем актуальную схему базы данных
            table_info = self.db_manager.get_table_info()
            schema_info = "\n".join([
                f"Таблица {table}: {', '.join(columns)}" 
                for table, columns in table_info.items()
            ])
            
            # ИСПОЛЬЗУЕМ ТОТ ЖЕ ПРОМПТ, что и в generate_sql_query для консистентности
            agent_prompt = f"""
            {SQL_AGENT_SYSTEM_PROMPT}
            
            АКТУАЛЬНАЯ СХЕМА БАЗЫ ДАННЫХ:
            {schema_info}
            
            Вопрос пользователя: {user_question}
            
            Найди ответ на этот вопрос, используя SQL запросы к базе данных.
            Можешь выполнить несколько запросов для получения полной картины.
            В конце предоставь анализ результатов на русском языке.
            """
            
            # Выполняем запрос через агента
            result = self.agent.invoke({"input": agent_prompt})
            
            # Извлекаем SQL запрос из результата агента (если он есть)
            sql_query = self._extract_sql_from_agent_result(result)
            
            # Если агент вернул данные напрямую, используем их
            if 'output' in result:
                agent_output = result['output']
                
                # Пытаемся извлечь данные из ответа агента
                data = self._extract_data_from_agent_output(agent_output, sql_query)
                
                return {
                    "question": user_question,
                    "sql_query": sql_query,
                    "data": data,
                    "success": True,
                    "error": None,
                    "agent_output": agent_output,
                    "method": "sql_toolkit_agent"
                }
            else:
                raise ValueError("Агент не вернул ожидаемый результат")
                
        except Exception as e:
            logger.error(f"Ошибка анализа с помощью SQL агента: {e}")
            return {
                "question": user_question,
                "sql_query": None,
                "data": pd.DataFrame(),
                "success": False,
                "error": f"Ошибка SQL агента: {str(e)}",
                "method": "sql_toolkit_agent"
            }

    def _extract_sql_from_agent_result(self, result: Dict[str, Any]) -> Optional[str]:
        """
        Извлечение SQL запроса из результата работы агента.
        
        Args:
            result: Результат работы LangChain агента
            
        Returns:
            SQL запрос если найден, иначе None
        """
        try:
            # Проверяем intermediate_steps на наличие SQL запросов
            if 'intermediate_steps' in result:
                for step in result['intermediate_steps']:
                    if isinstance(step, tuple) and len(step) >= 2:
                        action, observation = step
                        if hasattr(action, 'tool') and 'sql_db_query' in str(action.tool):
                            if hasattr(action, 'tool_input'):
                                # Извлекаем SQL из tool_input
                                tool_input = action.tool_input
                                if isinstance(tool_input, dict) and 'query' in tool_input:
                                    return tool_input['query']
                                elif isinstance(tool_input, str):
                                    return tool_input
            
            # Пытаемся найти SQL в выводе агента
            if 'output' in result:
                output = result['output']
                # Простой поиск паттернов SELECT, INSERT, UPDATE, DELETE
                sql_patterns = [
                    r'(?i)(SELECT\s+.*?(?:FROM|;|$))',
                    r'(?i)(INSERT\s+.*?(?:VALUES|;|$))',
                    r'(?i)(UPDATE\s+.*?(?:SET|;|$))',
                    r'(?i)(DELETE\s+.*?(?:FROM|;|$))'
                ]
                
                for pattern in sql_patterns:
                    match = re.search(pattern, output, re.MULTILINE | re.DOTALL)
                    if match:
                        return match.group(1).strip()
            
            return None
            
        except Exception as e:
            logger.warning(f"Не удалось извлечь SQL из результата агента: {e}")
            return None
    
    def _extract_data_from_agent_output(self, agent_output: str, sql_query: Optional[str] = None) -> pd.DataFrame:
        """
        Извлечение данных из текстового вывода агента или выполнение SQL запроса.
        
        Args:
            agent_output: Текстовый вывод агента
            sql_query: SQL запрос если был извлечен
            
        Returns:
            DataFrame с данными
        """
        try:
            # Если есть SQL запрос, выполняем его напрямую
            if sql_query:
                try:
                    return self.execute_query_with_retry(sql_query)
                except Exception as e:
                    logger.warning(f"Не удалось выполнить извлеченный SQL запрос: {e}")
            
            # Пытаемся парсить табличные данные из текста агента
            # Ищем простые табличные форматы
            lines = agent_output.split('\n')
            data_rows = []
            headers = None
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Ищем разделители таблиц (|, \t, множественные пробелы)
                if '|' in line and line.count('|') >= 2:
                    # Парсим как таблицу с разделителем |
                    cells = [cell.strip() for cell in line.split('|') if cell.strip()]
                    if headers is None:
                        headers = cells
                    else:
                        data_rows.append(cells)
                elif '\t' in line:
                    # Парсим как таблицу с табуляцией
                    cells = [cell.strip() for cell in line.split('\t') if cell.strip()]
                    if headers is None:
                        headers = cells
                    else:
                        data_rows.append(cells)
            
            # Создаем DataFrame если нашли данные
            if headers and data_rows:
                # Убеждаемся что все строки имеют одинаковую длину
                max_cols = len(headers)
                normalized_rows = []
                for row in data_rows:
                    if len(row) == max_cols:
                        normalized_rows.append(row)
                
                if normalized_rows:
                    return pd.DataFrame(normalized_rows, columns=headers)
            
            # Если не удалось извлечь табличные данные, возвращаем пустой DataFrame
            logger.info("Не удалось извлечь табличные данные из вывода агента")
            return pd.DataFrame()
            
        except Exception as e:
            logger.warning(f"Ошибка извлечения данных из вывода агента: {e}")
            return pd.DataFrame()

    def analyze_question_hybrid(self, user_question: str) -> Dict[str, Any]:
        """
        Полный анализ вопроса пользователя с использованием SQL агента и fallback на кастомную логику.
        
        Args:
            user_question: Вопрос пользователя на естественном языке
            
        Returns:
            Словарь с результатами анализа
        """
        try:
            logger.info(f"Гибридный анализ вопроса: {user_question}")
            
            # Сначала пытаемся использовать SQLDatabaseToolkit агента
            try:
                logger.info("Попытка анализа через SQLDatabaseToolkit агента...")
                result = self.analyze_question_with_agent(user_question)
                
                # Проверяем качество результата
                if (result.get("success", False) and 
                    result.get("data") is not None and 
                    not result["data"].empty):
                    
                    logger.info("SQLDatabaseToolkit агент успешно выполнил анализ")
                    result["method"] = "sql_toolkit_agent"
                    return result
                else:
                    logger.warning("SQLDatabaseToolkit агент не дал качественного результата")
                    
            except Exception as e:
                logger.warning(f"SQLDatabaseToolkit агент завершился с ошибкой: {e}")
            
            # Fallback на кастомную логику с тем же промптом
            logger.info("Переключение на кастомную логику...")
            result = self.analyze_question(user_question)
            
            if result.get("success", False):
                result["method"] = "custom_logic_fallback"
                logger.info("Кастомная логика успешно выполнила анализ")
            else:
                result["method"] = "failed_all_methods"
                logger.error("Все методы анализа завершились неудачей")
            
            return result
            
        except Exception as e:
            logger.error(f"Критическая ошибка в гибридном анализе: {e}")
            return {
                "question": user_question,
                "sql_query": None,
                "data": pd.DataFrame(),
                "success": False,
                "error": f"Критическая ошибка гибридного анализа: {str(e)}",
                "method": "critical_error"
            }


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


def reset_sql_agent() -> None:
    """
    Сброс синглтона SQL-агента для перезагрузки конфигурации.
    """
    global sql_agent
    if sql_agent is not None and hasattr(sql_agent, '__del__'):
        sql_agent.__del__()
    sql_agent = None
    logger.info("SQL-агент сброшен, будет создан заново при следующем обращении") 