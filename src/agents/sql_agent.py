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
            
            # Инициализация LLM для кастомной логики (с отключенным streaming)
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
            
            # Инициализация ОТДЕЛЬНОГО LLM для SQLDatabaseToolkit с disable_streaming=True
            agent_llm_kwargs = {
                "model": OPENAI_MODEL,
                "temperature": OPENAI_TEMPERATURE,
                "openai_api_key": OPENAI_API_KEY,
                "disable_streaming": True,  # КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ для SQLDatabaseToolkit
                "verbose": False
            }
            
            if OPENAI_BASE_URL:
                agent_llm_kwargs["base_url"] = OPENAI_BASE_URL
            
            agent_llm = ChatOpenAI(**agent_llm_kwargs)
            
            # Создаем временную копию базы данных для LangChain
            temp_db = tempfile.mktemp(suffix='.db')
            shutil.copy2(self.db_manager.database_path, temp_db)
            
            # Создаем подключение для LangChain с указанием всех доступных таблиц
            db_uri = f"duckdb:///{temp_db}"
            
            # Получаем все доступные таблицы из нашего database manager
            available_tables = list(self.db_manager.get_table_info().keys())
            logger.info(f"Доступные таблицы для SQLDatabaseToolkit: {available_tables}")
            
            # ВАЖНО: Получаем таблицы из ВСЕХ схем, не только main
            # Сначала проверяем какие схемы доступны
            try:
                # Получаем таблицы из схемы main
                temp_conn = duckdb.connect(temp_db)
                temp_conn.execute('USE main')
                main_tables = temp_conn.execute('SHOW TABLES').fetchdf()['name'].tolist()
                
                # Получаем таблицы из схемы sber_index  
                temp_conn.execute('USE sber_index')
                sber_tables = temp_conn.execute('SHOW TABLES').fetchdf()['name'].tolist()
                temp_conn.close()
                
                # Объединяем уникальные таблицы из обеих схем
                all_available_tables = list(set(main_tables + sber_tables))
                logger.info(f"Таблицы из main схемы: {main_tables}")
                logger.info(f"Таблицы из sber_index схемы: {sber_tables}")
                logger.info(f"Все уникальные таблицы для SQLDatabaseToolkit: {all_available_tables}")
                
            except Exception as e:
                logger.warning(f"Не удалось получить таблицы из всех схем: {e}")
                # Fallback к существующему подходу
                all_available_tables = available_tables
            
            # Создаем SQLDatabase с указанием всех таблиц из всех схем
            # Для доступа к таблицам из sber_index нужно указать schema='sber_index'
            # или использовать полные имена table_name в SQL
            self.sql_db = SQLDatabase.from_uri(
                db_uri,
                include_tables=None,                      # Пусть LangChain сам определит доступные таблицы
                sample_rows_in_table_info=3,              # Показываем примеры строк для LLM
                schema='sber_index',                      # Используем расширенную схему sber_index как основную
                view_support=True,                        # Поддержка представлений если есть
                max_string_length=300                     # Ограничиваем длину строк для экономии токенов
            )
            self.temp_db_path = temp_db
            
            # Создание toolkit с LLM без streaming
            toolkit = SQLDatabaseToolkit(db=self.sql_db, llm=agent_llm)
            
            # Кастомный промпт для SQL агента БЕЗ ограничения на 10 результатов
            custom_prefix = """You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct duckdb query to run, then look at the results of the query and return the answer.
You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for the relevant columns given the question.
You have access to tools for interacting with the database.
Only use the below tools. Only use the information returned by the below tools to construct your final answer.
You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

If the question does not seem related to the database, just return "I don't know" as the answer.

ВАЖНО: Если пользователь не указывает конкретное количество результатов, показывай все доступные данные без ограничений.

ОБРАБОТКА ОШИБОК:
- Если получаешь ошибку SQL синтаксиса, исправь запрос и попробуй снова
- Если колонка не найдена, проверь схему таблицы и используй правильное имя
- Если таблица не найдена, используй sql_db_list_tables для проверки доступных таблиц
- Максимум 3 попытки исправления для каждого запроса

"""

            custom_suffix = """Begin!

Question: {input}
Thought: I should look at the tables in the database to see what I can query. Then I should query the schema of the most relevant tables.
{agent_scratchpad}"""
            
            # Создание агента с кастомным промптом и улучшенными настройками
            self.agent = create_sql_agent(
                llm=agent_llm,  # Используем LLM без streaming
                toolkit=toolkit,
                agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True,
                prefix=custom_prefix,  # Кастомный промпт с инструкциями по исправлению ошибок
                suffix=custom_suffix,
                max_iterations=10,  # Увеличиваем количество итераций для самоисправления
                max_execution_time=120,  # Увеличиваем время для обработки с retry
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
    
    def generate_sql_query(self, user_question: str, attempt: int = 1, previous_error: Optional[str] = None, previous_sql: Optional[str] = None, previous_empty_result: bool = False) -> str:
        """
        Генерация SQL-запроса на основе вопроса пользователя с retry логикой.
        
        Args:
            user_question: Вопрос пользователя на естественном языке
            attempt: Номер попытки генерации (для логирования)
            previous_error: Ошибка предыдущей попытки для исправления
            previous_sql: SQL запрос предыдущей попытки для анализа
            previous_empty_result: Был ли предыдущий запрос успешным но пустым
            
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
            
            # Добавляем информацию об ошибке предыдущей попытки для исправления
            if attempt > 1 and previous_error and previous_sql:
                base_prompt += f"""
                
                ИСПРАВЛЕНИЕ ОШИБКИ:
                Предыдущий SQL-запрос: {previous_sql}
                Ошибка выполнения: {previous_error}
                
                Проанализируй ошибку и исправь SQL-запрос:
                - Если ошибка "table not found" или "column not found", используй правильные имена из схемы выше
                - Если ошибка синтаксиса, исправь синтаксические проблемы
                - Если ошибка типов данных, приведи к правильным типам
                - Убедись что используешь только существующие таблицы и колонки ИЗ СХЕМЫ ВЫШЕ
                """
            elif attempt > 1 and previous_empty_result and previous_sql:
                base_prompt += f"""
                
                РАСШИРЕНИЕ ПОИСКА (предыдущий запрос был успешен, но не вернул данных):
                Предыдущий SQL-запрос: {previous_sql}
                Результат: пустой (0 строк)
                
                Переформулируй запрос для получения данных:
                - Убери или ослабь слишком строгие условия WHERE
                - Проверь правильность значений в условиях (возможно нужны частичные совпадения через LIKE)
                - Рассмотри использование UNION для объединения результатов из разных таблиц
                - Попробуй более широкие диапазоны для числовых и временных условий
                - Используй LEFT JOIN вместо INNER JOIN если нужно показать все записи
                - Рассмотри группировку или агрегацию данных если ищешь сводную информацию
                - Проверь, возможно нужно искать в других таблицах или связанных данных
                
                Цель: найти релевантные данные, даже если они не точно соответствуют исходному запросу.
                """
            elif attempt > 1:
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
            if previous_error:
                logger.info(f"Учтена ошибка предыдущей попытки: {previous_error[:100]}...")
            elif previous_empty_result:
                logger.info(f"Учтен пустой результат предыдущей попытки, расширяем поиск")
            
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
        last_was_empty = False
        
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                logger.info(f"Анализ вопроса, попытка {attempt}/{MAX_RETRIES}")
                
                # Генерируем SQL-запрос с учетом ошибки или пустого результата предыдущей попытки
                sql_query = self.generate_sql_query(
                    user_question, 
                    attempt, 
                    previous_error=str(last_error) if last_error else None,
                    previous_sql=last_sql_query,
                    previous_empty_result=last_was_empty
                )
                last_sql_query = sql_query
                
                # Выполняем запрос
                data = self.execute_query_with_retry(sql_query)
                
                # Проверяем, не пустой ли результат
                if data.empty:
                    logger.warning(f"Запрос выполнен успешно, но не вернул данных (попытка {attempt})")
                    last_was_empty = True
                    last_error = None  # Сбрасываем ошибку, так как запрос технически успешен
                    
                    # Если это последняя попытка, возвращаем результат с пустыми данными
                    if attempt == MAX_RETRIES:
                        logger.warning(f"Все {MAX_RETRIES} попытки не дали данных для вопроса: {user_question}")
                        return {
                            "question": user_question,
                            "sql_query": sql_query,
                            "data": {},
                            "success": False,
                            "error": f"Запрос выполнен корректно, но не найдено данных после {MAX_RETRIES} попыток расширения поиска",
                            "attempts": attempt,
                            "empty_result": True
                        }
                    
                    # Продолжаем со следующей попыткой для расширения поиска
                    logger.info(f"Попытка {attempt + 1} с расширенным поиском...")
                    continue
                
                # Если данные найдены, формируем успешный результат
                # Преобразуем DataFrame в dict с pandas.Series
                data_dict = {}
                for column in data.columns:
                    data_dict[column] = data[column]  # pandas.Series объект
                
                result = {
                    "question": user_question,
                    "sql_query": sql_query,
                    "data": data_dict,  # dict с pandas.Series
                    "success": True,
                    "error": None,
                    "attempts": attempt,
                    "empty_result": False
                }
                
                logger.info(f"Анализ вопроса завершен успешно с {attempt} попытки, найдено {len(data)} строк")
                return result
                
            except Exception as e:
                last_error = e
                last_was_empty = False  # Сбрасываем флаг пустого результата при ошибке
                error_msg = str(e)
                
                logger.warning(f"Попытка {attempt}/{MAX_RETRIES} неуспешна: {error_msg}")
                
                # Если это последняя попытка, возвращаем ошибку
                if attempt == MAX_RETRIES:
                    logger.error(f"Все {MAX_RETRIES} попытки исчерпаны для вопроса: {user_question}")
                    break
                
                # Проверяем тип ошибки для определения стратегии retry
                if "syntax error" in error_msg.lower() or "parser error" in error_msg.lower():
                    logger.info(f"Обнаружена ошибка синтаксиса, перегенерируем запрос с учетом ошибки...")
                    continue
                elif "невалидный" in error_msg.lower() or "invalid" in error_msg.lower():
                    logger.info(f"Обнаружена ошибка валидации, перегенерируем запрос с учетом ошибки...")
                    continue
                elif "table" in error_msg.lower() and "not found" in error_msg.lower():
                    logger.info(f"Обнаружена ошибка 'table not found', перегенерируем с учетом схемы...")
                    continue
                elif "column" in error_msg.lower() and "not found" in error_msg.lower():
                    logger.info(f"Обнаружена ошибка 'column not found', перегенерируем с учетом схемы...")
                    continue
                else:
                    # Для других ошибок тоже пробуем еще раз с информацией об ошибке
                    logger.info(f"Обнаружена ошибка: {error_msg}, перегенерируем запрос с учетом ошибки...")
                    continue
        
        # Если все попытки неуспешны
        logger.error(f"Не удалось проанализировать вопрос после {MAX_RETRIES} попыток")
        return {
            "question": user_question,
            "sql_query": last_sql_query,
            "data": {},  # Пустой dict вместо пустого DataFrame
            "success": False,
            "error": f"Не удалось сгенерировать корректный SQL после {MAX_RETRIES} попыток. Последняя ошибка: {str(last_error)}",
            "attempts": MAX_RETRIES,
            "empty_result": last_was_empty
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
            ВАЖНО: Если получаешь ошибку SQL, исправь запрос и попробуй снова.
            """
            
            # Выполняем запрос через агента
            result = self.agent.invoke({"input": agent_prompt})
            
            logger.debug(f"Полный результат агента: {result}")
            
            # Проверяем структуру результата
            if not isinstance(result, dict):
                raise ValueError(f"Агент вернул неожиданный тип результата: {type(result)}")
            
            # Извлекаем SQL запрос из результата агента
            sql_query = self._extract_sql_from_agent_result(result)
            logger.info(f"Извлеченный SQL запрос: {sql_query}")
            
            # Пытаемся извлечь данные из intermediate_steps
            data = pd.DataFrame()
            
            # Приоритет 1: Извлекаем данные напрямую из observation в intermediate_steps
            if 'intermediate_steps' in result:
                for step in result['intermediate_steps']:
                    if isinstance(step, tuple) and len(step) >= 2:
                        action, observation = step
                        # Ищем результаты sql_db_query с данными
                        if (hasattr(action, 'tool') and 'sql_db_query' in str(action.tool) and 
                            observation and isinstance(observation, str)):
                            
                            logger.info("Найден результат SQL запроса в intermediate_steps")
                            # Пытаемся распарсить observation как табличные данные
                            parsed_data = self._parse_sql_observation(observation)
                            if not parsed_data.empty:
                                data = parsed_data
                                logger.info(f"Успешно извлечены данные из observation: {len(data)} строк, {len(data.columns)} колонок")
                                break
            
            # Приоритет 2: Если SQL запрос найден, выполняем его напрямую
            if data.empty and sql_query:
                try:
                    logger.info(f"Выполняю извлеченный SQL запрос для получения данных: {sql_query[:100]}...")
                    data = self.execute_query_with_retry(sql_query)
                    logger.info(f"Данные получены через прямое выполнение SQL: {len(data)} строк")
                except Exception as e:
                    logger.warning(f"Не удалось выполнить извлеченный SQL запрос: {e}")
            
            # Приоритет 3: Пытаемся извлечь из агентского вывода
            if data.empty and 'output' in result:
                agent_output = result['output']
                data = self._extract_data_from_agent_output(agent_output, sql_query)
                if not data.empty:
                    logger.info(f"Данные извлечены из текстового вывода агента: {len(data)} строк")
            
            # Валидация результата
            if data.empty and not sql_query:
                raise ValueError("Агент не вернул ни данных, ни SQL запроса")
            
            if data.empty and sql_query:
                logger.warning("SQL запрос найден, но данные не получены - возможно пустой результат")
            
            # Преобразуем DataFrame в dict с pandas.Series
            data_dict = {}
            if not data.empty:
                for column in data.columns:
                    data_dict[column] = data[column]  # pandas.Series объект
            
            # Проверяем успешность выполнения
            success = bool(data_dict) or (sql_query and len(sql_query.strip()) > 0)
            
            result_dict = {
                "question": user_question,
                "sql_query": sql_query,
                "data": data_dict,  # dict с pandas.Series
                "success": success,
                "error": None if success else "Агент не вернул данные",
                "agent_output": result.get('output', ''),
                "method": "sql_toolkit_agent"
            }
            
            if success:
                logger.info(f"Анализ успешно завершен: найден SQL={bool(sql_query)}, данные={len(data_dict)} колонок")
            else:
                logger.warning(f"Анализ завершен с предупреждениями: SQL={bool(sql_query)}, данные={len(data_dict)} колонок")
            
            return result_dict
                
        except Exception as e:
            error_msg = f"Ошибка SQL агента: {str(e)}"
            logger.error(error_msg)
            logger.debug(f"Подробности ошибки: {e}", exc_info=True)
            
            return {
                "question": user_question,
                "sql_query": None,
                "data": {},  # Пустой dict
                "success": False,
                "error": error_msg,
                "method": "sql_toolkit_agent"
            }

    def _parse_sql_observation(self, observation: str) -> pd.DataFrame:
        """
        Парсинг результата SQL запроса из observation в intermediate_steps.
        
        Args:
            observation: Строка с результатом SQL запроса
            
        Returns:
            DataFrame с данными
        """
        try:
            logger.debug(f"Парсинг SQL observation: {observation[:200]}...")
            
            lines = observation.strip().split('\n')
            data_rows = []
            headers = None
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Ищем строки с разделителем | (стандартный вывод DuckDB)
                if '|' in line and line.count('|') >= 1:
                    # Удаляем крайние пайпы если есть  
                    line = line.strip('|').strip()
                    cells = [cell.strip() for cell in line.split('|')]
                    
                    # Пропускаем строки-разделители
                    if all(cell.replace('-', '').replace(' ', '') == '' for cell in cells):
                        continue
                    
                    # Первая строка - заголовки
                    if headers is None:
                        headers = cells
                        logger.debug(f"Найдены заголовки: {headers}")
                    else:
                        # Добавляем строку данных если количество колонок совпадает
                        if len(cells) == len(headers):
                            data_rows.append(cells)
                            logger.debug(f"Добавлена строка: {cells}")
            
            # Создаем DataFrame
            if headers and data_rows:
                df = pd.DataFrame(data_rows, columns=headers)
                
                # Преобразуем числовые колонки
                for col in df.columns:
                    try:
                        # Убираем запятые и пробелы из чисел
                        if df[col].dtype == 'object':
                            df[col] = df[col].astype(str).str.replace(',', '').str.replace(' ', '')
                            df[col] = pd.to_numeric(df[col], errors='ignore')
                    except:
                        pass
                
                logger.info(f"Успешно создан DataFrame из observation: {len(df)} строк")
                return df
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.warning(f"Ошибка парсинга SQL observation: {e}")
            return pd.DataFrame()

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
                        # Ищем действия sql_db_query
                        if hasattr(action, 'tool') and 'sql_db_query' in str(action.tool):
                            if hasattr(action, 'tool_input'):
                                # Извлекаем SQL из tool_input
                                tool_input = action.tool_input
                                if isinstance(tool_input, dict) and 'query' in tool_input:
                                    return tool_input['query']
                                elif isinstance(tool_input, str):
                                    return tool_input
                        # Также ищем в самом action если это AgentAction
                        elif hasattr(action, 'tool_input') and isinstance(action.tool_input, str):
                            # Проверяем, является ли tool_input SQL запросом
                            tool_input = action.tool_input.strip()
                            if any(keyword in tool_input.upper() for keyword in ['SELECT', 'INSERT', 'UPDATE', 'DELETE']):
                                return tool_input
            
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
            # Если есть SQL запрос, выполняем его напрямую для получения актуальных данных
            if sql_query:
                try:
                    logger.info(f"Выполняю извлеченный SQL запрос: {sql_query[:100]}...")
                    return self.execute_query_with_retry(sql_query)
                except Exception as e:
                    logger.warning(f"Не удалось выполнить извлеченный SQL запрос: {e}")
            
            # Улучшенный парсинг табличных данных из текста агента
            lines = agent_output.split('\n')
            data_rows = []
            headers = None
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Ищем строки с разделителем | (как в выводе DuckDB)
                if '|' in line and line.count('|') >= 1:
                    # Удаляем крайние пайпы если есть
                    line = line.strip('|').strip()
                    cells = [cell.strip() for cell in line.split('|')]
                    
                    # Пропускаем строки-разделители (только дефисы и пайпы)
                    if all(cell.replace('-', '').replace(' ', '') == '' for cell in cells):
                        continue
                    
                    # Первая строка с данными - это заголовки
                    if headers is None:
                        headers = cells
                        logger.info(f"Найдены заголовки таблицы: {headers}")
                    else:
                        # Проверяем что количество колонок совпадает
                        if len(cells) == len(headers):
                            data_rows.append(cells)
                            logger.debug(f"Добавлена строка данных: {cells}")
            
            # Создаем DataFrame если нашли данные
            if headers and data_rows:
                logger.info(f"Создаю DataFrame с {len(data_rows)} строками и колонками: {headers}")
                
                # Преобразуем типы данных
                df = pd.DataFrame(data_rows, columns=headers)
                
                # Пытаемся преобразовать числовые колонки
                for col in df.columns:
                    # Пробуем преобразовать в числа
                    try:
                        # Убираем запятые из больших чисел
                        df[col] = df[col].str.replace(',', '')
                        # Пробуем преобразовать в float
                        df[col] = pd.to_numeric(df[col], errors='ignore')
                    except:
                        pass  # Оставляем как строку если не получается
                
                logger.info(f"Успешно извлечен DataFrame: {len(df)} строк, {len(df.columns)} колонок")
                return df
            
            # Если не удалось извлечь табличные данные, возвращаем пустой DataFrame
            logger.info("Не удалось извлечь табличные данные из вывода агента")
            return pd.DataFrame()
            
        except Exception as e:
            logger.warning(f"Ошибка извлечения данных из вывода агента: {e}")
            return pd.DataFrame()

    def analyze_question_hybrid(self, user_question: str) -> Dict[str, Any]:
        """
        Анализ вопроса пользователя с использованием SQLDatabaseToolkit агента с механизмом самоисправления ошибок.
        
        Args:
            user_question: Вопрос пользователя на естественном языке
            
        Returns:
            Словарь с результатами анализа
        """
        max_retries = 3
        last_error = None
        attempt_history = []  # Сохраняем историю попыток для анализа
        
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"SQLDatabaseToolkit анализ вопроса (попытка {attempt}/{max_retries}): {user_question}")
                
                # Если есть предыдущие ошибки, модифицируем вопрос для учета ошибок
                modified_question = user_question
                if attempt > 1 and attempt_history:
                    error_context = ""
                    empty_result_context = ""
                    
                    for i, prev_attempt in enumerate(attempt_history, 1):
                        if prev_attempt.get("empty_result", False):
                            empty_result_context += f"\nПопытка {i}: запрос выполнен успешно, но вернул 0 строк"
                            if prev_attempt.get("sql_query"):
                                empty_result_context += f" (SQL: {prev_attempt['sql_query'][:100]}...)"
                        elif prev_attempt.get("error"):
                            error_context += f"\nПопытка {i}: {prev_attempt['error'][:200]}..."
                    
                    context_parts = []
                    if error_context:
                        context_parts.append(f"ОШИБКИ ВЫПОЛНЕНИЯ:{error_context}")
                    if empty_result_context:
                        context_parts.append(f"ПУСТЫЕ РЕЗУЛЬТАТЫ:{empty_result_context}")
                    
                    if context_parts:
                        combined_context = "\n\n".join(context_parts)
                        modified_question = f"""
                        Исходный вопрос: {user_question}
                        
                        УЧТИ ПРЕДЫДУЩИЕ ПОПЫТКИ:
                        {combined_context}
                        
                        Исправь подход с учетом этого опыта:
                        - Если были ошибки: исправь имена таблиц/колонок, синтаксис, типы данных
                        - Если были пустые результаты: ослабь условия, используй LIKE вместо =, расширь поиск через UNION, попробуй другие таблицы
                        - Рассмотри группировку или агрегацию если ищешь сводные данные
                        """
                
                # Используем только SQLDatabaseToolkit агента
                result = self.analyze_question_with_agent(modified_question)
                
                # Сохраняем результат попытки в историю
                attempt_history.append({
                    "attempt": attempt,
                    "success": result.get("success", False),
                    "error": result.get("error"),
                    "sql_query": result.get("sql_query"),
                    "data_count": len(result.get("data", {})),
                    "empty_result": len(result.get("data", {})) == 0 and result.get("success", False)
                })
                
                # Проверяем качество результата
                if (result.get("success", False) and 
                    result.get("data") is not None and 
                    bool(result["data"])):
                    
                    logger.info(f"SQLDatabaseToolkit агент успешно выполнил анализ с попытки {attempt}")
                    result["method"] = "sql_toolkit_agent"
                    result["attempts"] = attempt
                    result["attempt_history"] = attempt_history
                    return result
                elif (result.get("success", False) and 
                      result.get("data") is not None and 
                      len(result["data"]) == 0):
                    # Случай успешного выполнения, но с пустым результатом
                    logger.warning(f"SQLDatabaseToolkit агент (попытка {attempt}) выполнил запрос успешно, но не вернул данных")
                    last_error = "Запрос выполнен корректно, но не найдено данных - нужно расширить поиск"
                    
                    # Если это не последняя попытка, пробуем еще раз с расширенным поиском
                    if attempt < max_retries:
                        logger.info(f"Повторная попытка {attempt + 1} с расширенным поиском...")
                        continue
                else:
                    error_msg = result.get("error", "Агент не вернул качественные данные")
                    logger.warning(f"SQLDatabaseToolkit агент (попытка {attempt}) не дал качественного результата: {error_msg}")
                    last_error = error_msg
                    
                    # Если это не последняя попытка, пробуем еще раз с модифицированным промптом
                    if attempt < max_retries:
                        logger.info(f"Повторная попытка {attempt + 1} с учетом ошибки: {error_msg[:100]}...")
                        continue
                
            except Exception as e:
                error_msg = str(e)
                logger.warning(f"SQLDatabaseToolkit агент (попытка {attempt}) завершился с ошибкой: {error_msg}")
                last_error = error_msg
                
                # Сохраняем ошибку в историю
                attempt_history.append({
                    "attempt": attempt,
                    "success": False,
                    "error": error_msg,
                    "sql_query": None,
                    "data_count": 0,
                    "empty_result": False
                })
                
                # Если это не последняя попытка, пробуем еще раз
                if attempt < max_retries:
                    logger.info(f"Повторная попытка {attempt + 1} после ошибки: {error_msg[:100]}...")
                    continue
        
        # Если все попытки неудачны, возвращаем ошибку без fallback
        logger.error(f"SQLDatabaseToolkit агент не смог выполнить анализ после {max_retries} попыток")
        return {
            "question": user_question,
            "sql_query": None,
            "data": {},
            "success": False,
            "error": f"SQLDatabaseToolkit не смог выполнить анализ после {max_retries} попыток. Последняя ошибка: {last_error}",
            "method": "sql_toolkit_failed",
            "attempts": max_retries,
            "attempt_history": attempt_history
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