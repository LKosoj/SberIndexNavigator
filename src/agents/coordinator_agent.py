"""
Координирующий агент на базе LangGraph для интеллектуального управления задачами.
Планирует последовательность действий и координирует работу специализированных инструментов.
"""

import logging
from typing import Dict, Any, Optional, List, Literal, Annotated
import json
import pandas as pd
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.graph.message import AnyMessage
from langgraph.types import Command
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

from src.config.settings import (
    OPENAI_API_KEY,
    OPENAI_BASE_URL, 
    OPENAI_MODEL,
    OPENAI_TEMPERATURE
)
from src.data.database import get_database_manager
from src.agents.sql_agent import get_sql_agent
from src.agents.analysis_agent import get_analysis_agent
from src.agents.visualize_tool import get_visualization_analyzer
from src.agents.smart_analysis_agent import get_smart_analysis_agent
from src.memory.agent_memory import get_agent_memory, MemoryType

logger = logging.getLogger(__name__)


def _has_data(data) -> bool:
    """Безопасная проверка наличия данных."""
    if data is None:
        return False
    if isinstance(data, dict):
        return bool(data)
    elif isinstance(data, pd.DataFrame):
        return not data.empty
    elif isinstance(data, list):
        return bool(data)
    else:
        return data is not None


# === СОСТОЯНИЕ ГРАФА ===
class CoordinatorState(TypedDict):
    """Состояние координирующего агента."""
    messages: Annotated[List[AnyMessage], add_messages]
    user_question: str
    current_plan: List[str] 
    executed_steps: List[str]
    data: Optional[Dict[str, Any]]
    sql_query: Optional[str]
    analysis_result: Optional[Dict[str, Any]]
    visualization_config: Optional[Dict[str, Any]]
    final_response: Optional[str]
    error_message: Optional[str]
    session_id: Optional[str]
    memory_context: Optional[List[Dict[str, Any]]]
    smart_insights: Optional[List[Dict[str, Any]]]


# === ИНСТРУМЕНТЫ ===
class DatabaseTool(BaseTool):
    """Инструмент для работы с базой данных."""
    
    name: str = "database_tool"
    description: str = "Генерирует и выполняет SQL-запросы к базе данных индексов Сбербанка"
    
    def _run(self, question: str) -> Dict[str, Any]:
        """Выполнение SQL-запроса."""
        try:
            sql_agent = get_sql_agent()
            result = sql_agent.analyze_question(question)
            #result = sql_agent.analyze_question_hybrid(question)
            
            # SQL агент теперь возвращает dict с pandas.Series, передаем как есть
            data_dict = result.get("data", {})
            data_length = 0
            if data_dict:
                # Подсчитываем количество записей из любой pandas.Series
                for key, series in data_dict.items():
                    if isinstance(series, pd.Series):
                        data_length = len(series)
                        break
            
            return {
                "success": True,
                "sql_query": result.get("sql_query", ""),
                "data": data_dict,  # Передаем dict с pandas.Series как есть
                "message": f"Получено {data_length} строк данных"
            }
        except Exception as e:
            logger.error(f"Ошибка в DatabaseTool: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": f"Ошибка выполнения SQL-запроса: {e}"
            }


class AnalysisTool(BaseTool):
    """Инструмент для анализа данных."""
    
    name: str = "analysis_tool" 
    description: str = "Проводит интеллектуальный анализ данных и генерирует бизнес-инсайты"
    
    def _run(self, data_dict: Dict[str, Any], question: str) -> Dict[str, Any]:
        """Проведение анализа данных."""
        try:
            if not data_dict:
                return {
                    "success": False,
                    "error": "Нет данных для анализа",
                    "message": "Данные отсутствуют"
                }
            
            # Преобразуем dict с pandas.Series в DataFrame
            df_data = {}
            for key, values in data_dict.items():
                if isinstance(values, pd.Series):
                    df_data[key] = values
                elif isinstance(values, list):
                    df_data[key] = pd.Series(values)
                else:
                    df_data[key] = pd.Series([values])
            
            df = pd.DataFrame(df_data)
            
            analysis_agent = get_analysis_agent()
            result = analysis_agent.analyze_data(df, question)
            
            return {
                "success": True,
                "analysis": result,
                "message": f"Анализ выполнен, получено {len(result.get('recommendations', []))} рекомендаций"
            }
        except Exception as e:
            logger.error(f"Ошибка в AnalysisTool: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": f"Ошибка анализа данных: {e}"
            }


class VisualizationTool(BaseTool):
    """Инструмент для создания визуализации."""
    
    name: str = "visualization_tool"
    description: str = "Определяет оптимальный тип визуализации для данных"
    
    def _run(self, data_dict: Dict[str, Any], question: str) -> Dict[str, Any]:
        """
        Создание визуализации для данных.
        
        Args:
            data_dict: Словарь с данными
            question: Вопрос пользователя
            
        Returns:
            Результат создания визуализации
        """
        try:
            # Преобразуем данные в DataFrame с поддержкой pandas.Series
            if isinstance(data_dict, dict) and 'records' in data_dict:
                df = pd.DataFrame(data_dict['records'])
            elif isinstance(data_dict, list):
                df = pd.DataFrame(data_dict)
            elif isinstance(data_dict, pd.DataFrame):
                df = data_dict
            elif isinstance(data_dict, dict):
                # Обрабатываем dict с pandas.Series
                df_data = {}
                for key, values in data_dict.items():
                    if isinstance(values, pd.Series):
                        df_data[key] = values
                    elif isinstance(values, list):
                        df_data[key] = pd.Series(values)
                    else:
                        df_data[key] = pd.Series([values])
                df = pd.DataFrame(df_data)
            else:
                df = pd.DataFrame()
            
            if df.empty:
                return {
                    "success": False,
                    "error": "Нет данных для визуализации"
                }
            
            # Получаем анализатор визуализации
            analyzer = get_visualization_analyzer()
            
            # Определяем тип графика
            chart_type = analyzer.determine_chart_type(df, question)
            
            # Получаем конфигурацию визуализации
            viz_config = analyzer.get_visualization_config(chart_type, df)
            
            # Проверяем что конфигурация корректна для графиков
            if chart_type in ["line", "bar", "scatter"]:
                if not viz_config.get("x_column") or not viz_config.get("y_column"):
                    # Автоматически определяем колонки если не найдены
                    numeric_cols = list(df.select_dtypes(include=['number']).columns)
                    text_cols = list(df.select_dtypes(include=['object']).columns)
                    
                    # Для временных данных
                    if 'month' in df.columns:
                        viz_config['x_column'] = 'month'
                    elif text_cols:
                        viz_config['x_column'] = text_cols[0]
                    
                    # Для значений исключаем служебные колонки
                    value_cols = [col for col in numeric_cols if not any(
                        keyword in col.lower() for keyword in ['year', 'id', 'code', '_id', 'год', 'record_count']
                    )]
                    if value_cols:
                        viz_config['y_column'] = value_cols[0]
                    elif numeric_cols:
                        viz_config['y_column'] = numeric_cols[0]
            
            logger.info(f"Создана конфигурация визуализации: {chart_type}")
            
            return {
                "success": True,
                "chart_type": chart_type,
                "config": viz_config,
                "message": f"Создана конфигурация для {chart_type} графика"
            }
            
        except Exception as e:
            logger.error(f"Ошибка создания визуализации: {e}")
            return {
                "success": False,
                "error": str(e),
                "chart_type": "table",
                "config": {}
            }


# === КООРДИНИРУЮЩИЙ АГЕНТ ===
class CoordinatorAgent:
    """Главный координирующий агент на базе LangGraph."""
    
    def __init__(self, session_id: str = None):
        """Инициализация координатора."""
        self.session_id = session_id
        self.llm = self._initialize_llm()
        self.tools = self._initialize_tools()
        self.graph = self._build_graph()
        
        # Инициализируем память агента
        self.memory = get_agent_memory(session_id=session_id)
        
        # Инициализируем SmartAnalysisAgent
        self.smart_agent = get_smart_analysis_agent()
        
        # Системный промпт для планирования
        self.system_prompt = """
Ты - интеллектуальный координатор анализа данных индексов Сбербанка.

ТВОЯ РОЛЬ:
- Планируешь последовательность действий для ответа на вопрос пользователя
- Координируешь работу специализированных инструментов
- Адаптируешь план на основе полученных результатов
- Генерируешь итоговый ответ с инсайтами

ДОСТУПНЫЕ ИНСТРУМЕНТЫ:
1. database_tool - для SQL-запросов к базе данных
2. analysis_tool - для анализа данных и генерации инсайтов
3. visualization_tool - для создания визуализации

ПРИНЦИПЫ ПЛАНИРОВАНИЯ:
- Всегда начинай с получения данных (database_tool)
- Проводи анализ полученных данных (analysis_tool)
- Создавай визуализацию для наглядности (visualization_tool)
- Адаптируй план если инструмент возвращает ошибку
- Предоставляй развернутый ответ с рекомендациями

ФОРМАТ ПЛАНИРОВАНИЯ:
1. Определи что нужно сделать
2. Составь план из 3-5 шагов
3. Выполняй шаги последовательно
4. Анализируй результаты каждого шага
5. Корректируй план при необходимости
"""

    def _initialize_llm(self) -> ChatOpenAI:
        """Инициализация LLM."""
        llm_kwargs = {
            "model": OPENAI_MODEL,
            "temperature": OPENAI_TEMPERATURE,
            "openai_api_key": OPENAI_API_KEY,
            "streaming": False
        }
        
        if OPENAI_BASE_URL:
            llm_kwargs["base_url"] = OPENAI_BASE_URL
        
        return ChatOpenAI(**llm_kwargs)

    def _initialize_tools(self) -> List[BaseTool]:
        """Инициализация инструментов."""
        return [
            DatabaseTool(),
            AnalysisTool(),
            VisualizationTool()
        ]

    def _build_graph(self) -> StateGraph:
        """Построение графа координатора."""
        # Создаем граф
        builder = StateGraph(CoordinatorState)
        
        # Добавляем узлы
        builder.add_node("planner", self._planner_node)
        builder.add_node("executor", self._executor_node)
        builder.add_node("reviewer", self._reviewer_node)
        
        # Убираем ToolNode - будем вызывать инструменты напрямую в executor
        
        # Добавляем связи
        builder.add_edge(START, "planner")
        builder.add_edge("planner", "executor")
        builder.add_conditional_edges(
            "executor",
            self._should_continue_execution,
            {
                "continue": "executor",  # Продолжаем выполнение плана
                "review": "reviewer"      # Переходим к финальному обзору
            }
        )
        builder.add_edge("reviewer", END)
        
        return builder.compile()

    def _planner_node(self, state: CoordinatorState) -> CoordinatorState:
        """Узел планирования задач."""
        try:
            user_question = state["user_question"]
            
            # Создаем план выполнения
            planning_prompt = f"""
{self.system_prompt}

ВОПРОС ПОЛЬЗОВАТЕЛЯ: {user_question}

Создай детальный план выполнения этого запроса. План должен содержать 3-5 шагов.

Ответь в формате JSON:
{{
    "plan": ["шаг1", "шаг2", "шаг3", "шаг4"],
    "reasoning": "объяснение логики планирования"
}}
"""
            
            response = self.llm.invoke([
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=planning_prompt)
            ])
            
            try:
                plan_data = json.loads(response.content)
                plan = plan_data.get("plan", [])
            except:
                # Fallback план
                plan = [
                    "Получить данные из базы",
                    "Проанализировать данные", 
                    "Создать визуализацию",
                    "Сформировать ответ"
                ]
            
            logger.info(f"Создан план: {plan}")
            
            return {
                **state,
                "current_plan": plan,
                "executed_steps": [],
                "messages": state["messages"] + [
                    AIMessage(content=f"📋 **План выполнения:**\n" + "\n".join([f"{i+1}. {step}" for i, step in enumerate(plan)]))
                ]
            }
            
        except Exception as e:
            logger.error(f"Ошибка в planner_node: {e}")
            return {
                **state,
                "error_message": f"Ошибка планирования: {e}"
            }

    def _executor_node(self, state: CoordinatorState) -> CoordinatorState:
        """Узел выполнения плана с прямыми вызовами инструментов."""
        try:
            plan = state.get("current_plan", [])
            executed = state.get("executed_steps", [])
            
            # Если план выполнен полностью, завершаем
            if len(executed) >= len(plan):
                return state
            
            # Определяем следующий шаг для выполнения
            next_step = plan[len(executed)]
            logger.info(f"Выполняю шаг: {next_step}")
            
            # Выполняем шаг в зависимости от его типа
            step_lower = next_step.lower()
            
            # Шаг 1: Получение данных из базы  
            if any(keyword in step_lower for keyword in ["данные", "sql", "базы", "получ", "запрос"]) and not any(keyword in step_lower for keyword in ["анализ", "визуализ", "график", "диаграмм"]):
                logger.info("Выполняю запрос к базе данных...")
                db_tool = DatabaseTool()
                result = db_tool._run(state["user_question"])
                
                if result.get("success"):
                    executed.append(next_step)
                    return {
                        **state,
                        "executed_steps": executed,
                        "data": result.get("data", {}),
                        "sql_query": result.get("sql_query", ""),
                        "messages": state["messages"] + [
                            AIMessage(content=f"✅ {next_step}: {result.get('message', 'Данные получены')}")
                        ]
                    }
                else:
                    return {
                        **state,
                        "error_message": f"Ошибка получения данных: {result.get('error', 'Неизвестная ошибка')}"
                    }
            
            # Шаг 2: Анализ данных
            elif any(keyword in step_lower for keyword in ["анализ", "анализир", "обработ"]):
                logger.info("Выполняю анализ данных...")
                data = state.get("data", {})
                if _has_data(data):
                    analysis_tool = AnalysisTool()
                    result = analysis_tool._run(data, state["user_question"])
                    
                    if result.get("success"):
                        executed.append(next_step)
                        return {
                            **state,
                            "executed_steps": executed,
                            "analysis_result": result.get("analysis", {}),
                            "messages": state["messages"] + [
                                AIMessage(content=f"✅ {next_step}: {result.get('message', 'Анализ выполнен')}")
                            ]
                        }
                    else:
                        # Анализ не критичен - продолжаем без него
                        executed.append(next_step)
                        return {
                            **state,
                            "executed_steps": executed,
                            "messages": state["messages"] + [
                                AIMessage(content=f"⚠️ {next_step}: Анализ пропущен - {result.get('error', 'ошибка')}")
                            ]
                        }
                else:
                    # Нет данных для анализа - пропускаем шаг
                    executed.append(next_step)
                    return {
                        **state,
                        "executed_steps": executed,
                        "messages": state["messages"] + [
                            AIMessage(content=f"⚠️ {next_step}: Пропущен - нет данных для анализа")
                        ]
                    }
            
            # Шаг 3: Создание визуализации
            elif any(keyword in step_lower for keyword in ["визуализ", "график", "диаграмм"]):
                logger.info("Создаю визуализацию...")
                data = state.get("data", {})
                logger.info(f"Данные для визуализации: type={type(data)}, keys={list(data.keys()) if isinstance(data, dict) else 'не dict'}, len={len(data) if hasattr(data, '__len__') else 'нет len'}")
                if _has_data(data):
                    # Преобразуем данные в правильный формат для VisualizationTool
                    # SQL агент возвращает dict с колонками как pandas.Series, нужно преобразовать в {'records': [...]}
                    if isinstance(data, dict) and 'records' not in data:
                        # Конвертируем формат pandas.Series в records
                        records = []
                        if _has_data(data):
                            # Конвертируем все pandas.Series в списки для унификации
                            series_data = {}
                            max_length = 0
                            
                            for key, values in data.items():
                                if isinstance(values, pd.Series):
                                    # Конвертируем pandas.Series в список
                                    series_data[key] = values.tolist()
                                    max_length = max(max_length, len(values))
                                elif isinstance(values, list):
                                    series_data[key] = values
                                    max_length = max(max_length, len(values))
                                else:
                                    # Скалярное значение - преобразуем в список
                                    series_data[key] = [values]
                                    max_length = max(max_length, 1)
                            
                            logger.info(f"Обработанные данные: {len(series_data)} колонок, максимальная длина: {max_length}")
                            
                            # Создаем записи для каждого индекса
                            for i in range(max_length):
                                record = {}
                                for key, values in series_data.items():
                                    if i < len(values):
                                        record[key] = values[i]
                                    else:
                                        # Если значений меньше чем индексов, берем последнее значение
                                        record[key] = values[-1] if values else None
                                records.append(record)
                        
                        viz_data = {'records': records}
                        logger.info(f"Преобразованные данные: {len(records)} записей из {max_length} исходных pandas.Series")
                    else:
                        viz_data = data
                    
                    viz_tool = VisualizationTool()
                    result = viz_tool._run(viz_data, state["user_question"])
                    logger.info(f"Результат визуализации: success={result.get('success')}, keys={list(result.keys())}, error={result.get('error')}")
                    
                    if result.get("success"):
                        executed.append(next_step)
                        # VisualizationTool уже правильно создал конфигурацию с DataFrame и колонками
                        viz_config = result.get("config", {})
                        logger.info(f"Конфигурация визуализации: тип={viz_config.get('type')}, x_column={viz_config.get('x_column')}, y_column={viz_config.get('y_column')}, data_shape={viz_config.get('data').shape if viz_config.get('data') is not None else 'None'}")
                        
                        return {
                            **state,
                            "executed_steps": executed,
                            "visualization_config": viz_config,
                            "messages": state["messages"] + [
                                AIMessage(content=f"✅ {next_step}: {result.get('message', 'Визуализация создана')}")
                            ]
                        }
                    else:
                        # Визуализация не критична - продолжаем без неё
                        executed.append(next_step)
                        return {
                            **state,
                            "executed_steps": executed,
                            "messages": state["messages"] + [
                                AIMessage(content=f"⚠️ {next_step}: Визуализация пропущена - {result.get('error', 'ошибка')}")
                            ]
                        }
                else:
                    # Нет данных для визуализации - пропускаем шаг
                    executed.append(next_step)
                    return {
                        **state,
                        "executed_steps": executed,
                        "messages": state["messages"] + [
                            AIMessage(content=f"⚠️ {next_step}: Пропущен - нет данных для визуализации")
                        ]
                    }
            
            # Остальные шаги (планирование, обзор и т.д.) - просто отмечаем как выполненные
            else:
                executed.append(next_step)
                return {
                    **state,
                    "executed_steps": executed,
                    "messages": state["messages"] + [
                        AIMessage(content=f"✅ {next_step}: Выполнен")
                    ]
                }
            
        except Exception as e:
            logger.error(f"Ошибка в executor_node: {e}")
            return {
                **state,
                "error_message": f"Ошибка выполнения: {e}"
            }

    def _reviewer_node(self, state: CoordinatorState) -> CoordinatorState:
        """Узел обзора и формирования итогового ответа."""
        try:
            # Собираем всю информацию
            user_question = state["user_question"]
            data = state.get("data", {})
            analysis = state.get("analysis_result", {})
            viz_config = state.get("visualization_config", {})
            
            # Формируем итоговый ответ
            review_prompt = f"""
На основе выполненного анализа сформируй итоговый ответ на вопрос пользователя.

ВОПРОС: {user_question}

ПОЛУЧЕННЫЕ ДАННЫЕ: {"Есть данные" if _has_data(data) else "Нет данных"}
РЕЗУЛЬТАТЫ АНАЛИЗА: {"Есть анализ" if analysis else "Нет анализа"} 
ВИЗУАЛИЗАЦИЯ: {"Настроена" if viz_config else "Не настроена"}

Создай структурированный ответ, который включает:
1. Прямой ответ на вопрос
2. Ключевые инсайты из данных
3. Практические рекомендации
4. Краткое резюме

Ответ должен быть понятным и полезным для пользователя.
"""
            
            response = self.llm.invoke([
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=review_prompt)
            ])
            
            final_answer = response.content
            
            return {
                **state,
                "final_response": final_answer,
                "messages": state["messages"] + [
                    AIMessage(content=f"✅ **Итоговый ответ:**\n\n{final_answer}")
                ]
            }
            
        except Exception as e:
            logger.error(f"Ошибка в reviewer_node: {e}")
            return {
                **state,
                "error_message": f"Ошибка формирования ответа: {e}",
                "final_response": "Извините, произошла ошибка при формировании ответа."
            }

    def _should_continue_execution(self, state: CoordinatorState) -> Literal["continue", "review"]:
        """Определяет нужно ли продолжать выполнение плана."""
        plan = state.get("current_plan", [])
        executed = state.get("executed_steps", [])
        
        # Если есть ошибка, переходим к обзору
        if state.get("error_message"):
            return "review"
        
        # Если план выполнен полностью, идем к reviewer
        if len(executed) >= len(plan):
            return "review"
        
        # Иначе продолжаем выполнение
        return "continue"

    def process_question(self, question: str) -> Dict[str, Any]:
        """
        Обработка вопроса пользователя.
        
        Args:
            question: Вопрос пользователя
            
        Returns:
            Результат обработки
        """
        try:
            # Поиск контекста в памяти
            memory_context = []
            if self.memory:
                memory_results = self.memory.search_memory(question, limit=3)
                memory_context = [
                    {
                        "type": record.memory_type.value,
                        "content": record.content,
                        "timestamp": record.timestamp.isoformat(),
                        "relevance": record.relevance_score
                    }
                    for record in memory_results
                ]
                logger.info(f"Найдено {len(memory_context)} релевантных записей в памяти")
            
            # Начальное состояние с контекстом памяти
            initial_state = CoordinatorState(
                messages=[HumanMessage(content=question)],
                user_question=question,
                current_plan=[],
                executed_steps=[],
                data=None,
                sql_query=None,
                analysis_result=None,
                visualization_config=None,
                final_response=None,
                error_message=None,
                session_id=self.session_id,
                memory_context=memory_context,
                smart_insights=[]
            )
            
            # Выполняем граф с ограничением рекурсии
            result = self.graph.invoke(initial_state, config={"recursion_limit": 10})
            
            # Результаты инструментов уже находятся в состоянии, не нужно извлекать из сообщений
            # Данные уже сохранены в состоянии через executor_node
            pass
            
            # Правильно извлекаем данные из состояния с учетом pandas.Series
            final_data = result.get("data", {})
            if _has_data(final_data) and not isinstance(final_data, pd.DataFrame):
                if isinstance(final_data, dict) and 'records' in final_data:
                    final_data = pd.DataFrame(final_data['records'])
                elif isinstance(final_data, list):
                    final_data = pd.DataFrame(final_data)
                elif isinstance(final_data, dict):
                    # Обрабатываем dict с pandas.Series
                    df_data = {}
                    for key, values in final_data.items():
                        if isinstance(values, pd.Series):
                            df_data[key] = values
                        elif isinstance(values, list):
                            df_data[key] = pd.Series(values)
                        else:
                            df_data[key] = pd.Series([values])
                    final_data = pd.DataFrame(df_data)
                else:
                    final_data = pd.DataFrame(final_data)
            elif not _has_data(final_data):
                final_data = pd.DataFrame()
                
            # Сохраняем результаты в память
            if self.memory and result.get("final_response"):
                # Сохраняем вопрос-ответ
                self.memory.add_memory(
                    content={
                        "question": question,
                        "response": result.get("final_response", ""),
                        "has_data": _has_data(final_data),
                        "data_summary": f"{len(final_data)} записей" if _has_data(final_data) else "Нет данных"
                    },
                    memory_type=MemoryType.CONTEXT,
                    metadata={"interaction_type": "question_answer"},
                    tags=["question", "context"],
                    relevance_score=0.8
                )
                
                # Сохраняем инсайты из анализа если есть
                analysis_data = result.get("analysis_result", {})
                if analysis_data and analysis_data.get("insights"):
                    for insight in analysis_data["insights"][:3]:  # Первые 3 инсайта
                        self.memory.save_insight(
                            insight=insight,
                            confidence=0.7,
                            tags=["analysis", "insight"]
                        )
                
                logger.info("Результаты сохранены в память")
            
            return {
                "success": True,
                "response": result.get("final_response", "Ответ получен"),
                "data": final_data,
                "sql_query": result.get("sql_query", ""),
                "analysis": result.get("analysis_result", {}),
                "visualization": result.get("visualization_config", {}),
                "messages": result.get("messages", []),
                "memory_context": memory_context
            }
            
        except Exception as e:
            logger.error(f"Ошибка обработки вопроса: {e}")
            return {
                "success": False,
                "error": str(e),
                "response": f"Извините, произошла ошибка: {e}",
                "data": pd.DataFrame(),
                "sql_query": "",
                "analysis": {},
                "visualization": {},
                "messages": []
            }


# === ФУНКЦИИ ДОСТУПА ===
_coordinator_agent = None

def get_coordinator_agent(session_id: str = None) -> CoordinatorAgent:
    """Получение экземпляра координирующего агента."""
    global _coordinator_agent
    if _coordinator_agent is None or (session_id and _coordinator_agent.session_id != session_id):
        _coordinator_agent = CoordinatorAgent(session_id=session_id)
    return _coordinator_agent

def reset_coordinator_agent() -> None:
    """Сброс экземпляра агента."""
    global _coordinator_agent
    _coordinator_agent = None 