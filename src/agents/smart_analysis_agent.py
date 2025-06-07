"""
SmartAnalysisAgent - Интеллектуальный агент анализа с возможностями самооценки и многоступенчатого рассуждения.

Этот агент предназначен для:
- Multi-step reasoning (многоступенчатые рассуждения)
- Self-reflection механизм (самооценка результатов)
- Contextual analysis (контекстуальный анализ)  
- Proactive insights (проактивные инсайты)
"""

import logging
import json
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import pandas as pd
import numpy as np
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage

from src.config.settings import (
    OPENAI_API_KEY, 
    OPENAI_BASE_URL,
    OPENAI_MODEL, 
    OPENAI_TEMPERATURE
)

# Настройка логгирования
logger = logging.getLogger(__name__)

class SmartAnalysisAgent:
    """
    Интеллектуальный агент анализа с расширенными возможностями рассуждения.
    
    Основные функции:
    - Многоступенчатое планирование и анализ
    - Самооценка качества результатов
    - Контекстуальное понимание задач
    - Генерация проактивных инсайтов
    """
    
    def __init__(self, model_name: str = None, temperature: float = None):
        """
        Инициализация агента.
        
        Args:
            model_name: Название модели LLM (по умолчанию из настроек)
            temperature: Температура для генерации (по умолчанию из настроек)
        """
        self.model_name = model_name or OPENAI_MODEL
        self.temperature = temperature or OPENAI_TEMPERATURE
        
        # Настройка LLM с учетом конфигурации
        llm_kwargs = {
            "model": self.model_name,
            "temperature": self.temperature,
            "openai_api_key": OPENAI_API_KEY,
            "streaming": False
        }
        
        if OPENAI_BASE_URL:
            llm_kwargs["base_url"] = OPENAI_BASE_URL
        
        self.llm = ChatOpenAI(**llm_kwargs)
        
        # История рассуждений для self-reflection
        self.reasoning_history: List[Dict[str, Any]] = []
        
        # Контекст текущего анализа
        self.current_context: Dict[str, Any] = {}
        
        # Настройка системных промптов
        self._setup_prompts()
        
        logger.info(f"SmartAnalysisAgent инициализирован с моделью {model_name}")
    
    def _setup_prompts(self) -> None:
        """Настройка системных промптов для различных этапов анализа."""
        
        # Промпт для планирования анализа
        self.planning_prompt = ChatPromptTemplate.from_messages([
            ("system", """
Ты - экспертный аналитик данных со способностью к глубокому многоступенчатому планированию.

ТВОЯ РОЛЬ: Создавать детальные планы анализа данных, адаптируясь к специфике задачи.

ПРИНЦИПЫ ПЛАНИРОВАНИЯ:
1. Разбивай сложные задачи на логические этапы (3-7 шагов)
2. Определяй зависимости между этапами
3. Предусматривай проверочные точки
4. Планируй альтернативные пути при неудаче
5. Включай этап самооценки результатов

ФОРМАТ ОТВЕТА:
```json
{{
    "analysis_plan": {{
        "goal": "Четкая формулировка цели анализа",
        "steps": [
            {{
                "step_number": 1,
                "action": "Конкретное действие",
                "rationale": "Обоснование необходимости этого шага",
                "expected_output": "Ожидаемый результат",
                "dependencies": ["предыдущие шаги"],
                "fallback": "План действий при неудаче"
            }}
        ],
        "success_criteria": "Критерии успешного выполнения",
        "risk_assessment": "Оценка рисков и сложностей"
    }}
}}
```
            """),
            ("user", "Задача для анализа: {question}\nДанные: {data_summary}")
        ])
        
        # Промпт для выполнения анализа
        self.analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """
Ты - экспертный аналитик данных, выполняющий конкретный этап анализа.

ТВОЯ РОЛЬ: Качественно выполнять каждый этап согласно плану.

ПРИНЦИПЫ ВЫПОЛНЕНИЯ:
1. Строго следуй плану анализа
2. Используй только релевантные методы
3. Документируй все находки
4. Выявляй паттерны и аномалии
5. Готовь данные для следующего этапа

ФОРМАТ ОТВЕТА:
```json
{{
    "step_result": {{
        "step_number": 1,
        "status": "completed|failed|partial",
        "findings": ["ключевые находки"],
        "data_insights": ["инсайты из данных"],
        "next_step_data": "подготовленные данные для следующего шага",
        "confidence": 0.85,
        "issues": ["выявленные проблемы"]
    }}
}}
```
            """),
            ("user", "План: {plan}\nТекущий шаг: {current_step}\nДанные: {data}")
        ])
        
        # Промпт для самооценки
        self.reflection_prompt = ChatPromptTemplate.from_messages([
            ("system", """
Ты - критический эксперт по оценке качества анализа данных.

ТВОЯ РОЛЬ: Проводить честную самооценку результатов всей цепочки анализа.

КРИТЕРИИ ОЦЕНКИ:
1. Соответствие первоначальной цели (0-100%)
2. Качество использованных методов
3. Обоснованность выводов
4. Полнота анализа
5. Практическая ценность результатов

ВОПРОСЫ ДЛЯ РЕФЛЕКСИИ:
- Достигнута ли изначальная цель?
- Использованы ли оптимальные методы?
- Есть ли пропущенные аспекты?
- Насколько надежны выводы?
- Какие улучшения возможны?

ФОРМАТ ОТВЕТА:
```json
{{
    "self_assessment": {{
        "goal_achievement": 85,
        "method_quality": 90,
        "conclusion_validity": 80,
        "completeness": 75,
        "practical_value": 88,
        "overall_score": 83.6,
        "strengths": ["что получилось хорошо"],
        "weaknesses": ["что можно улучшить"],
        "missing_aspects": ["что пропущено"],
        "recommendations": ["рекомендации по улучшению"],
        "confidence_level": "high|medium|low"
    }}
}}
```
            """),
            ("user", "Исходная цель: {original_goal}\nВыполненный анализ: {analysis_results}\nПолученные результаты: {final_results}")
        ])
        
        # Промпт для проактивных инсайтов
        self.insights_prompt = ChatPromptTemplate.from_messages([
            ("system", """
Ты - экспертный консультант, генерирующий проактивные инсайты.

ТВОЯ РОЛЬ: Выходить за рамки поставленной задачи и предлагать дополнительную ценность.

ТИПЫ ПРОАКТИВНЫХ ИНСАЙТОВ:
1. Связанные паттерны в данных
2. Предупреждения о рисках
3. Возможности для оптимизации
4. Рекомендации для смежных областей
5. Прогнозы развития ситуации

ФОРМАТ ОТВЕТА:
```json
{{
    "proactive_insights": {{
        "related_patterns": ["связанные паттерны"],
        "risk_warnings": ["предупреждения о рисках"],
        "optimization_opportunities": ["возможности оптимизации"],
        "cross_domain_recommendations": ["рекомендации для смежных областей"],
        "future_predictions": ["прогнозы"],
        "additional_questions": ["вопросы для дальнейшего исследования"],
        "priority_level": "high|medium|low"
    }}
}}
```
            """),
            ("user", "Результаты анализа: {analysis_results}\nКонтекст задачи: {context}")
        ])
    
    def create_analysis_plan(self, question: str, data: Union[pd.DataFrame, Dict, str]) -> Dict[str, Any]:
        """
        Создание детального плана анализа.
        
        Args:
            question: Вопрос для анализа
            data: Данные для анализа
            
        Returns:
            Детальный план анализа
        """
        try:
            # Подготавливаем сводку данных
            data_summary = self._prepare_data_summary(data)
            
            # Генерируем план
            response = self.llm.invoke(
                self.planning_prompt.format_messages(
                    question=question,
                    data_summary=data_summary
                )
            )
            
            # Парсим JSON ответ
            plan_json = self._extract_json_from_response(response.content)
            
            # Сохраняем в историю рассуждений
            self.reasoning_history.append({
                "stage": "planning",
                "timestamp": datetime.now(),
                "input": {"question": question, "data_summary": data_summary},
                "output": plan_json,
                "reasoning": "Создан детальный план многоступенчатого анализа"
            })
            
            logger.info(f"Создан план анализа с {len(plan_json.get('analysis_plan', {}).get('steps', []))} шагами")
            return plan_json
            
        except Exception as e:
            logger.error(f"Ошибка создания плана анализа: {e}")
            return {
                "error": f"Ошибка планирования: {str(e)}",
                "analysis_plan": {
                    "goal": question,
                    "steps": [],
                    "success_criteria": "Не определены",
                    "risk_assessment": "Высокий риск из-за ошибки планирования"
                }
            }
    
    def execute_analysis_step(self, plan: Dict[str, Any], step_number: int, data: Any) -> Dict[str, Any]:
        """
        Выполнение конкретного шага анализа.
        
        Args:
            plan: План анализа
            step_number: Номер выполняемого шага
            data: Данные для анализа
            
        Returns:
            Результат выполнения шага
        """
        try:
            steps = plan.get("analysis_plan", {}).get("steps", [])
            if step_number > len(steps):
                raise ValueError(f"Шаг {step_number} не существует в плане")
            
            current_step = steps[step_number - 1]
            
            # Выполняем шаг
            response = self.llm.invoke(
                self.analysis_prompt.format_messages(
                    plan=json.dumps(plan, ensure_ascii=False, indent=2),
                    current_step=json.dumps(current_step, ensure_ascii=False, indent=2),
                    data=str(data)[:1000]  # Ограничиваем размер для промпта
                )
            )
            
            # Парсим результат
            step_result = self._extract_json_from_response(response.content)
            
            # Сохраняем в историю
            self.reasoning_history.append({
                "stage": f"execution_step_{step_number}",
                "timestamp": datetime.now(),
                "input": {"step": current_step, "data_size": len(str(data))},
                "output": step_result,
                "reasoning": f"Выполнен шаг {step_number} анализа"
            })
            
            logger.info(f"Выполнен шаг {step_number}, статус: {step_result.get('step_result', {}).get('status', 'unknown')}")
            return step_result
            
        except Exception as e:
            logger.error(f"Ошибка выполнения шага {step_number}: {e}")
            return {
                "step_result": {
                    "step_number": step_number,
                    "status": "failed",
                    "findings": [],
                    "data_insights": [],
                    "next_step_data": None,
                    "confidence": 0.0,
                    "issues": [f"Ошибка выполнения: {str(e)}"]
                }
            }
    
    def perform_self_reflection(self, original_goal: str, analysis_results: List[Dict[str, Any]], final_results: Any) -> Dict[str, Any]:
        """
        Проведение самооценки качества всей цепочки анализа.
        
        Args:
            original_goal: Изначальная цель анализа
            analysis_results: Результаты выполненных шагов
            final_results: Финальные результаты
            
        Returns:
            Результат самооценки
        """
        try:
            # Выполняем самооценку
            response = self.llm.invoke(
                self.reflection_prompt.format_messages(
                    original_goal=original_goal,
                    analysis_results=json.dumps(analysis_results, ensure_ascii=False, indent=2),
                    final_results=str(final_results)[:1000]
                )
            )
            
            # Парсим результат
            reflection_result = self._extract_json_from_response(response.content)
            
            # Сохраняем в историю
            self.reasoning_history.append({
                "stage": "self_reflection",
                "timestamp": datetime.now(),
                "input": {
                    "original_goal": original_goal,
                    "steps_completed": len(analysis_results),
                    "final_results_type": type(final_results).__name__
                },
                "output": reflection_result,
                "reasoning": "Проведена самооценка всей цепочки анализа"
            })
            
            overall_score = reflection_result.get("self_assessment", {}).get("overall_score", 0)
            logger.info(f"Самооценка завершена, общий балл: {overall_score}")
            return reflection_result
            
        except Exception as e:
            logger.error(f"Ошибка самооценки: {e}")
            return {
                "self_assessment": {
                    "goal_achievement": 0,
                    "method_quality": 0,
                    "conclusion_validity": 0,
                    "completeness": 0,
                    "practical_value": 0,
                    "overall_score": 0,
                    "strengths": [],
                    "weaknesses": [f"Ошибка самооценки: {str(e)}"],
                    "missing_aspects": ["Самооценка не выполнена"],
                    "recommendations": ["Исправить ошибки в системе самооценки"],
                    "confidence_level": "low"
                }
            }
    
    def generate_proactive_insights(self, analysis_results: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Генерация проактивных инсайтов.
        
        Args:
            analysis_results: Результаты анализа
            context: Контекст задачи
            
        Returns:
            Проактивные инсайты
        """
        try:
            # Генерируем инсайты
            response = self.llm.invoke(
                self.insights_prompt.format_messages(
                    analysis_results=str(analysis_results)[:1500],
                    context=json.dumps(context, ensure_ascii=False, indent=2)
                )
            )
            
            # Парсим результат
            insights_result = self._extract_json_from_response(response.content)
            
            # Сохраняем в историю
            self.reasoning_history.append({
                "stage": "proactive_insights",
                "timestamp": datetime.now(),
                "input": {"context": context},
                "output": insights_result,
                "reasoning": "Сгенерированы проактивные инсайты"
            })
            
            insights_count = len(insights_result.get("proactive_insights", {}).get("related_patterns", []))
            logger.info(f"Сгенерировано {insights_count} проактивных инсайтов")
            return insights_result
            
        except Exception as e:
            logger.error(f"Ошибка генерации инсайтов: {e}")
            return {
                "proactive_insights": {
                    "related_patterns": [],
                    "risk_warnings": [f"Ошибка генерации инсайтов: {str(e)}"],
                    "optimization_opportunities": [],
                    "cross_domain_recommendations": [],
                    "future_predictions": [],
                    "additional_questions": [],
                    "priority_level": "low"
                }
            }
    
    def _prepare_data_summary(self, data: Union[pd.DataFrame, Dict, str]) -> str:
        """Подготовка краткой сводки данных для промпта."""
        try:
            if isinstance(data, pd.DataFrame):
                return f"DataFrame: {data.shape[0]} строк, {data.shape[1]} колонок. Колонки: {list(data.columns)}"
            elif isinstance(data, dict):
                return f"Dictionary: {len(data)} ключей. Ключи: {list(data.keys())}"
            else:
                return f"Data type: {type(data).__name__}, length: {len(str(data))}"
        except Exception:
            return "Не удалось определить структуру данных"
    
    def _extract_json_from_response(self, response_text: str) -> Dict[str, Any]:
        """Извлечение JSON из ответа LLM."""
        try:
            # Ищем JSON между ```json и ```
            import re
            json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                return json.loads(json_str)
            
            # Пытаемся парсить весь ответ как JSON
            return json.loads(response_text)
            
        except Exception as e:
            logger.warning(f"Не удалось извлечь JSON из ответа: {e}")
            return {"error": "Не удалось распарсить ответ", "raw_response": response_text}
    
    def get_reasoning_history(self) -> List[Dict[str, Any]]:
        """Получение истории рассуждений для отладки."""
        return self.reasoning_history
    
    def clear_reasoning_history(self) -> None:
        """Очистка истории рассуждений."""
        self.reasoning_history.clear()
        logger.info("История рассуждений очищена")


def get_smart_analysis_agent() -> SmartAnalysisAgent:
    """
    Фабричная функция для создания экземпляра SmartAnalysisAgent.
    
    Returns:
        Настроенный экземпляр SmartAnalysisAgent
    """
    return SmartAnalysisAgent()


# Тестирование (если запускается напрямую)
if __name__ == "__main__":
    # Настройка логгирования для тестирования
    logging.basicConfig(level=logging.INFO)
    
    # Добавляем путь для импорта модулей
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Создание агента
    agent = get_smart_analysis_agent()
    
    # Тестовые данные
    test_data = pd.DataFrame({
        'region': ['Москва', 'СПб', 'Екатеринбург'],
        'spending': [100, 85, 75],
        'population': [12.5, 5.4, 1.5]
    })
    
    # Тестовый анализ
    question = "Проанализируй расходы по регионам"
    
    print("🧠 Тестирование SmartAnalysisAgent...")
    
    # 1. Планирование
    plan = agent.create_analysis_plan(question, test_data)
    print(f"✅ План создан: {len(plan.get('analysis_plan', {}).get('steps', []))} шагов")
    
    # 2. Выполнение первого шага
    step_result = None
    if plan.get('analysis_plan', {}).get('steps'):
        step_result = agent.execute_analysis_step(plan, 1, test_data)
        print(f"✅ Шаг 1 выполнен: {step_result.get('step_result', {}).get('status', 'unknown')}")
    else:
        step_result = {"step_result": {"status": "not_executed", "findings": []}}
        print("⚠️ Шаги плана не найдены")
    
    # 3. Самооценка
    reflection = agent.perform_self_reflection(question, [step_result], test_data)
    score = reflection.get('self_assessment', {}).get('overall_score', 0)
    print(f"✅ Самооценка: {score} баллов")
    
    # 4. Проактивные инсайты
    insights = agent.generate_proactive_insights(step_result, {"domain": "regional_analysis"})
    insights_count = len(insights.get('proactive_insights', {}).get('related_patterns', []))
    print(f"✅ Инсайты: {insights_count} предложений")
    
    print("\n🎉 SmartAnalysisAgent готов к работе!") 