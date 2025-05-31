"""
Главное приложение SberIndexNavigator.
Streamlit интерфейс для интеллектуального анализа данных индексов Сбербанка.
"""

import streamlit as st
import pandas as pd
import logging
from typing import Dict, Any, Optional
import sys
import os

# Добавляем путь к модулям
sys.path.append('.')

from src.config.settings import (
    APP_TITLE, 
    APP_SUBTITLE, 
    DEMO_QUESTIONS,
    OPENAI_API_KEY
)
from src.data.database import initialize_database, get_database_manager
from src.agents.sql_agent import get_sql_agent
from src.agents.visualize_tool import get_visualization_analyzer
from src.visualization.charts import get_chart_creator
from src.visualization.maps import get_map_creator

logger = logging.getLogger(__name__)


def setup_page_config():
    """Настройка конфигурации страницы Streamlit."""
    st.set_page_config(
        page_title="SberIndexNavigator",
        page_icon="🧭",
        layout="wide",
        initial_sidebar_state="expanded"
    )


def initialize_session_state():
    """Инициализация состояния сессии."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "database_initialized" not in st.session_state:
        st.session_state.database_initialized = False
    
    if "agents_initialized" not in st.session_state:
        st.session_state.agents_initialized = False


@st.cache_resource
def initialize_app():
    """Инициализация приложения с кэшированием."""
    try:
        # Инициализация базы данных
        initialize_database()
        
        # Проверка API ключа
        if not OPENAI_API_KEY or OPENAI_API_KEY == "your_openai_api_key_here":
            return False, "OpenAI API ключ не настроен. Пожалуйста, установите OPENAI_API_KEY в .env файле."
        
        return True, "Приложение инициализировано успешно"
        
    except Exception as e:
        logger.error(f"Ошибка инициализации приложения: {e}")
        return False, f"Ошибка инициализации: {e}"


def render_sidebar():
    """Отрисовка боковой панели с примерами вопросов."""
    with st.sidebar:
        st.header("🎯 Примеры вопросов")
        st.markdown("Нажмите на вопрос, чтобы использовать его:")
        
        for i, question in enumerate(DEMO_QUESTIONS):
            if st.button(question, key=f"demo_q_{i}", use_container_width=True):
                st.session_state.demo_question = question
        
        st.markdown("---")
        
        st.header("📊 Доступные данные")
        st.markdown("""
        **Таблицы:**
        - `region_spending` - расходы по регионам
        - `demographics` - демографические данные  
        - `transport_data` - транспортная доступность
        
        **Регионы:**
        - Москва, Санкт-Петербург, Казань
        - Владивосток, Новосибирск и др.
        """)
        
        st.markdown("---")
        
        st.header("🔧 Статус системы")
        if st.session_state.database_initialized:
            st.success("✅ База данных подключена")
        else:
            st.error("❌ База данных не подключена")
        
        if st.session_state.agents_initialized:
            st.success("✅ AI агенты готовы")
        else:
            st.warning("⚠️ AI агенты не инициализированы")


def render_chat_interface():
    """Отрисовка чат-интерфейса."""
    st.header("💬 Задайте вопрос о данных")
    
    # Отображение истории сообщений
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Отображение данных и визуализации
            if "data" in message and not message["data"].empty:
                st.subheader("📊 Данные")
                st.dataframe(message["data"], use_container_width=True)
            
            if "visualization" in message:
                st.subheader("📈 Визуализация")
                render_visualization(message["visualization"])
    
    # Обработка демо-вопроса из sidebar
    if "demo_question" in st.session_state:
        process_user_input(st.session_state.demo_question)
        del st.session_state.demo_question
    
    # Поле ввода для нового вопроса
    if prompt := st.chat_input("Введите ваш вопрос о данных индексов..."):
        process_user_input(prompt)


def process_user_input(user_input: str):
    """
    Обработка пользовательского ввода.
    
    Args:
        user_input: Вопрос пользователя
    """
    # Добавляем сообщение пользователя
    st.session_state.messages.append({
        "role": "user", 
        "content": user_input
    })
    
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Обработка запроса
    with st.chat_message("assistant"):
        with st.spinner("Анализирую ваш вопрос..."):
            try:
                # Получаем SQL-агент
                sql_agent = get_sql_agent()
                
                # Анализируем вопрос
                result = sql_agent.analyze_question(user_input)
                
                if result["success"]:
                    # Отображаем результат
                    st.markdown(f"**Анализ:** {user_input}")
                    
                    if not result["data"].empty:
                        st.subheader("📊 Найденные данные")
                        st.dataframe(result["data"], use_container_width=True)
                        
                        # Создаем визуализацию
                        visualization_config = create_visualization(result["data"], user_input)
                        
                        if visualization_config:
                            st.subheader("📈 Визуализация")
                            render_visualization(visualization_config)
                        
                        # Сохраняем в историю
                        assistant_message = {
                            "role": "assistant",
                            "content": f"**Анализ:** {user_input}",
                            "data": result["data"],
                            "sql_query": result["sql_query"]
                        }
                        
                        if visualization_config:
                            assistant_message["visualization"] = visualization_config
                        
                        st.session_state.messages.append(assistant_message)
                    else:
                        st.warning("Данные не найдены. Попробуйте переформулировать вопрос.")
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": "Данные не найдены. Попробуйте переформулировать вопрос."
                        })
                else:
                    st.error(f"Ошибка анализа: {result['error']}")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"Ошибка анализа: {result['error']}"
                    })
                    
            except Exception as e:
                logger.error(f"Ошибка обработки запроса: {e}")
                st.error(f"Произошла ошибка: {e}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Произошла ошибка: {e}"
                })


def create_visualization(data: pd.DataFrame, question: str) -> Optional[Dict[str, Any]]:
    """
    Создание конфигурации визуализации для данных.
    
    Args:
        data: DataFrame с данными
        question: Вопрос пользователя
        
    Returns:
        Конфигурация визуализации или None
    """
    try:
        # Получаем анализатор визуализации
        viz_analyzer = get_visualization_analyzer()
        
        # Определяем тип визуализации
        chart_type = viz_analyzer.determine_chart_type(data, question)
        
        # Получаем конфигурацию
        config = viz_analyzer.get_visualization_config(chart_type, data)
        config["title"] = f"Анализ: {question}"
        
        return config
        
    except Exception as e:
        logger.error(f"Ошибка создания визуализации: {e}")
        return None


def render_visualization(config: Dict[str, Any]):
    """
    Отрисовка визуализации на основе конфигурации.
    
    Args:
        config: Конфигурация визуализации
    """
    try:
        chart_type = config.get("type", "table")
        
        if chart_type == "map":
            # Используем карты
            map_creator = get_map_creator()
            map_creator.display_map("scatter", config)
        elif chart_type == "table":
            # Простая таблица
            st.dataframe(config["data"], use_container_width=True)
        else:
            # Используем графики
            chart_creator = get_chart_creator()
            chart_creator.display_chart(chart_type, config)
            
    except Exception as e:
        logger.error(f"Ошибка отрисовки визуализации: {e}")
        st.error(f"Ошибка отрисовки визуализации: {e}")


def render_header():
    """Отрисовка заголовка приложения."""
    st.title(APP_TITLE)
    st.markdown(f"*{APP_SUBTITLE}*")
    st.markdown("---")


def main():
    """Главная функция приложения."""
    setup_page_config()
    initialize_session_state()
    
    # Инициализация приложения
    success, message = initialize_app()
    
    if not success:
        st.error(f"❌ {message}")
        st.stop()
    
    # Обновляем статус инициализации
    st.session_state.database_initialized = True
    st.session_state.agents_initialized = success
    
    # Отрисовка интерфейса
    render_header()
    
    # Основной контент в колонках
    col1, col2 = st.columns([3, 1])
    
    with col1:
        render_chat_interface()
    
    with col2:
        render_sidebar()
    
    # Футер
    st.markdown("---")
    st.markdown(
        "🏆 **SberIndexNavigator** - Демонстрация возможностей "
        "комбинации 'Вопрос → Аналитический ответ → Автовизуализация'"
    )


if __name__ == "__main__":
    main() 