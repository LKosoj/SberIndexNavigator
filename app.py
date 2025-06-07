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
    OPENAI_API_KEY,
    create_streamlit_config
)
from src.data.database import initialize_database, get_database_manager
from src.agents.coordinator_agent import get_coordinator_agent
from src.visualization.charts import get_chart_creator
from src.visualization.maps import get_map_creator
from src.utils.pdf_export import generate_qa_pdf, generate_full_history_pdf
from src.utils.analysis_ui import get_analysis_ui_renderer, render_analysis_quick_summary

logger = logging.getLogger(__name__)


@st.cache_data(ttl=300)  # Кэшируем на 5 минут
def get_database_info():
    """
    Получение информации о базе данных с кэшированием.
    
    Returns:
        Словарь с информацией о таблицах и данных
    """
    try:
        from datetime import datetime
        db_manager = get_database_manager()
        summary = db_manager.get_database_summary()
        
        # Добавляем время обновления
        summary["last_updated"] = datetime.now().strftime("%H:%M:%S")
        
        return summary
    except Exception as e:
        logger.error(f"Ошибка получения информации о БД: {e}")
        return {
            "tables": {},
            "total_tables": 0,
            "total_records": 0,
            "available_regions": [],
            "last_updated": "Ошибка"
        }


def render_database_info():
    """Отрисовка информации о доступных данных в базе."""
    # Заголовок с кнопкой обновления
    col1, col2 = st.columns([3, 1])
    with col1:
        st.header("📊 Доступные данные")
    with col2:
        if st.button("🔄", help="Обновить информацию о данных", key="refresh_db_info"):
            # Очищаем кэш и принудительно обновляем данные
            get_database_info.clear()
            st.rerun()
    
    db_info = get_database_info()
    
    if db_info["total_tables"] == 0:
        st.warning("⚠️ Данные не загружены")
        return
    
    # Общая статистика
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📈 Таблиц", db_info["total_tables"])
    with col2:
        st.metric("📋 Всего записей", f"{db_info['total_records']:,}")
    with col3:
        st.metric("🕐 Обновлено", db_info.get("last_updated", "N/A"))
    
    # Информация о таблицах
    st.subheader("🗃️ Таблицы:")
    for table_name, table_info in db_info["tables"].items():
        with st.expander(f"`{table_name}` - {table_info['description']}", expanded=False):
            st.write(f"**Записей:** {table_info['record_count']:,}")
            st.write(f"**Колонки:** {', '.join(table_info['columns'][:5])}")
            if len(table_info['columns']) > 5:
                st.write(f"... и ещё {len(table_info['columns']) - 5} колонок")
    
    # Информация о регионах
    if db_info["available_regions"]:
        st.subheader("🌍 Муниципалитеты:")
        region_count = len(db_info["available_regions"])
        st.write(f"**Доступно:** {region_count} муниципалитетов")
        
        # Показываем несколько примеров
        if region_count > 0:
            examples = db_info["available_regions"][:5]
            st.write(f"**Примеры:** {', '.join(examples)}")
            if region_count > 5:
                st.write(f"... и ещё {region_count - 5} муниципалитетов")
    else:
        st.subheader("🌍 Муниципалитеты:")
        st.write("Информация о регионах недоступна")


def setup_page_config():
    """Настройка конфигурации страницы Streamlit."""
    # Создаем конфигурацию темы на основе переменной окружения
    create_streamlit_config()
    
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
    
# agents_initialized больше не нужен - используется только координатор
    
    # Добавляем флаг для предотвращения дублирования обработки
    if "processing_request" not in st.session_state:
        st.session_state.processing_request = False
    
    # Добавляем настройки анализа
    if "enable_analysis" not in st.session_state:
        st.session_state.enable_analysis = True
    
    if "analysis_mode" not in st.session_state:
        st.session_state.analysis_mode = "auto"  # auto, basic, advanced


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
                # Устанавливаем флаг для обработки в основном интерфейсе
                st.session_state.demo_question = question
        
        st.markdown("---")
        
        # Настройки анализа
        st.header("🔬 Настройки анализа")
        
        # Включение/отключение анализа
        st.session_state.enable_analysis = st.checkbox(
            "🧠 Интеллектуальный анализ",
            value=st.session_state.enable_analysis,
            help="Включить AI-анализ данных и генерацию рекомендаций"
        )
        
        if st.session_state.enable_analysis:
            # Режим анализа
            st.session_state.analysis_mode = st.selectbox(
                "📊 Режим анализа:",
                options=["auto", "basic", "advanced"],
                format_func=lambda x: {
                    "auto": "🤖 Автоматический",
                    "basic": "📈 Базовый", 
                    "advanced": "🔬 Расширенный"
                }[x],
                index=["auto", "basic", "advanced"].index(st.session_state.analysis_mode),
                help="Выберите глубину анализа данных"
            )
        
        st.markdown("---")
        
        # Экспорт истории в PDF
        st.header("📄 Экспорт")
        if len(st.session_state.messages) > 0:
            if st.button("📋 Экспорт всей истории в PDF", use_container_width=True):
                export_full_history_to_pdf()
        else:
            st.info("История пуста")
        
        st.markdown("---")
        
        # Динамическая информация о базе данных
        render_database_info()
        
        st.markdown("---")
        
        st.header("🔧 Статус системы")
        if st.session_state.database_initialized:
            st.success("✅ База данных подключена")
        else:
            st.error("❌ База данных не подключена")
        
        st.success("✅ LangGraph Coordinator готов")
        
        if st.session_state.enable_analysis:
            st.success("✅ Анализ данных включен")
        else:
            st.info("ℹ️ Анализ данных отключен")


def render_chat_interface():
    """Отрисовка чат-интерфейса."""
    st.header("💬 Задайте вопрос о данных")
    
    # Отображение истории сообщений
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Отображение данных и визуализации
            if "data" in message and not message["data"].empty:
                st.subheader("📊 Данные")
                st.dataframe(message["data"], use_container_width=True)
            
            if "visualization" in message:
                st.subheader("📈 Визуализация")
                render_visualization(message["visualization"])
            
            # === ОТОБРАЖЕНИЕ РЕЗУЛЬТАТОВ АНАЛИЗА ===
            if "analysis" in message and message["analysis"]:
                analysis_result = message["analysis"]
                if analysis_result.get("success", False):
                    st.markdown("---")
                    
                    # Получаем рендерер анализа
                    analysis_ui = get_analysis_ui_renderer()
                    
                    # Определяем режим отображения (из настроек или по умолчанию)
                    display_mode = st.session_state.get("analysis_mode", "basic")
                    
                    if display_mode == "advanced":
                        # Полное отображение анализа
                        with st.expander("🔬 Детальный анализ данных", expanded=False):
                            analysis_ui.render_analysis_results(analysis_result)
                    else:
                        # Краткое отображение
                        ai_insights = analysis_result.get("ai_insights", {})
                        if ai_insights and "error" not in ai_insights:
                            with st.expander("🧠 Результаты анализа", expanded=False):
                                render_analysis_quick_summary(ai_insights)
                        
                        # Всегда показываем рекомендации
                        recommendations = analysis_result.get("recommendations", [])
                        if recommendations:
                            with st.expander("💡 Рекомендации", expanded=True):
                                for idx, rec in enumerate(recommendations, 1):
                                    st.success(f"**{idx}.** {rec}")
            
            # Кнопка экспорта в PDF для ответов с данными
            # Проверяем все возможные условия для отладки
            has_data = "data" in message and message["data"] is not None and not message["data"].empty
            has_viz = "visualization" in message and message["visualization"] is not None
            is_assistant = message["role"] == "assistant"
            
            # Добавляем дебаг информацию (только в режиме разработки)
            if is_assistant and st.session_state.get("debug_mode", False):
                st.caption(f"Debug: has_data={has_data}, has_viz={has_viz}, keys={list(message.keys())}")
            
            if is_assistant and (has_data or has_viz):
                # Получаем предыдущий вопрос пользователя
                user_question = ""
                if i > 0 and st.session_state.messages[i-1]["role"] == "user":
                    user_question = st.session_state.messages[i-1]["content"]
                
                col1, col2, col3 = st.columns([1, 1, 4])
                with col1:
                    if st.button(f"📄 Экспорт PDF", key=f"export_pdf_{i}"):
                        export_to_pdf(
                            question=user_question,
                            answer=message["content"],
                            data=message.get("data"),
                            sql_query=message.get("sql_query"),
                            visualization_config=message.get("visualization"),
                            analysis_result=message.get("analysis"),
                            message_index=i
                        )
                
                # Кнопка повторного анализа (через координатора)
                with col2:
                    if (has_data and st.session_state.enable_analysis):
                        if st.button(f"🔄 Переанализ", key=f"reanalyze_{i}"):
                            st.info("💡 Для переанализа задайте вопрос заново - координатор выполнит полный анализ")
    
    # Поле ввода для нового вопроса
    if prompt := st.chat_input("Введите ваш вопрос о данных индексов..."):
        process_user_input(prompt)


def export_full_history_to_pdf():
    """Экспорт всей истории чата в PDF с полными разделами для каждого Q&A."""
    try:
        if not st.session_state.messages:
            st.warning("История чата пуста")
            return
        
        with st.spinner("Создание полного PDF отчета..."):
            # Собираем все Q&A пары
            qa_pairs = []
            current_question = ""
            
            for message in st.session_state.messages:
                if message["role"] == "user":
                    current_question = message["content"]
                elif message["role"] == "assistant" and current_question:
                    qa_pairs.append({
                        "question": current_question,
                        "answer": message["content"],
                        "data": message.get("data"),
                        "sql_query": message.get("sql_query"),
                        "visualization": message.get("visualization"),
                        "analysis": message.get("analysis")
                    })
                    current_question = ""
            
            if not qa_pairs:
                st.warning("В истории нет завершенных вопросов и ответов")
                return
            
            # Создаем PDF с несколькими отдельными Q&A секциями
            # Мы будем генерировать один большой PDF со всеми разделами
            
            # Используем модифицированную версию PDF генератора для множественных Q&A
            pdf_bytes = generate_full_history_pdf(qa_pairs)
            
            # Создаем имя файла
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"sber_index_full_history_{timestamp}.pdf"
            
            # Кнопка скачивания
            st.download_button(
                label="⬇️ Скачать полный PDF отчет",
                data=pdf_bytes,
                file_name=filename,
                mime="application/pdf",
                key=f"download_full_pdf_{timestamp}"
            )
            
            st.success(f"✅ Полный PDF отчет готов! Включает {len(qa_pairs)} вопросов и ответов.")
            
    except Exception as e:
        logger.error(f"Ошибка экспорта полной истории в PDF: {e}")
        st.error(f"Ошибка создания полного PDF отчета: {e}")


def export_to_pdf(
    question: str,
    answer: str,
    data: Optional[pd.DataFrame] = None,
    sql_query: Optional[str] = None,
    visualization_config: Optional[Dict[str, Any]] = None,
    analysis_result: Optional[Dict[str, Any]] = None,
    message_index: int = 0
):
    """
    Экспорт вопроса и ответа в PDF.
    
    Args:
        question: Вопрос пользователя
        answer: Ответ системы
        data: Данные таблицы
        sql_query: SQL запрос
        visualization_config: Конфигурация визуализации
        analysis_result: Результат анализа
        message_index: Индекс сообщения для уникальности
    """
    try:
        with st.spinner("Создание PDF отчета..."):
            # Генерируем PDF
            pdf_bytes = generate_qa_pdf(
                question=question,
                answer=answer,
                data=data,
                sql_query=sql_query,
                visualization_config=visualization_config,
                analysis_result=analysis_result
            )
            
            # Создаем имя файла
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"sber_index_report_{timestamp}.pdf"
            
            # Кнопка скачивания
            st.download_button(
                label="⬇️ Скачать PDF отчет",
                data=pdf_bytes,
                file_name=filename,
                mime="application/pdf",
                key=f"download_pdf_{message_index}_{timestamp}"
            )
            
            st.success("✅ PDF отчет готов к скачиванию!")
            
    except Exception as e:
        logger.error(f"Ошибка экспорта в PDF: {e}")
        st.error(f"Ошибка создания PDF отчета: {e}")


def process_user_input(user_input: str):
    """
    Обработка пользовательского ввода.
    
    Args:
        user_input: Вопрос пользователя
    """
    # Предотвращаем дублирование обработки
    if st.session_state.processing_request:
        return
        
    st.session_state.processing_request = True
    
    try:
        # Добавляем сообщение пользователя
        st.session_state.messages.append({
            "role": "user", 
            "content": user_input
        })
        
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Обработка запроса с помощью LangGraph Coordinator
        with st.chat_message("assistant"):
            with st.spinner("🧠 Планирую выполнение задачи..."):
                try:
                    # Используем новую систему на базе LangGraph
                    coordinator = get_coordinator_agent()
                    result = coordinator.process_question(user_input)
                    
                    if result["success"]:
                        # Отображаем ответ координатора
                        st.markdown(result["response"])
                        
                        # Если есть данные, отображаем их
                        if not result["data"].empty:
                            st.subheader("📊 Найденные данные")
                            st.dataframe(result["data"], use_container_width=True)
                        
                        # Если есть визуализация, отображаем её
                        if result.get("visualization"):
                            st.subheader("📈 Визуализация")
                            render_visualization(result["visualization"])
                        
                        # Если есть анализ, отображаем его
                        if result.get("analysis"):
                            analysis = result["analysis"]
                            if analysis.get("ai_insights"):
                                st.markdown("---")
                                st.subheader("🧠 AI Анализ")
                                render_analysis_quick_summary(analysis["ai_insights"])
                            
                            # Рекомендации
                            recommendations = analysis.get("recommendations", [])
                            if recommendations:
                                st.subheader("💡 Рекомендации")
                                for i, rec in enumerate(recommendations[:3], 1):
                                    st.success(f"**{i}.** {rec}")
                        
                        # Сохраняем в историю
                        assistant_message = {
                            "role": "assistant",
                            "content": result["response"],
                            "data": result["data"],
                            "sql_query": result.get("sql_query", ""),
                            "analysis": result.get("analysis", {}),
                            "visualization": result.get("visualization", {})
                        }
                        
                        st.session_state.messages.append(assistant_message)
                        
                        # Добавляем кнопку экспорта
                        if not result["data"].empty or result.get("analysis") or result.get("visualization"):
                            st.markdown("---")
                            col1, col2, col3 = st.columns([1, 1, 4])
                            with col1:
                                if st.button("📄 Экспорт PDF", key=f"export_current_result"):
                                    export_to_pdf(
                                        question=user_input,
                                        answer=result["response"],
                                        data=result["data"] if not result["data"].empty else None,
                                        sql_query=result.get("sql_query"),
                                        visualization_config=result.get("visualization"),
                                        analysis_result=result.get("analysis"),
                                        message_index=len(st.session_state.messages)-1
                                    )
                    else:
                        error_msg = result.get("error", "Неизвестная ошибка")
                        st.error(f"❌ Ошибка: {error_msg}")
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"Ошибка: {error_msg}"
                        })
                        
                except Exception as e:
                    logger.error(f"Ошибка обработки запроса: {e}")
                    st.error(f"Произошла ошибка: {e}")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"Произошла ошибка: {e}"
                    })
    finally:
        # Сбрасываем флаг обработки
        st.session_state.processing_request = False


# create_visualization удалена - теперь визуализация создается в координаторе


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


def check_and_process_demo_question():
    """Проверка и обработка демо-вопроса в конце выполнения."""
    if "demo_question" in st.session_state:
        demo_question = st.session_state.demo_question
        del st.session_state.demo_question
        
        # Используем ту же логику что и для обычных вопросов
        # Это гарантирует что анализ данных тоже будет выполнен
        process_user_input(demo_question)


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
    
    # Отрисовка интерфейса
    render_header()
    
    # Основной контент в колонках
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Основной чат-интерфейс
        render_chat_interface()
    
    with col2:
        # Сайдбар с кнопками
        render_sidebar()
    
    # Футер
    st.markdown("---")
    st.markdown(
        "🏆 **SberIndexNavigator** - Демонстрация возможностей "
        "комбинации 'Вопрос → Аналитический ответ → Автовизуализация'"
    )
    
    # В самом конце проверяем и обрабатываем демо-вопросы
    check_and_process_demo_question()


if __name__ == "__main__":
    main() 