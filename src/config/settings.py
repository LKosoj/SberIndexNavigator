"""
Конфигурационные настройки для SberIndexNavigator.
Управляет переменными окружения, API ключами и другими параметрами приложения.
"""

import os
import logging
from typing import Optional
from dotenv import load_dotenv

# Загружаем переменные окружения из .env файла
load_dotenv()

# Логирование
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
DEBUG = os.getenv("DEBUG", "False").lower() == "true"

# OpenAI API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.1"))
OPENAI_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "10000"))

# Database Configuration
DATABASE_URL = os.getenv("DATABASE_URL", "data/sber_index.db")

# Streamlit Configuration
STREAMLIT_SERVER_PORT = int(os.getenv("STREAMLIT_SERVER_PORT", "8501"))
STREAMLIT_THEME_BASE = os.getenv("STREAMLIT_THEME_BASE", "light")

# Cache Settings
CACHE_TTL = int(os.getenv("CACHE_TTL", "3600").split('#')[0].strip())  # 1 час в секундах

# Приложение
APP_TITLE = "🧭 SberIndexNavigator"
APP_SUBTITLE = "Интеллектуальный навигатор по данным индексов Сбербанка"

# Демо сценарии
DEMO_QUESTIONS = [
    "Покажи динамику потребительских расходов в Москве за 2023 год",
    "Сравни доступность рынков в Казани и Владивостоке",
    "Где самые проблемные муниципалитеты по транспортной доступности?",
    "Какие муниципалитеты имеют самые высокие потребительские расходы?",
    "Покажи корреляцию между доходами и потребительскими расходами",
    "В каких муниципалитетах самая низкая безработица?"
]

# Промпты для агентов
SQL_AGENT_SYSTEM_PROMPT = """
Ты - эксперт по SQL и аналитике данных индексов Сбербанка.

Доступные таблицы:
1. region_spending: region, region_code, month, year, consumer_spending, housing_index, transport_accessibility, market_accessibility
2. demographics: region, region_code, population, age_median, income_median, unemployment_rate, education_index  
3. transport_data: region, region_code, transport_score, public_transport_coverage, road_quality_index, airport_accessibility, railway_connectivity

Твоя задача - создавать точные SQL-запросы для ответа на вопросы пользователей.

Правила:
- Используй *ТОЛЬКО* существующие таблицы и колонки
- Всегда включай названия муниципалитетов (region) в результаты
- Для временных данных используй поля month и year
- Возвращай готовые к визуализации данные
- Используй понятные алиасы для колонок
"""

VISUALIZATION_TOOL_PROMPT = """
Определи оптимальный тип визуализации для данных:

Правила выбора:
- line: Временные ряды, динамика показателей
- bar: Сравнение значений между регионами/категориями  
- scatter: Корреляции и зависимости между показателями
- map: Географическое распределение данных
- table: Детальные данные, топ-списки

Ответь одним словом: line, bar, scatter, map или table.
"""

# Настройки визуализации
CHART_HEIGHT = 400
CHART_WIDTH = 800
MAP_DEFAULT_ZOOM = 3
MAP_CENTER_LAT = 61.5240
MAP_CENTER_LON = 105.3188

# Цветовая схема
COLOR_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
]

# Настройки производительности
MAX_RETRIES = 3
API_TIMEOUT = 30
QUERY_TIMEOUT = 10

# Валидация конфигурации
def validate_config() -> bool:
    """
    Валидация конфигурации приложения.
    
    Returns:
        True если конфигурация валидна, False иначе
    """
    errors = []
    
    if not OPENAI_API_KEY:
        errors.append("OPENAI_API_KEY не установлен")
    
    if not os.path.exists(os.path.dirname(DATABASE_URL)):
        try:
            os.makedirs(os.path.dirname(DATABASE_URL), exist_ok=True)
        except Exception as e:
            errors.append(f"Не удается создать директорию для БД: {e}")
    
    if errors:
        for error in errors:
            logging.error(error)
        return False
    
    return True


def setup_logging() -> None:
    """Настройка логирования для приложения."""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    if DEBUG:
        level = logging.DEBUG
    else:
        level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
    
    logging.basicConfig(
        level=level,
        format=log_format,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("logs/app.log") if os.path.exists("logs") or os.makedirs("logs", exist_ok=True) else logging.StreamHandler()
        ]
    )


# Инициализация при импорте
setup_logging()

if not validate_config():
    logging.warning("Конфигурация содержит ошибки. Проверьте настройки.") 