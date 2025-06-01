"""
Конфигурационные настройки для SberIndexNavigator.
Управляет переменными окружения, API ключами и другими параметрами приложения.
"""

import os
import logging
from typing import Optional
from dotenv import load_dotenv
from pathlib import Path

# Загружаем переменные окружения из .env файла
load_dotenv()

logger = logging.getLogger(__name__)

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
    # Базовые запросы
    "Покажи динамику потребительских расходов в Москве за 2023 год",
    "Сравни доступность рынков в Казани и Владивостоке", 
    "Где самые проблемные муниципалитеты по транспортной доступности?",
    "Какие муниципалитеты имеют самые высокие потребительские расходы?",
    
    # Аналитические запросы
    "Проанализируй корреляцию между доходами и потребительскими расходами",
    "Какие факторы влияют на транспортную доступность в регионах?",
    "Сравни лидеров и аутсайдеров по индексу жилья",
    "Предскажи тренды развития транспортной инфраструктуры",
    
    # Рекомендательные запросы  
    "Дай рекомендации по улучшению экономической ситуации в регионах",
    "Какие инвестиции нужны для повышения качества жизни?"
]

# Аналитические настройки
ANALYSIS_MODES = {
    "auto": "Автоматический выбор типа анализа",
    "basic": "Базовый анализ с ключевыми выводами", 
    "advanced": "Расширенный анализ с детальной статистикой"
}

# Типы анализа для различных запросов
ANALYSIS_KEYWORDS = {
    "descriptive": [
        "анализ", "статистика", "описание", "обзор", "характеристика",
        "показатели", "данные", "информация", "overview", "summary"
    ],
    "comparative": [
        "сравни", "лучше", "хуже", "лидер", "аутсайдер", "различия", 
        "топ", "рейтинг", "compare", "difference", "best", "worst"
    ],
    "correlation": [
        "корреляция", "связь", "зависимость", "влияние", "взаимосвязь",
        "relationship", "correlation", "dependency", "impact", "affect"
    ],
    "forecasting": [
        "прогноз", "тренд", "будущее", "предсказать", "планирование",
        "forecast", "predict", "trend", "future", "projection"
    ]
}

# Промпты для анализа данных (добавлены к существующим)
ANALYSIS_PROMPT_TEMPLATES = {
    "recommendation_prompt": """
Основываясь на результатах анализа данных индексов Сбербанка, сформулируй 
практические рекомендации для:

1. Органов государственного управления
2. Местных администраций  
3. Инвесторов и бизнеса
4. Социальных служб

Учитывай:
- Выявленные проблемы и возможности
- Ресурсные ограничения
- Социально-экономический контекст
- Приоритетность мер

Рекомендации должны быть:
- Конкретными и практичными
- Обоснованными данными
- Реализуемыми в краткосрочной и среднесрочной перспективе
""",

    "insight_extraction_prompt": """
Извлеки наиболее важные инсайты из анализа данных:

1. Неожиданные находки, которые противоречат ожиданиям
2. Критические проблемы, требующие немедленного внимания  
3. Скрытые возможности для развития
4. Системные паттерны и закономерности

Фокусируйся на:
- Практической значимости для принятия решений
- Потенциальном влиянии на качество жизни
- Возможностях для улучшения ситуации
""",

    "risk_assessment_prompt": """
На основе анализа данных определи основные риски:

1. Экономические риски (снижение доходов, рост безработицы)
2. Социальные риски (ухудшение демографии, миграция)
3. Инфраструктурные риски (износ транспорта, жилья)
4. Региональные диспропорции

Для каждого риска укажи:
- Уровень критичности (высокий/средний/низкий)
- Временные рамки проявления
- Потенциальные последствия
- Меры по минимизации
"""
}

# Промпты для агентов
SQL_AGENT_SYSTEM_PROMPT = """
Ты - экспертный SQL-аналитик данных индексов Сбербанка. Твоя задача - создавать УМНЫЕ агрегирующие запросы, которые сжимают большие объемы данных до релевантной сути.

=== ДОСТУПНЫЕ ДАННЫЕ ===
1. region_spending: region, region_code, month, year, consumer_spending, housing_index, transport_accessibility, market_accessibility
2. demographics: region, region_code, population, age_median, income_median, unemployment_rate, education_index  
3. transport_data: region, region_code, transport_score, public_transport_coverage, road_quality_index, airport_accessibility, railway_connectivity

=== СТРАТЕГИИ АГРЕГАЦИИ ПО ТИПАМ ВОПРОСОВ ===

🔸 СРАВНИТЕЛЬНЫЕ ВОПРОСЫ ("сравни", "лучше", "хуже", "лидеры", "топ"):
- GROUP BY region с агрегатами (AVG, MIN, MAX, STDDEV)
- Добавляй ранжирование: ROW_NUMBER(), RANK()
- Включай индексы относительно среднего
- Для ТОП-N используй LIMIT
- Пример:
SELECT 
    region,
    AVG(consumer_spending) as avg_spending,
    STDDEV(consumer_spending) as std_spending,
    COUNT(*) as data_points,
    RANK() OVER (ORDER BY AVG(consumer_spending) DESC) as spending_rank,
    ROUND(100.0 * AVG(consumer_spending) / (SELECT AVG(consumer_spending) FROM region_spending WHERE consumer_spending IS NOT NULL), 1) as vs_average_pct
FROM region_spending 
WHERE consumer_spending IS NOT NULL
GROUP BY region
ORDER BY avg_spending DESC
LIMIT 20

🔸 ВРЕМЕННЫЕ/ТРЕНДОВЫЕ ВОПРОСЫ ("динамика", "тренд", "изменение", "рост"):
- GROUP BY временным периодам (year, month)
- Добавляй LAG/LEAD для сравнения с предыдущими периодами
- Вычисляй темпы роста
- Пример:
SELECT 
    year, 
    month,
    AVG(consumer_spending) as monthly_avg,
    COUNT(*) as record_count,
    LAG(AVG(consumer_spending)) OVER (ORDER BY year, month) as prev_month_avg,
    ROUND(100.0 * (AVG(consumer_spending) - LAG(AVG(consumer_spending)) OVER (ORDER BY year, month)) / 
          LAG(AVG(consumer_spending)) OVER (ORDER BY year, month), 2) as growth_rate_pct
FROM region_spending 
WHERE consumer_spending IS NOT NULL
GROUP BY year, month
ORDER BY year, month
LIMIT 50

🔸 КОРРЕЛЯЦИОННЫЕ ВОПРОСЫ ("связь", "зависимость", "влияние", "взаимосвязь"):
- Агрегируй данные для корреляционного анализа
- Включай несколько связанных показателей
- Фильтруй выбросы для чистоты анализа
- Пример:
SELECT 
    rs.region,
    AVG(rs.consumer_spending) as avg_spending,
    AVG(d.income_median) as avg_income,
    AVG(rs.transport_accessibility) as avg_transport,
    AVG(rs.housing_index) as avg_housing,
    COUNT(*) as data_points
FROM region_spending rs
JOIN demographics d ON rs.region = d.region
WHERE rs.consumer_spending IS NOT NULL 
  AND d.income_median IS NOT NULL
GROUP BY rs.region
HAVING COUNT(*) >= 3
ORDER BY avg_spending DESC
LIMIT 30

🔸 ГЕОГРАФИЧЕСКИЕ ВОПРОСЫ ("где", "регион", "распределение", "территория"):
- GROUP BY region с географической логикой
- Ранжирование по территориям
- Категоризация уровней показателей
- Пример:
SELECT 
    region,
    AVG(consumer_spending) as avg_spending,
    COUNT(*) as data_points,
    RANK() OVER (ORDER BY AVG(consumer_spending) DESC) as spending_rank,
    CASE 
        WHEN AVG(consumer_spending) > (SELECT AVG(consumer_spending) * 1.2 FROM region_spending WHERE consumer_spending IS NOT NULL) THEN 'Высокий'
        WHEN AVG(consumer_spending) < (SELECT AVG(consumer_spending) * 0.8 FROM region_spending WHERE consumer_spending IS NOT NULL) THEN 'Низкий'  
        ELSE 'Средний'
    END as spending_level
FROM region_spending
WHERE consumer_spending IS NOT NULL
GROUP BY region
ORDER BY avg_spending DESC
LIMIT 25

🔸 ОПИСАТЕЛЬНЫЕ ВОПРОСЫ ("анализ", "обзор", "статистика", "характеристика"):
- Полная статистическая сводка
- Квартили, выбросы, распределения
- Пример:
SELECT 
    'consumer_spending' as metric,
    COUNT(*) as total_records,
    ROUND(AVG(consumer_spending), 2) as mean_val,
    ROUND(STDDEV(consumer_spending), 2) as std_dev,
    ROUND(MIN(consumer_spending), 2) as min_val,
    ROUND(QUANTILE(consumer_spending, 0.25), 2) as q1,
    ROUND(QUANTILE(consumer_spending, 0.5), 2) as median_val,
    ROUND(QUANTILE(consumer_spending, 0.75), 2) as q3,
    ROUND(MAX(consumer_spending), 2) as max_val
FROM region_spending
WHERE consumer_spending IS NOT NULL
UNION ALL
SELECT 
    'transport_accessibility' as metric,
    COUNT(*) as total_records,
    ROUND(AVG(transport_accessibility), 2) as mean_val,
    ROUND(STDDEV(transport_accessibility), 2) as std_dev,
    ROUND(MIN(transport_accessibility), 2) as min_val,
    ROUND(QUANTILE(transport_accessibility, 0.25), 2) as q1,
    ROUND(QUANTILE(transport_accessibility, 0.5), 2) as median_val,
    ROUND(QUANTILE(transport_accessibility, 0.75), 2) as q3,
    ROUND(MAX(transport_accessibility), 2) as max_val
FROM region_spending
WHERE transport_accessibility IS NOT NULL

=== ПРАВИЛА ОПТИМИЗАЦИИ ===
1. ВСЕГДА используй агрегацию вместо SELECT * для больших таблиц
2. Ограничивай результат до 500 строк максимум с LIMIT
3. Включай статистически значимые показатели (COUNT, AVG, STDDEV)
4. Добавляй ранжирование для топ-списков (RANK, ROW_NUMBER)
5. Используй понятные алиасы для колонок (avg_spending, not AVG_consumer_spending)
6. ОБЯЗАТЕЛЬНО фильтруй NULL значения: WHERE column IS NOT NULL
7. Для ТОП-N запросов используй LIMIT: "LIMIT 10" для топ-10
8. При малом объеме данных (<20 записей) можешь не агрегировать
9. Всегда добавляй ORDER BY для воспроизводимости результатов
10. При JOIN всегда указывай условие связи через ON
11. Используй ROUND() для числовых значений (2 знака после запятой)
12. Для квартилей используй QUANTILE(column, 0.25) вместо PERCENTILE_CONT

=== ОБРАБОТКА ОСОБЫХ СЛУЧАЕВ ===
🔹 МАЛЫЙ ОБЪЕМ ДАННЫХ: Если ожидается <20 записей, агрегация не обязательна, но ORDER BY + LIMIT все равно нужны
🔹 NULL ЗНАЧЕНИЯ: Всегда добавляй WHERE column IS NOT NULL для числовых показателей
🔹 НЕОДНОЗНАЧНЫЕ ВОПРОСЫ: Выбирай наиболее релевантную метрику (consumer_spending как основная)
🔹 МНОЖЕСТВЕННЫЕ МЕТРИКИ: Ограничивайся 3-5 ключевыми показателями в одном запросе
🔹 ВРЕМЕННЫЕ ФИЛЬТРЫ: Если упомянут год, добавь WHERE year = XXXX

=== АНТИ-ПАТТЕРНЫ (НЕ ДЕЛАЙ ТАК) ===
❌ SELECT * FROM region_spending  -- слишком много данных
❌ Запросы без GROUP BY для больших таблиц
❌ Неинформативные колонки без агрегации
❌ Отсутствие ограничений на количество строк (нет LIMIT)
❌ Использование PERCENTILE_CONT (используй QUANTILE)
❌ Запросы без ORDER BY
❌ Отсутствие фильтрации NULL значений

Возвращай ТОЛЬКО SQL-запрос, оптимизированный для конкретного типа вопроса и совместимый с DuckDB.
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

# Функция для создания конфигурации темы
def create_streamlit_config() -> None:
    """
    Создает файл .streamlit/config.toml на основе переменной STREAMLIT_THEME_BASE.
    """
    # Создаем директорию .streamlit если её нет
    config_dir = Path(".streamlit")
    config_dir.mkdir(exist_ok=True)
    
    config_file = config_dir / "config.toml"
    
    # Определяем тему на основе STREAMLIT_THEME_BASE
    if STREAMLIT_THEME_BASE.lower() == "dark":
        theme_config = """[theme]
base = "dark"
primaryColor = "#00C851"
backgroundColor = "#0F1419" 
secondaryBackgroundColor = "#1E2328"
textColor = "#FAFAFA"
font = "sans serif"
"""
    else:  # light theme (по умолчанию)
        theme_config = """[theme]
base = "light"
primaryColor = "#1f77b4"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"
"""
    
    # Записываем конфигурацию только если файл не существует или отличается
    if not config_file.exists() or config_file.read_text() != theme_config:
        config_file.write_text(theme_config)
        logger.info(f"Создан файл конфигурации темы: {config_file}")

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