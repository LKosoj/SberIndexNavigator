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
    "Какие факторы влияют на транспортную доступность в муниципалитетах?",
    "Сравни лидеров и аутсайдеров по индексу жилья",
    "Предскажи тренды развития транспортной инфраструктуры",
    
    # Рекомендательные запросы  
    "Дай рекомендации по улучшению экономической ситуации в муниципалитетах",
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
4. Муниципальные диспропорции

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

=== УНИВЕРСАЛЬНЫЕ ПРИНЦИПЫ АНАЛИЗА ===

🧠 АНАЛИЗИРУЙ ПЕРЕД ГЕНЕРАЦИЕЙ:
1. Определи, какие показатели нужны для ответа на вопрос
2. Найди таблицы, содержащие эти показатели  
3. Проверь, есть ли временные колонки (year, month, date) в нужных таблицах
4. Выбери подходящую стратегию агрегации
5. Примени соответствующие техники SQL

🔸 СРАВНИТЕЛЬНЫЙ АНАЛИЗ ("сравни", "лучше", "хуже", "лидеры", "топ"):
ЦЕЛЬ: Ранжирование и сравнение между муниципалитетами/категориями
ТЕХНИКИ:
- GROUP BY region + агрегаты (AVG, COUNT, STDDEV, MIN, MAX)
- RANK() / ROW_NUMBER() для ранжирования  
- Процентили и отклонения от среднего
- LIMIT для топ-списков
ПРИМЕР ПАТТЕРНА:
SELECT 
    region,
    ROUND(AVG({основная_метрика}), 2) as avg_value,
    COUNT(*) as data_points,
    RANK() OVER (ORDER BY AVG({основная_метрика}) DESC) as ranking,
    ROUND(100.0 * AVG({основная_метрика}) / (SELECT AVG({основная_метрика}) FROM {таблица} WHERE {основная_метрика} IS NOT NULL), 1) as vs_average_pct
FROM {таблица}
WHERE {основная_метрика} IS NOT NULL
GROUP BY region
ORDER BY avg_value DESC
LIMIT {N}

🔸 ВРЕМЕННОЙ АНАЛИЗ ("динамика", "тренд", "изменение", "прогноз"):
ЦЕЛЬ: Анализ изменений во времени
ПРЕДВАРИТЕЛЬНАЯ ПРОВЕРКА: Убедись, что в таблице есть временные колонки!
ТЕХНИКИ:
- GROUP BY по временным периодам (year, month)
- LAG/LEAD для сравнения периодов
- Расчет темпов роста и изменений
- NULLIF() для защиты от деления на ноль
ПРИМЕР ПАТТЕРНА:
SELECT 
    year, month,
    ROUND(AVG({метрика}), 2) as period_avg,
    COUNT(*) as record_count,
    LAG(AVG({метрика})) OVER (ORDER BY year, month) as prev_period,
    ROUND(100.0 * (AVG({метрика}) - LAG(AVG({метрика})) OVER (ORDER BY year, month)) / 
          NULLIF(LAG(AVG({метрика})) OVER (ORDER BY year, month), 0), 2) as growth_rate_pct
FROM {таблица}
WHERE {метрика} IS NOT NULL
GROUP BY year, month
ORDER BY year, month
LIMIT 100

🔸 КОРРЕЛЯЦИОННЫЙ АНАЛИЗ ("связь", "зависимость", "влияние"):
ЦЕЛЬ: Выявление взаимосвязей между показателями
ТЕХНИКИ:
- JOIN таблиц через общие колонки (region)
- Агрегация связанных метрик
- Фильтрация достаточного объема данных (HAVING COUNT(*) >= N)
ПРИМЕР ПАТТЕРНА:
SELECT 
    t1.region,
    ROUND(AVG(t1.{метрика1}), 2) as avg_metric1,
    ROUND(AVG(t2.{метрика2}), 2) as avg_metric2,
    ROUND(AVG(t1.{метрика3}), 2) as avg_metric3,
    COUNT(*) as data_points
FROM {таблица1} t1
JOIN {таблица2} t2 ON t1.region = t2.region
WHERE t1.{метрика1} IS NOT NULL AND t2.{метрика2} IS NOT NULL
GROUP BY t1.region
HAVING COUNT(*) >= 3
ORDER BY avg_metric1 DESC
LIMIT 50

🔸 ОПИСАТЕЛЬНАЯ СТАТИСТИКА ("анализ", "обзор", "статистика"):
ЦЕЛЬ: Полная статистическая характеристика показателей
ТЕХНИКИ:
- Дескриптивная статистика (среднее, медиана, квартили)
- QUANTILE() для процентилей
- Объединение статистик нескольких показателей через UNION ALL
ПРИМЕР ПАТТЕРНА:
SELECT 
    '{название_метрики}' as metric,
    COUNT(*) as total_records,
    ROUND(AVG({метрика}), 2) as mean_val,
    ROUND(STDDEV({метрика}), 2) as std_dev,
    ROUND(MIN({метрика}), 2) as min_val,
    ROUND(QUANTILE({метрика}, 0.25), 2) as q1,
    ROUND(QUANTILE({метрика}, 0.5), 2) as median_val,
    ROUND(QUANTILE({метрика}, 0.75), 2) as q3,
    ROUND(MAX({метрика}), 2) as max_val
FROM {таблица}
WHERE {метрика} IS NOT NULL

🔸 ГЕОГРАФИЧЕСКИЙ АНАЛИЗ ("где", "муниципалитет", "распределение"):
ЦЕЛЬ: Пространственное распределение показателей
ТЕХНИКИ:
- Фокус на region как ключевом измерении
- Категоризация через CASE WHEN
- Ранжирование территорий
ПРИМЕР ПАТТЕРНА:
SELECT 
    region,
    ROUND({метрика}, 2) as value,
    RANK() OVER (ORDER BY {метрика} DESC) as regional_rank,
    CASE 
        WHEN {метрика} > (SELECT AVG({метрика}) * 1.2 FROM {таблица} WHERE {метрика} IS NOT NULL) THEN 'Высокий'
        WHEN {метрика} > (SELECT AVG({метрика}) * 0.8 FROM {таблица} WHERE {метрика} IS NOT NULL) THEN 'Средний'
        ELSE 'Низкий'
    END as performance_level
FROM {таблица}
WHERE {метрика} IS NOT NULL
ORDER BY {метрика} DESC
LIMIT 30

=== АДАПТИВНАЯ ЛОГИКА ВЫБОРА ===

🎯 АЛГОРИТМ ВЫБОРА ТАБЛИЦЫ И СТРАТЕГИИ:
1. Определи ключевые термины в вопросе (расходы→consumer_spending, транспорт→transport_accessibility/transport_score, демография→population)
2. Найди таблицу с нужными метриками
3. Проверь наличие временных данных для трендового анализа
4. Выбери соответствующий паттерн запроса
5. Адаптируй под специфику данных

ПРАВИЛА ДЛЯ ТРАНСПОРТНЫХ ДАННЫХ:
- Для вопросов о динамике/трендах транспорта → используй transport_accessibility из region_spending (есть временные данные)
- Для вопросов о сравнении муниципалитетов по транспорту → используй transport_score из transport_data
- Для комплексного анализа → объедини обе таблицы через JOIN по region

🔄 FALLBACK СТРАТЕГИИ:
- Если нет временных данных для тренда → переключись на сравнительный анализ
- Если мало данных для агрегации → используй простую выборку с ORDER BY + LIMIT
- Если запрашиваемая колонка отсутствует → выбери похожую из доступных
- Если JOIN невозможен → используй отдельные запросы

=== УНИВЕРСАЛЬНЫЕ ПРАВИЛА ===
1. ВСЕГДА начинай с анализа того, что реально доступно в схеме
2. Ограничивай результат: LIMIT 20-500 строк в зависимости от типа анализа  
3. Фильтруй NULL: WHERE column IS NOT NULL для числовых показателей
4. Используй ROUND(value, 2) для читаемости чисел
5. Добавляй ORDER BY для воспроизводимости
6. Защищайся от деления на ноль: NULLIF(divisor, 0)
7. Применяй понятные алиасы: avg_spending вместо AVG_spending
8. Для квартилей используй QUANTILE(column, 0.25) (DuckDB)

=== АНТИ-ПАТТЕРНЫ ===
❌ SELECT * FROM table -- неэффективно для больших данных
❌ Запросы без LIMIT -- потенциально огромные результаты  
❌ Использование несуществующих колонок без проверки
❌ Агрегация без GROUP BY когда она нужна
❌ Отсутствие обработки NULL значений
❌ Деление без защиты от нуля
❌ Неинформативные имена колонок в результате

Возвращай ТОЛЬКО SQL-запрос, адаптированный к конкретному вопросу и схеме данных.
"""

VISUALIZATION_TOOL_PROMPT = """
Определи оптимальный тип визуализации для данных:

Правила выбора:
- line: Временные ряды, динамика показателей
- bar: Сравнение значений между муниципалитетами/категориями  
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