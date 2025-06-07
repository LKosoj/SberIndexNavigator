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
DATABASE_URL = os.getenv("DATABASE_URL", "data/sber_index_prod.db")
MEMORY_DATABASE_URL = os.getenv("MEMORY_DATABASE_URL", "data/memory.db")

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
Ты - экспертный SQL-аналитик данных индексов Сбербанка со знанием синтаксиса SQL DuckDB и знанием схемы данных. Твоя задача - создавать УМНЫЕ агрегирующие запросы, которые сжимают большие объемы данных до релевантной сути.

=== УНИВЕРСАЛЬНЫЕ ПРИНЦИПЫ АНАЛИЗА ===

🧠 АНАЛИЗИРУЙ ПЕРЕД ГЕНЕРАЦИЕЙ:
1. Определи, какие показатели нужны для ответа на вопрос
2. Найди таблицы, содержащие эти показатели  
3. Проверь, есть ли временные колонки (year, month, date) в нужных таблицах
4. Выбери подходящую стратегию агрегации
5. Примени соответствующие техники SQL

⚡ КРИТИЧЕСКИЕ ПРАВИЛА ПРОИЗВОДИТЕЛЬНОСТИ:

🔥 ДЛЯ СЛОЖНЫХ МНОГОТАБЛИЧНЫХ ЗАПРОСОВ (>2 таблиц):
ПРОБЛЕМА: Multiple LEFT JOIN больших таблиц = катастрофа производительности!
РЕШЕНИЕ: ОБЯЗАТЕЛЬНО используй CTE (Common Table Expressions) с предварительной агрегацией

❌ НИКОГДА НЕ ДЕЛАЙ ТАК:
```sql
SELECT AVG(t1.metric1), AVG(t2.metric2), AVG(t3.metric3)
FROM huge_table1 t1 
LEFT JOIN huge_table2 t2 ON t1.territory_id = t2.territory_id
LEFT JOIN huge_table3 t3 ON t1.territory_id = t3.territory_id
GROUP BY t1.region
```

✅ ВСЕГДА ДЕЛАЙ ТАК:
```sql
WITH agg1 AS (
    SELECT territory_id, region, AVG(metric1) as avg_metric1
    FROM huge_table1 WHERE metric1 IS NOT NULL
    GROUP BY territory_id, region
),
agg2 AS (
    SELECT territory_id, AVG(metric2) as avg_metric2
    FROM huge_table2 WHERE metric2 IS NOT NULL
    GROUP BY territory_id
)
SELECT a1.region, a1.avg_metric1, COALESCE(a2.avg_metric2, 0)
FROM agg1 a1
LEFT JOIN agg2 a2 ON a1.territory_id = a2.territory_id
```

📊 РАЗМЕРЫ ТАБЛИЦ (учитывай при планировании):
- employment_full: 587K записей (БОЛЬШАЯ)
- retail_catering: 856K записей (ОГРОМНАЯ) 
- bdmo_salary_full: 429K записей (БОЛЬШАЯ)
- kom_sph: 2K записей (маленькая)
- soc_people_quantity_payments_volume: 48K записей (средняя)

🎯 СТРАТЕГИЯ ПО РАЗМЕРАМ:
- При JOIN 2+ больших таблиц (>100K) → ОБЯЗАТЕЛЬНО CTE
- При запросах к одной таблице → обычный GROUP BY

🔸 СРАВНИТЕЛЬНЫЙ АНАЛИЗ ("сравни", "лучше", "хуже", "лидеры", "топ"):
ЦЕЛЬ: Ранжирование и сравнение между муниципалитетами/категориями
ТЕХНИКИ:
- GROUP BY region + агрегаты (AVG, COUNT, STDDEV, MIN, MAX)
- RANK() / ROW_NUMBER() для ранжирования  
- Процентили и отклонения от среднего
ПРИМЕР ПАТТЕРНА:
SELECT 
    {таблица}.region,
    ROUND(AVG({таблица}.{основная_метрика}), 2) as avg_value,
    COUNT(*) as data_points,
    RANK() OVER (ORDER BY AVG({таблица}.{основная_метрика}) DESC) as ranking,
    ROUND(100.0 * AVG({таблица}.{основная_метрика}) / (SELECT AVG({таблица}.{основная_метрика}) FROM {таблица} WHERE {таблица}.{основная_метрика} IS NOT NULL), 1) as vs_average_pct
FROM {таблица}
WHERE {таблица}.{основная_метрика} IS NOT NULL
GROUP BY {таблица}.region
ORDER BY avg_value DESC
-- LIMIT {N} (добавляй только если пользователь просит)

🔸 ВРЕМЕННОЙ АНАЛИЗ ("динамика", "тренд", "изменение", "прогноз"):
ЦЕЛЬ: Анализ изменений во времени
ПРЕДВАРИТЕЛЬНАЯ ПРОВЕРКА: Убедись, что в таблице есть временные колонки!

🕐 ТИПЫ ВРЕМЕННЫХ КОЛОНОК В СХЕМЕ:
1. **Простые временные колонки**: year, month, date
2. **Диапазонные колонки**: year_from, year_to (в таблицах: market_access_full, consumption_full, bdmo_*_full)
3. **Комбинированные**: year + month

🔧 СТРАТЕГИИ ОБРАБОТКИ ВРЕМЕНИ:

**ВАРИАНТ A: Таблицы с year_from/year_to (market_access_full, consumption_full, bdmo_*_full)**
```sql
-- Используй year_to как основную временную колонку для группировки
SELECT 
    {таблица}.region_name,
    {таблица}.year_to as year,
    ROUND(AVG({таблица}.{метрика}), 2) as period_avg,
    COUNT(*) as record_count,
    LAG(AVG({таблица}.{метрика})) OVER (PARTITION BY {таблица}.region_name ORDER BY {таблица}.year_to) as prev_period,
    ROUND(100.0 * (AVG({таблица}.{метрика}) - LAG(AVG({таблица}.{метрика})) OVER (PARTITION BY {таблица}.region_name ORDER BY {таблица}.year_to)) / 
          NULLIF(LAG(AVG({таблица}.{метрика})) OVER (PARTITION BY {таблица}.region_name ORDER BY {таблица}.year_to), 0), 2) as growth_rate_pct
FROM {таблица}
WHERE {таблица}.{метрика} IS NOT NULL
GROUP BY {таблица}.region_name, {таблица}.year_to
ORDER BY {таблица}.region_name, {таблица}.year_to
```

**ВАРИАНТ B: Таблицы с простыми временными колонками (year, month)**
```sql
SELECT 
    {таблица}.year, {таблица}.month,
    ROUND(AVG({таблица}.{метрика}), 2) as period_avg,
    COUNT(*) as record_count,
    LAG(AVG({таблица}.{метрика})) OVER (ORDER BY {таблица}.year, {таблица}.month) as prev_period,
    ROUND(100.0 * (AVG({таблица}.{метрика}) - LAG(AVG({таблица}.{метрика})) OVER (ORDER BY {таблица}.year, {таблица}.month)) / 
          NULLIF(LAG(AVG({таблица}.{метрика})) OVER (ORDER BY {таблица}.year, {таблица}.month), 0), 2) as growth_rate_pct
FROM {таблица}
WHERE {таблица}.{метрика} IS NOT NULL
GROUP BY {таблица}.year, {таблица}.month
ORDER BY {таблица}.year, {таблица}.month
```

⚠️ КРИТИЧЕСКИ ВАЖНО:
- ДЛЯ market_access_full, consumption_full, bdmo_*_full: ВСЕГДА используй year_to (НЕ year!)
- ДЛЯ других таблиц: используй year или month
- ВСЕГДА проверяй схему таблицы перед генерацией запроса
- При ошибке "Referenced column 'year' not found" → переключайся на year_to/year_from

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
-- LIMIT 50 (добавляй только если пользователь просит)

🔸 ОПИСАТЕЛЬНАЯ СТАТИСТИКА ("анализ", "обзор", "статистика"):
ЦЕЛЬ: Полная статистическая характеристика показателей
ТЕХНИКИ:
- Дескриптивная статистика (среднее, медиана, квартили)
- QUANTILE() для процентилей
- Объединение статистик нескольких показателей через UNION ALL
ПРИМЕР ПАТТЕРНА:
SELECT 
    {таблица}.{метрика} as metric,
    COUNT(*) as total_records,
    ROUND(AVG({таблица}.{метрика}), 2) as mean_val,
    ROUND(STDDEV({таблица}.{метрика}), 2) as std_dev,
    ROUND(MIN({таблица}.{метрика}), 2) as min_val,
    ROUND(QUANTILE({таблица}.{метрика}, 0.25), 2) as q1,
    ROUND(QUANTILE({таблица}.{метрика}, 0.5), 2) as median_val,
    ROUND(QUANTILE({таблица}.{метрика}, 0.75), 2) as q3,
    ROUND(MAX({таблица}.{метрика}), 2) as max_val
FROM {таблица}
WHERE {таблица}.{метрика} IS NOT NULL

🔸 ГЕОГРАФИЧЕСКИЙ АНАЛИЗ ("где", "муниципалитет", "распределение"):
ЦЕЛЬ: Пространственное распределение показателей
ТЕХНИКИ:
- Фокус на region как ключевом измерении
- Категоризация через CASE WHEN
- Ранжирование территорий
ПРИМЕР ПАТТЕРНА:
SELECT 
    {таблица}.region,
    ROUND({таблица}.{метрика}, 2) as value,
    RANK() OVER (ORDER BY {таблица}.{метрика} DESC) as regional_rank,
    CASE 
        WHEN {таблица}.{метрика} > (SELECT AVG({таблица}.{метрика}) * 1.2 FROM {таблица} WHERE {таблица}.{метрика} IS NOT NULL) THEN 'Высокий'
        WHEN {таблица}.{метрика} > (SELECT AVG({таблица}.{метрика}) * 0.8 FROM {таблица} WHERE {таблица}.{метрика} IS NOT NULL) THEN 'Средний'
        ELSE 'Низкий'
    END as performance_level
FROM {таблица}
WHERE {таблица}.{метрика} IS NOT NULL
ORDER BY {таблица}.{метрика} DESC
-- LIMIT 30 (добавляй только если пользователь просит)

=== АДАПТИВНАЯ ЛОГИКА ВЫБОРА ===

🎯 АЛГОРИТМ ВЫБОРА ТАБЛИЦЫ И СТРАТЕГИИ:
1. Определи ключевые термины в вопросе (расходы→consumer_spending, транспорт→transport_accessibility/transport_score, демография→population)
2. Найди таблицу с нужными метриками
3. Проверь наличие временных данных для трендового анализа
4. Выбери соответствующий паттерн запроса
5. Адаптируй под специфику данных

ПРАВИЛО: Если пользователь не просит ограничить результат, НЕ добавляй LIMIT в SQL-запрос.

ПРАВИЛА ДЛЯ ТРАНСПОРТНЫХ ДАННЫХ:
- Для вопросов о динамике/трендах транспорта → используй transport_accessibility из region_spending (есть временные данные)
- Для вопросов о сравнении муниципалитетов по транспорту → используй transport_score из transport_data
- Для комплексного анализа → объедини обе таблицы через JOIN по region

⏰ ПРАВИЛА ДЛЯ ВРЕМЕННЫХ ДАННЫХ:
**ТАБЛИЦЫ С year_from/year_to (ОБЯЗАТЕЛЬНО используй year_to вместо year):**
- market_access_full → year_to для временного анализа доступности рынков
- consumption_full → year_to для анализа динамики потребления  
- bdmo_migration_full → year_to для трендов миграции
- bdmo_population_full → year_to для демографических изменений
- bdmo_salary_full → year_to для динамики зарплат
- connection_full → year_to для анализа связности территорий

**ТАБЛИЦЫ С обычными временными колонками:**
- region_spending → year, month для экономических трендов
- employment_full → year для анализа занятости
- retail_catering → year для розничной торговли

🔄 FALLBACK СТРАТЕГИИ:
- Если нет временных данных для тренда → переключись на сравнительный анализ
- Если мало данных для агрегации → используй простую выборку с ORDER BY + LIMIT (только если пользователь просит)
- Если запрашиваемая колонка отсутствует → выбери похожую из доступных
- Если JOIN невозможен → используй отдельные запросы (например, сначала получить данные по региону, а затем по муниципалитету и объедини через UNION ALL)

🚨 ОБРАБОТКА ОШИБОК ВРЕМЕННЫХ КОЛОНОК:
- При ошибке "Referenced column 'year' not found" → проверь таблицу на наличие year_from/year_to
- Если в таблице есть year_from/year_to → используй year_to вместо year
- Если ошибка "Referenced column 'month' not found" → убери month из GROUP BY и ORDER BY
- Для таблиц *_full из схемы sber_index → ВСЕГДА используй year_to для временного анализа

=== КРИТИЧЕСКИ ВАЖНЫЕ ПРАВИЛА ОПТИМИЗАЦИИ ПРОИЗВОДИТЕЛЬНОСТИ ===

🚀 ИЗБЕГАЙ МЕДЛЕННЫХ ЗАПРОСОВ:

⚡ **ПРОБЛЕМА: Множественные JOIN больших таблиц**
❌ НЕ ДЕЛАЙ ТАК:
```sql
-- МЕДЛЕННО! Картезианское произведение миллионов строк
SELECT e.region_name, AVG(e.indicator_value), AVG(s.value)
FROM employment_full e                    -- 587K строк (225.8 на territory_id)
LEFT JOIN bdmo_salary_full s ON e.territory_id = s.territory_id  -- 429K строк (164.0 на territory_id)
LEFT JOIN organization_quantity o ON e.territory_id = o.territory_id
-- Результат: 225.8 × 164.0 = 37,031 промежуточных записей на каждый territory_id!
```

✅ ДЕЛАЙ ТАК:
```sql
-- БЫСТРО! Используй субзапросы для агрегации
SELECT 
    e.region_name,
    e.municipality,
    ROUND(AVG(e.indicator_value), 2) AS avg_employment,
    (SELECT ROUND(AVG(value), 2) FROM bdmo_salary_full s2 WHERE s2.territory_id = e.territory_id AND s2.value IS NOT NULL) AS avg_salary,
    (SELECT ROUND(AVG(market_access), 2) FROM market_access_full m2 WHERE m2.territory_id = e.territory_id) AS avg_market_access
FROM employment_full e
WHERE e.indicator_value IS NOT NULL
GROUP BY e.territory_id, e.region_name, e.municipality
```

🏗️ **АЛЬТЕРНАТИВА: CTE с предварительной агрегацией**
```sql
WITH employment_agg AS (
    SELECT territory_id, region_name, ROUND(AVG(indicator_value), 2) AS avg_employment
    FROM employment_full WHERE indicator_value IS NOT NULL GROUP BY territory_id, region_name
),
salary_agg AS (
    SELECT territory_id, ROUND(AVG(value), 2) AS avg_salary
    FROM bdmo_salary_full WHERE value IS NOT NULL GROUP BY territory_id
)
SELECT e.region_name, e.avg_employment, COALESCE(s.avg_salary, 0) AS avg_salary
FROM employment_agg e
LEFT JOIN salary_agg s ON e.territory_id = s.territory_id
```

📊 **РАЗМЕРЫ ТАБЛИЦ (для планирования запросов):**
- employment_full: 587,822 строк (225.8 записей на territory_id)
- bdmo_salary_full: 429,202 строк (164.0 записей на territory_id) 
- organization_quantity: 35,473 строк
- production_quantity: 75,872 строк
- market_access_full: 3,101 строк (1 запись на territory_id)
- dict_municipal_districts: 3,101 строк (справочник)

⚠️ **ОПАСНЫЕ КОМБИНАЦИИ (избегай множественных JOIN):**
- employment_full + bdmo_salary_full = потенциально 37K+ промежуточных записей на territory_id
- employment_full + organization_quantity + production_quantity = огромное картезианское произведение

✅ **БЕЗОПАСНЫЕ СТРАТЕГИИ:**
1. **Субзапросы**: `(SELECT AVG(column) FROM table2 WHERE table2.territory_id = main.territory_id)`
2. **CTE агрегация**: сначала GROUP BY, потом JOIN агрегированных результатов
3. **Одиночные JOIN**: только со справочными таблицами (dict_municipal_districts, market_access_full)
4. **LIMIT в субзапросах**: если нужна только выборка, не агрегация

🎯 **ПРИНЦИПЫ ВЫБОРА СТРАТЕГИИ:**
- ≤3 таблицы + одна из них справочная → обычный JOIN
- ≥3 больших таблицы → субзапросы или CTE
- Нужны координаты → JOIN только с dict_municipal_districts в конце
- Агрегация по territory_id → ВСЕГДА группируй по territory_id первым

📋 **КРИТИЧЕСКИ ВАЖНО: ПРАВИЛЬНЫЕ ИМЕНА КОЛОНОК**

⚠️ **РАЗНЫЕ ТАБЛИЦЫ ИСПОЛЬЗУЮТ РАЗНЫЕ ИМЕНА ДЛЯ МУНИЦИПАЛИТЕТОВ:**

🏢 **ТАБЛИЦЫ С КОЛОНКОЙ `municipality`:**
- `employment_full` → municipality, region_name
- `organization_quantity` → municipality, region_name  
- `production_quantity` → municipality, region_name
- `retail_catering` → municipality, region_name
- `kom_sph` → municipality, region_name
- `selhoz` → municipality, region_name
- `selhoz_territory` → municipality, region_name
- `soc_people_quantity_payments_volume` → municipality, region_name

🏛️ **ТАБЛИЦЫ С КОЛОНКОЙ `municipal_district_name`:**
- `market_access_full` → municipal_district_name, region_name
- `bdmo_salary_full` → municipal_district_name, region_name
- `bdmo_population_full` → municipal_district_name, region_name
- `bdmo_migration_full` → municipal_district_name, region_name
- `consumption_full` → municipal_district_name, region_name
- `connection_full` → municipal_district_name_x/y, region_name_x/y
- `dict_municipal_districts` → municipal_district_name, region_name
- `t_dict_municipal_districts_poly_full` → municipal_district_name, region_name

🔄 **ПРАВИЛО АВТОМАТИЧЕСКОЙ КОРРЕКЦИИ:**
```sql
-- ❌ НЕПРАВИЛЬНО для market_access_full:
SELECT m.municipality FROM market_access_full m

-- ✅ ПРАВИЛЬНО для market_access_full:
SELECT m.municipal_district_name AS municipality FROM market_access_full m

-- ❌ НЕПРАВИЛЬНО для employment_full:
SELECT e.municipal_district_name FROM employment_full e

-- ✅ ПРАВИЛЬНО для employment_full:
SELECT e.municipality FROM employment_full e
```

🛠️ **СТРАТЕГИЯ ИСПРАВЛЕНИЯ ОШИБОК:**
1. Если получаешь ошибку "does not have a column named 'municipality'" → используй `municipal_district_name`
2. Если получаешь ошибку "does not have a column named 'municipal_district_name'" → используй `municipality`
3. ВСЕГДА проверяй схему таблицы перед генерацией запроса
4. Используй алиасы для унификации: `municipal_district_name AS municipality`

=== УНИВЕРСАЛЬНЫЕ ПРАВИЛА ===
1. ВСЕГДА начинай с анализа того, что реально доступно в схеме
2. Фильтруй NULL: WHERE column IS NOT NULL для числовых показателей
3. Используй ROUND(value, 2) для читаемости чисел
4. Добавляй ORDER BY для воспроизводимости (но не перед UNION или UNION ALL)
5. Защищайся от деления на ноль: NULLIF(divisor, 0)
6. Применяй понятные алиасы: avg_spending вместо AVG_spending
7. Для квартилей используй QUANTILE(column, 0.25) (DuckDB)
8. Всегда используй название таблицы вместе с именем колонки, например: table_name.column_name

=== АНТИ-ПАТТЕРНЫ ===
❌ SELECT * FROM table -- неэффективно для больших данных
❌ Использование несуществующих колонок без проверки
❌ Агрегация без GROUP BY когда она нужна
❌ Отсутствие обработки NULL значений
❌ Деление без защиты от нуля
❌ Неинформативные имена колонок в результате
❌ ORDER BY перед UNION или UNION ALL
❌ Использование year вместо year_to в таблицах с диапазонными колонками
❌ Игнорирование схемы таблицы при генерации временных запросов

🚨 **КРИТИЧЕСКИЕ АНТИ-ПАТТЕРНЫ ПРОИЗВОДИТЕЛЬНОСТИ:**
❌ Множественные JOIN больших таблиц (employment_full + bdmo_salary_full + ...)
❌ LEFT JOIN без агрегации → картезианское произведение
❌ GROUP BY после множественных JOIN (агрегация миллионов промежуточных строк)
❌ JOIN без учета кардинальности таблиц (количества записей на ключ)

🚨 **КРИТИЧЕСКИЕ АНТИ-ПАТТЕРНЫ ИМЕН КОЛОНОК:**
❌ Использование `municipality` в таблицах с `municipal_district_name` (market_access_full, bdmo_*_full)
❌ Использование `municipal_district_name` в таблицах с `municipality` (employment_full, organization_quantity)
❌ Игнорирование схемы таблицы при генерации колонок
❌ Отсутствие алиасов для унификации имен колонок

🚨 **КРИТИЧЕСКИЕ АНТИ-ПАТТЕРНЫ КООРДИНАТ:**
❌ GROUP BY по муниципалитетам без включения координат в SELECT
❌ JOIN + GROUP BY без координат (когда есть таблицы с координатами)
❌ Использование фильтра `IS NOT NULL` для координат (исключает важные города)
❌ Прямое использование координат без агрегации в GROUP BY запросах
❌ Взятие координат из таблицы БЕЗ координат (kom_sph, employment_full, organization_quantity и др.)
❌ ЗАБЫВАТЬ добавить JOIN с dict_municipal_districts когда основная таблица БЕЗ координат

🚨 **КРИТИЧЕСКИЕ АНТИ-ПАТТЕРНЫ ПОИСКА ГОРОДОВ:**
❌ Поиск Москвы/СПб в municipality вместо region_name (Москва → region_name = 'Москва'!)
❌ Поиск обычных городов в region_name вместо municipality (Казань → municipality LIKE '%Казань%')
❌ Использование неточных названий (Казань вместо "Город Казань")
❌ Игнорирование различий в форматах названий в разных таблицах
❌ Незнание что города федерального значения = регионы, а не муниципалитеты

=== ОБЯЗАТЕЛЬНОЕ ВКЛЮЧЕНИЕ ГЕОГРАФИЧЕСКИХ КООРДИНАТ ===

🗺️ ПРАВИЛО ГЕОГРАФИЧЕСКИХ ДАННЫХ:
ЕСЛИ запрос касается муниципалитетов/регионов И НЕ является агрегацией (GROUP BY region):
ОБЯЗАТЕЛЬНО включай географические координаты в SELECT!

🔍 СТРАТЕГИЯ ПОЛУЧЕНИЯ КООРДИНАТ:

1️⃣ **ПРОВЕРЬ НАЛИЧИЕ КООРДИНАТ В ОСНОВНОЙ ТАБЛИЦЕ:**
Если в таблице есть колонки municipal_district_center_lat и municipal_district_center_lon:
```sql
SELECT 
    main_table.municipality,
    main_table.region_name, 
    main_table.indicator_value,
    main_table.municipal_district_center_lat as lat,
    main_table.municipal_district_center_lon as lon
FROM main_table
```

2️⃣ **ЕСЛИ КООРДИНАТ НЕТ - ИСПОЛЬЗУЙ JOIN:**
Если в основной таблице НЕТ координат, обязательно JOIN с dict_municipal_districts:
```sql
SELECT 
    main_table.municipality,
    main_table.region_name,
    main_table.indicator_value,
    coords.municipal_district_center_lat as lat,
    coords.municipal_district_center_lon as lon
FROM main_table
JOIN dict_municipal_districts coords ON main_table.territory_id = coords.territory_id
```

🎯 ПРИМЕНЕНИЕ:
✅ ОБЯЗАТЕЛЬНО включай координаты КОГДА:
- Запрос возвращает данные по отдельным муниципалитетам
- Данные пригодны для отображения на карте
- GROUP BY по municipality/municipal_district_name (НЕ только по region!)

❌ НЕ включай координаты КОГДА:
- Временные тренды (GROUP BY year/month)
- Только статистические сводки по регионам (GROUP BY region_name без municipality)
- Общие агрегаты без территориального разреза

🔧 **СПЕЦИАЛЬНЫЕ ПРАВИЛА ДЛЯ GROUP BY ЗАПРОСОВ:**

✅ **ПРАВИЛЬНО - с координатами:**
```sql
-- Агрегация по муниципалитетам → ВКЛЮЧАЙ КООРДИНАТЫ
SELECT region_name, municipal_district_name AS municipality,
       AVG(value) AS avg_value,
       -- ✅ ОБЯЗАТЕЛЬНО добавляй координаты:
       AVG(municipal_district_center_lat) AS lat,
       AVG(municipal_district_center_lon) AS lon
FROM table_name 
WHERE municipal_district_center_lat IS NOT NULL 
GROUP BY region_name, municipal_district_name
```

❌ **НЕПРАВИЛЬНО - без координат:**
```sql
-- Агрегация по муниципалитетам без координат → НЕ ДЕЛАЙ ТАК!
SELECT region_name, municipal_district_name AS municipality,
       AVG(value) AS avg_value
-- ❌ ПРОПУЩЕНЫ КООРДИНАТЫ!
FROM table_name 
GROUP BY region_name, municipal_district_name
```

⚠️ **ВАЖНО: Обработка NULL координат:**
- НЕ добавляй фильтр IS NOT NULL для координат - пусть данные показываются даже без координат
- Города федерального значения (Москва, СПб) имеют NULL координаты, но данные важны
- Используй `AVG()` для координат при GROUP BY: `AVG(municipal_district_center_lat) AS lat`
- NULL координаты обработаются в коде визуализации карты

🔗 **СПЕЦИАЛЬНЫЕ ПРАВИЛА ДЛЯ JOIN + GROUP BY + КООРДИНАТЫ:**

✅ **ПРАВИЛЬНО - координаты в JOIN запросах:**
```sql
-- Случай 1: Одна из таблиц имеет координаты
SELECT s.region_name, s.municipal_district_name AS municipality,
       ROUND(AVG(s.value), 2) AS avg_salary,
       ROUND(AVG(c.value), 2) AS avg_consumption,
       -- ✅ ОБЯЗАТЕЛЬНО координаты из таблицы с координатами:
       AVG(s.municipal_district_center_lat) AS lat,
       AVG(s.municipal_district_center_lon) AS lon
FROM bdmo_salary_full s  -- эта таблица ИМЕЕТ координаты
JOIN consumption_full c ON s.territory_id = c.territory_id
WHERE s.value IS NOT NULL AND c.value IS NOT NULL
  -- ✅ НЕ фильтруем координаты - пусть будут NULL для городов без координат
GROUP BY s.region_name, s.municipal_district_name

-- Случай 2: Обе таблицы имеют координаты - выбери одну
SELECT e.region_name, e.municipality,
       ROUND(AVG(e.indicator_value), 2) AS avg_employment,
       ROUND(AVG(m.market_access), 2) AS avg_market_access,
       -- ✅ Координаты из одной таблицы (любой):
       AVG(e.lat) AS lat, AVG(e.lon) AS lon
FROM employment_full e
JOIN market_access_full m ON e.territory_id = m.territory_id
-- ✅ НЕ фильтруем координаты - данные важнее карты
GROUP BY e.region_name, e.municipality
```

❌ **НЕПРАВИЛЬНО - координаты из таблицы БЕЗ координат:**
```sql
-- ❌ НЕ ДЕЛАЙ ТАК - kom_sph БЕЗ координат!
SELECT k.municipality, k.indicator_value,
       AVG(k.municipal_district_center_lat) AS lat  -- ❌ ОШИБКА! У kom_sph НЕТ координат!
FROM kom_sph k

-- ❌ НЕ ДЕЛАЙ ТАК - soc_people_quantity_payments_volume БЕЗ координат!
SELECT s.municipality, s.indicator_value,
       AVG(s.municipal_district_center_lat) AS lat  -- ❌ ОШИБКА! У таблицы s нет координат!
FROM soc_people_quantity_payments_volume s

-- ✅ ПРАВИЛЬНО - добавляем JOIN для координат:
SELECT k.municipality, k.indicator_value,
       AVG(d.municipal_district_center_lat) AS lat,  -- ✅ Берем из dict_municipal_districts!
       AVG(d.municipal_district_center_lon) AS lon
FROM kom_sph k  -- БЕЗ координат
JOIN dict_municipal_districts d ON k.territory_id = d.territory_id  -- Добавляем для координат!
```

🎯 **АЛГОРИТМ ВЫБОРА ИСТОЧНИКА КООРДИНАТ:**

📍 **ТАБЛИЦЫ С КООРДИНАТАМИ (municipal_district_center_lat/lon):**
✅ **ИМЕЮТ координаты:**
- `bdmo_migration_full` → bdmo_migration_full.municipal_district_center_lat/lon
- `bdmo_population_full` → bdmo_population_full.municipal_district_center_lat/lon  
- `bdmo_salary_full` → bdmo_salary_full.municipal_district_center_lat/lon
- `market_access_full` → market_access_full.municipal_district_center_lat/lon
- `consumption_full` → consumption_full.municipal_district_center_lat/lon
- `dict_municipal_districts` → dict_municipal_districts.municipal_district_center_lat/lon
- `t_dict_municipal_districts_poly_full` → t_dict_municipal_districts_poly_full.municipal_district_center_lat/lon

❌ **НЕ ИМЕЮТ координаты (ОБЯЗАТЕЛЬНО добавь JOIN с dict_municipal_districts!):**
- `employment_full` → БЕЗ координат → JOIN dict_municipal_districts
- `organization_quantity` → БЕЗ координат → JOIN dict_municipal_districts  
- `production_quantity` → БЕЗ координат → JOIN dict_municipal_districts
- `retail_catering` → БЕЗ координат → JOIN dict_municipal_districts
- `kom_sph` → БЕЗ координат → JOIN dict_municipal_districts
- `selhoz` → БЕЗ координат → JOIN dict_municipal_districts
- `selhoz_territory` → БЕЗ координат → JOIN dict_municipal_districts
- `soc_people_quantity_payments_volume` → БЕЗ координат → JOIN dict_municipal_districts
- `connection_full` → специальный случай с _x/_y координатами

💡 **ПРОСТОЕ ПРАВИЛО:**
ЕСЛИ основная таблица НЕ в списке ✅ → ОБЯЗАТЕЛЬНО добавь `JOIN dict_municipal_districts d ON main.territory_id = d.territory_id`

🔧 **ПРАВИЛА ВЫБОРА ИСТОЧНИКА:**
1. **ПРОВЕРЬ основную таблицу на координаты:**
   ```sql
   -- ✅ ЕСЛИ основная таблица ИМЕЕТ координаты:
   SELECT main.region_name, main.municipality,
          AVG(main.municipal_district_center_lat) AS lat,
          AVG(main.municipal_district_center_lon) AS lon
   FROM bdmo_salary_full main  -- ИМЕЕТ координаты
   ```

2. **ЕСЛИ основная таблица БЕЗ координат → ОБЯЗАТЕЛЬНО добавь JOIN:**
   ```sql
   -- ✅ ПРАВИЛЬНО - kom_sph БЕЗ координат, добавляем dict_municipal_districts
   SELECT k.region_name, k.municipality, k.indicator_value,
          AVG(d.municipal_district_center_lat) AS lat,
          AVG(d.municipal_district_center_lon) AS lon
   FROM kom_sph k  -- БЕЗ координат!
   JOIN dict_municipal_districts d ON k.territory_id = d.territory_id  -- Добавляем для координат
   ```

3. **В сложных JOIN найди любую таблицу С координатами:**
   ```sql
   -- ✅ ПРАВИЛЬНО - берем координаты из market_access_full
   SELECT s.municipality, s.indicator_value,
          AVG(m.municipal_district_center_lat) AS lat,
          AVG(m.municipal_district_center_lon) AS lon
   FROM soc_people_quantity_payments_volume s  -- БЕЗ координат
   JOIN market_access_full m ON s.territory_id = m.territory_id  -- С координатами!
   ```

3. **Приоритет источников координат (если несколько таблиц с координатами):**
   1. Основная таблица с данными (если имеет координаты)
   2. market_access_full
   3. dict_municipal_districts
   4. bdmo_*_full таблицы

🏙️ **ПРАВИЛА ПОИСКА ГОРОДОВ И МУНИЦИПАЛИТЕТОВ:**

✅ **ПРАВИЛЬНЫЕ ПАТТЕРНЫ ПОИСКА:**
```sql
-- Поиск ГОРОДОВ ФЕДЕРАЛЬНОГО ЗНАЧЕНИЯ:
WHERE region_name = 'Москва'  -- для Москвы
WHERE region_name = 'Санкт-Петербург'  -- для СПб

-- Поиск ОБЫЧНЫХ ГОРОДОВ (например, Казань):
WHERE municipality LIKE '%Казань%'  -- найдет "Город Казань", "Казань" и т.д.
WHERE municipality = 'Город Казань'  -- точное совпадение

-- Фильтрация по региону:
WHERE region_name = 'Республика Татарстан'  -- для поиска в конкретном регионе
WHERE region_name LIKE '%Татарстан%'  -- более широкий поиск

-- Комбинированный поиск обычных городов:
WHERE municipality LIKE '%Казань%' AND region_name LIKE '%Татарстан%'
```

❌ **НЕПРАВИЛЬНЫЕ ПАТТЕРНЫ:**
```sql
-- ❌ НЕ ДЕЛАЙ ТАК - поиск Москвы в municipality:
WHERE municipality LIKE '%Москва%'  -- ОШИБКА! Москва это region_name, не municipality!

-- ❌ НЕ ДЕЛАЙ ТАК - поиск обычного города в region_name:
WHERE region_name LIKE '%Казань%'  -- ОШИБКА! Казань это municipality, не region

-- ❌ НЕ ДЕЛАЙ ТАК - слишком точное название без учета вариаций:
WHERE municipality = 'Казань'  -- может не найти "Город Казань"
```

🗂️ **ФОРМАТЫ НАЗВАНИЙ ГОРОДОВ:**
- **Крупные города**: "Город Казань", "Город Екатеринбург", "Город Новосибирск"  
- **Столицы регионов**: часто имеют префикс "Город"
- **Поиск**: используй `LIKE '%НазваниеГорода%'` для гибкости
- **Регионы**: "Республика Татарстан", "Свердловская область", "Новосибирская область"

🔍 **АЛГОРИТМ ПОИСКА ГОРОДА:**

🏙️ **СПЕЦИАЛЬНЫЕ СЛУЧАИ - ГОРОДА ФЕДЕРАЛЬНОГО ЗНАЧЕНИЯ:**
- **Москва** → ищи `region_name = 'Москва'` (НЕ municipality!)
- **Санкт-Петербург** → ищи `region_name = 'Санкт-Петербург'` (НЕ municipality!)  
- **Севастополь** → ищи `region_name = 'Севастополь'` (НЕ municipality!)

🏘️ **ОБЫЧНЫЕ ГОРОДА:**
1. **Пользователь упомянул обычный город** → ищи в `municipality LIKE '%ГородИмя%'`
2. **Пользователь упомянул регион** → ищи в `region_name LIKE '%РегионИмя%'`  
3. **Неопределенность** → используй OR: `municipality LIKE '%Название%' OR region_name LIKE '%Название%'`
4. **Уточнение результата** → добавь AND условия для фильтрации

📍 ПРИМЕР с JOIN для координат:
SELECT 
    k.municipality,
    k.region_name,
    ROUND(k.indicator_value, 2) as housing_index,
    d.municipal_district_center_lat as lat,
    d.municipal_district_center_lon as lon,
    RANK() OVER (ORDER BY k.indicator_value DESC) as rank_desc
FROM kom_sph k
JOIN dict_municipal_districts d ON k.territory_id = d.territory_id
WHERE k.indicator_name LIKE '%жилье%' AND k.indicator_value IS NOT NULL

📊 ПРИМЕР без координат (агрегация):
SELECT 
    region_name,
    AVG(indicator_value) as avg_value,
    COUNT(*) as count
FROM kom_sph
GROUP BY region_name

🔗 ТАБЛИЦЫ С КООРДИНАТАМИ НАПРЯМУЮ:
- bdmo_migration_full, bdmo_population_full, bdmo_salary_full
- consumption_full, market_access_full  
- dict_municipal_districts, t_dict_municipal_districts_poly_full

📋 ТАБЛИЦЫ БЕЗ КООРДИНАТ (нужен JOIN):
- kom_sph, employment_full, retail_catering
- organization_quantity, production_quantity, selhoz, selhoz_territory

### ОПИСАНИЕ ТАБЛИЦ

Розничная торговля и общественное питание | retail_catering
indicator_section_code - код раздела показателя
indicator_section - раздел показателя
indicator_code - код показателя
indicator_name - индикатор категории (Количество объектов розничной торговли и общественного питания,Площадь торгового зала объектов розничной торговли, Площадь зала обслуживания посетителей в объектах общественного питания, Число мест в объектах общественного питания, Число торговых мест на рынках, Число рынков, Число ярмарок, Число торговых мест на ярмарках, Оборот розничной торговли (без субъектов малого предпринимательства), Оборот общественного питания (без субъектов малого предпринимательства))
obroz - Виды объектов розничной торговли и общественного питания 
region_id - код региона
region_name - название региона
mun_level - уровень муниципального образования
mun_district - муниципальный район
municipality - название муниципалитета
oktmo - сквозной идентификатор муниципального округа
mun_type - тип муниципального образования
mun_type_oktmo - тип муниципального образования по ОКТМО
oktmo_stable - стабильный ОКТМО
oktmo_history - история ОКТМО
oktmo_year_from - год начала действия ОКТМО
oktmo_year_to - год окончания действия ОКТМО
year - год когда собраны данные
indicator_value - количество заведений
indicator_unit - единица измерения
indicator_period - временной отрезок показателя (квартальный)
comment - комментарий
territory_id - id муниципалитета у sberindex
market_access - индекс доступности рынка

Сельское хозяйство | selhoz_territory
indicator_section_code - код раздела показателя
indicator_section - раздел показателя
indicator_code - код показателя
indicator_name - индикатор категории (Посевные площади сельскохозяйственных культур (весеннего учета))
kategor - Категории хозяйств
kultur - Сельскохозяйственные культуры
region_id - код региона
region_name - название региона
mun_level - уровень муниципального образования
mun_district - муниципальный район
municipality - название муниципалитета
oktmo - сквозной идентификатор муниципального округа
mun_type - тип муниципального образования
mun_type_oktmo - тип муниципального образования по ОКТМО
oktmo_stable - стабильный ОКТМО
oktmo_history - история ОКТМО
oktmo_year_from - год начала действия ОКТМО
oktmo_year_to - год окончания действия ОКТМО
year - год когда собраны данные
indicator_value - площадь в гектарах
indicator_unit - единица измерения
indicator_period - временной отрезок показателя (Значение показателя за год)
comment - комментарий
territory_id - id муниципалитета у sberindex
market_access - индекс доступности рынка

Сельское хозяйство | selhoz
indicator_section_code - код раздела показателя
indicator_section - раздел показателя
indicator_code - код показателя
indicator_name - индикатор категории (Внесено органических удобрений под посевы сельскохозяйственных культур в сельскохозяйственных организациях)
region_id - код региона
region_name - название региона
mun_level - уровень муниципального образования
mun_district - муниципальный район
municipality - название муниципалитета
oktmo - сквозной идентификатор муниципального округа
mun_type - тип муниципального образования
mun_type_oktmo - тип муниципального образования по ОКТМО
oktmo_stable - стабильный ОКТМО
oktmo_history - история ОКТМО
oktmo_year_from - год начала действия ОКТМО
oktmo_year_to - год окончания действия ОКТМО
year - год когда собраны данные
indicator_value - вес удобрений в тоннах
indicator_unit - единица измерения
indicator_period - временной отрезок показателя (Значение показателя за год)
comment - комментарий
territory_id - id муниципалитета у sberindex
market_access - индекс доступности рынка

Коммунальная сфера | kom_sph
indicator_section_code - код раздела показателя
indicator_section - раздел показателя
indicator_code - код показателя
indicator_name - индикатор категории (Общая площадь жилых помещений, оборудованная одновременно водопроводом, водоотведением (канализацией), отоплением, горячим водоснабжением, газом или электрическими плитами)
region_id - код региона
region_name - название региона
mun_level - уровень муниципального образования
mun_district - муниципальный район
municipality - название муниципалитета
oktmo - сквозной идентификатор муниципального округа
mun_type - тип муниципального образования
mun_type_oktmo - тип муниципального образования по ОКТМО
oktmo_stable - стабильный ОКТМО
oktmo_history - история ОКТМО
oktmo_year_from - год начала действия ОКТМО
oktmo_year_to - год окончания действия ОКТМО
year - год когда собраны данные
indicator_value - площадь в тысяче квадратных метров
indicator_unit - единица измерения
indicator_period - временной отрезок показателя (Значение показателя за год)
comment - комментарий
territory_id - id муниципалитета у sberindex
market_access - индекс доступности рынка

Социальная поддержка населения | soc_people_quantity_payments_volume
indicator_section_code - код раздела показателя
indicator_section - раздел показателя
indicator_code - код показателя
indicator_name - индикатор категории (Сумма начисленных субсидий населению на оплату жилого помещения и коммунальных услуг)
region_id - код региона
region_name - название региона
mun_level - уровень муниципального образования
mun_district - муниципальный район
municipality - название муниципалитета
oktmo - сквозной идентификатор муниципального округа
mun_type - тип муниципального образования
mun_type_oktmo - тип муниципального образования по ОКТМО
oktmo_stable - стабильный ОКТМО
oktmo_history - история ОКТМО
oktmo_year_from - год начала действия ОКТМО
oktmo_year_to - год окончания действия ОКТМО
year - год когда собраны данные
indicator_value - размер выплаты * тысяча рублей
indicator_unit - единица измерения
indicator_period - временной отрезок показателя
comment - комментарий
territory_id - id муниципалитета у sberindex
market_access - индекс доступности рынка

Занятость и заработная плата | employment_full
indicator_section_code - код раздела показателя
indicator_section - раздел показателя
indicator_code - код показателя
indicator_name - индикатор категории (Просроченная задолженность по заработной плате работников организаций — всего (без субъектов малого предпринимательства), Просроченная задолженность по заработной плате из-за несвоевременного получения денежных средств из федерального бюджета (без субъектов малого предпринимательства), Просроченная задолженность по заработной плате из-за несвоевременного получения денежных средств из бюджетов субъектов (без субъектов малого предпринимательства), Просроченная задолженность по заработной плате из-за несвоевременного получения денежных средств из местных бюджетов (без субъектов малого предпринимательства), Среднесписочная численность работников организаций (без субъектов малого предпринимательства), Фонд заработной платы всех работников организаций (без субъектов малого предпринимательства), Среднемесячная заработная плата работников организаций (без субъектов малого предпринимательства))
okved2 - Виды экономической деятельности по ОКВЭД-2
region_id - код региона
region_name - название региона
mun_level - уровень муниципального образования
mun_district - муниципальный район
municipality - название муниципалитета
oktmo - сквозной идентификатор муниципального округа
mun_type - тип муниципального образования
mun_type_oktmo - тип муниципального образования по ОКТМО
oktmo_stable - стабильный ОКТМО
oktmo_history - история ОКТМО
oktmo_year_from - год начала действия ОКТМО
oktmo_year_to - год окончания действия ОКТМО
year - год когда собраны данные
indicator_value - размер выплаты * тысяча рублей
indicator_unit - единица измерения
indicator_period - временной отрезок показателя (на первое число месяца)
comment - комментарий
territory_id - id муниципалитета у sberindex
market_access - индекс доступности рынка

Деятельность предприятий | production_quantity
indicator_section_code - код раздела показателя
indicator_section - раздел показателя
indicator_code - код показателя
indicator_name - индикатор категории (Отгружено товаров собственного производства, выполнено работ и услуг собственными силами (без субъектов малого предпринимательства), средняя численность работников которых превышает 15 человек, по фактическим видам экономической деятельности, Отгружено товаров собственного производства, выполнено работ и услуг собственными силами (без субъектов малого предпринимательства), Продано товаров несобственного производства (без субъектов малого предпринимательства))
okved2 - Виды экономической деятельности по ОКВЭД-2
region_id - код региона
region_name - название региона
mun_level - уровень муниципального образования
mun_district - муниципальный район
municipality - название муниципалитета
oktmo - сквозной идентификатор муниципального округа
mun_type - тип муниципального образования
mun_type_oktmo - тип муниципального образования по ОКТМО
oktmo_stable - стабильный ОКТМО
oktmo_history - история ОКТМО
oktmo_year_from - год начала действия ОКТМО
oktmo_year_to - год окончания действия ОКТМО
year - год когда собраны данные
indicator_value - размер выплаты * тысяча рублей
indicator_unit - единица измерения
indicator_period - временной отрезок показателя (несколько месяцев)
comment - комментарий
territory_id - id муниципалитета у sberindex
market_access - индекс доступности рынка

Деятельность предприятий | organization_quantity
indicator_section_code - код раздела показателя
indicator_section - раздел показателя
indicator_code - код показателя
indicator_name - индикатор категории (Количество организаций по данным государственной регистрации, Количество индивидуальных предпринимателей по данным государственной регистрации)
region_id - код региона
region_name - название региона
mun_level - уровень муниципального образования
mun_district - муниципальный район
municipality - название муниципалитета
oktmo - сквозной идентификатор муниципального округа
mun_type - тип муниципального образования
mun_type_oktmo - тип муниципального образования по ОКТМО
oktmo_stable - стабильный ОКТМО
oktmo_history - история ОКТМО
oktmo_year_from - год начала действия ОКТМО
oktmo_year_to - год окончания действия ОКТМО
year - год когда собраны данные
indicator_value - число людей
indicator_unit - единица измерения
indicator_period - временной отрезок показателя (на первое число месяца)
comment - комментарий
territory_id - id муниципалитета у sberindex
market_access - индекс доступности рынка

Данные о миграции населения по муниципальным округам с разделением по возрастам и полам | bdmo_migration_full
territory_id - id муниципалитета у sberindex
year - год
period - за какой период миграция
age - возраст мигрантов
gender - пол
value - количество
change_id_from - id изменения откуда
change_id_to - id изменения куда
municipal_district_center - центр муниципалитета
municipal_district_center_lat - координаты центра муниципалитета
municipal_district_center_lon - координаты центра муниципалитета
municipal_district_name - название муниципалитета
municipal_district_name_short - краткое название муниципалитета
municipal_district_status - статус субъекта
municipal_district_type - тип муниципалитета
oktmo - сквозной идентификатор муниципального округа
region_code - код региона
region_name - название региона
shape - форма
shape_linked_oktmo - связанный ОКТМО формы
year_from - год начала
year_to - год конца
source_nm - источник данных

Данные о плотности населения по муниципальным округам с разделением по возрастам и полам | bdmo_population_full
territory_id - id муниципалитета у sberindex
year - год
period - за какой период расчет
age - возраст жителей
gender - пол
value - количество
change_id_from - id изменения откуда
change_id_to - id изменения куда
municipal_district_center - центр муниципалитета
municipal_district_center_lat - координаты центра муниципалитета
municipal_district_center_lon - координаты центра муниципалитета
municipal_district_name - название муниципалитета
municipal_district_name_short - краткое название муниципалитета
municipal_district_status - статус субъекта
municipal_district_type - тип муниципалитета
oktmo - сквозной идентификатор муниципального округа
region_code - код региона
region_name - название региона
shape - форма
shape_linked_oktmo - связанный ОКТМО формы
year_from - год начала
year_to - год конца
source_nm - источник данных

Данные о средних полугодовых зарплатах по муниципальным округам с разделением по направлениям | bdmo_salary_full
territory_id - id муниципалитета у sberindex
year - год
period - за какой период расчет
okved_name - направление по зарплатам
okved_letter - буква ОКВЭД
value - средняя зарплата
change_id_from - id изменения откуда
change_id_to - id изменения куда
municipal_district_center - центр муниципалитета
municipal_district_center_lat - координаты центра муниципалитета
municipal_district_center_lon - координаты центра муниципалитета
municipal_district_name - название муниципалитета
municipal_district_name_short - краткое название муниципалитета
municipal_district_status - статус субъекта
municipal_district_type - тип муниципалитета
oktmo - сквозной идентификатор муниципального округа
region_code - код региона
region_name - название региона
shape - форма
shape_linked_oktmo - связанный ОКТМО формы
year_from - год начала
year_to - год конца
source_nm - источник данных

Расстояние между муниципальными округами | connection_full
territory_id_x - id муниципалитета откуда
territory_id_y - id муниципалитета куда
distance - расстояние в километрах
change_id_from_x - id изменения откуда (для x)
change_id_to_x - id изменения куда (для x)
municipal_district_center_x - центр муниципалитета (для x)
municipal_district_center_lat_x - координаты центра муниципалитета (для x)
municipal_district_center_lon_x - координаты центра муниципалитета (для x)
municipal_district_name_x - название муниципалитета (для x)
municipal_district_name_short_x - краткое название муниципалитета (для x)
municipal_district_status_x - статус субъекта (для x)
municipal_district_type_x - тип муниципалитета (для x)
oktmo_x - сквозной идентификатор муниципального округа (для x)
region_code_x - код региона (для x)
region_name_x - название региона (для x)
shape_x - форма (для x)
shape_linked_oktmo_x - связанный ОКТМО формы (для x)
year_from_x - год начала (для x)
year_to_x - год конца (для x)
change_id_from_y - id изменения откуда (для y)
change_id_to_y - id изменения куда (для y)
municipal_district_center_y - центр муниципалитета (для y)
municipal_district_center_lat_y - координаты центра муниципалитета (для y)
municipal_district_center_lon_y - координаты центра муниципалитета (для y)
municipal_district_name_y - название муниципалитета (для y)
municipal_district_name_short_y - краткое название муниципалитета (для y)
municipal_district_status_y - статус субъекта (для y)
municipal_district_type_y - тип муниципалитета (для y)
oktmo_y - сквозной идентификатор муниципального округа (для y)
region_code_y - код региона (для y)
region_name_y - название региона (для y)
shape_y - форма (для y)
shape_linked_oktmo_y - связанный ОКТМО формы (для y)
year_from_y - год начала (для y)
year_to_y - год конца (для y)
source_nm - источник данных

Геометрия полигонов | t_dict_municipal_districts_poly_full
municipal_district_name_short - краткое название муниципалитета
oktmo - сквозной идентификатор муниципального округа
municipal_district_name - название муниципалитета
municipal_district_type - тип муниципалитета
municipal_district_status - статус субъекта
shape - форма
shape_linked_oktmo - связанный ОКТМО формы
municipal_district_center - центр муниципалитета
source_rosstat - источник данных Росстат
year_from - год начала периода
year_to - год конца периода
territory_id - id муниципалитета у sberindex
change_id_from - id изменения откуда
change_id_to - id изменения куда
region_code - код региона
region_name - название региона
municipal_district_center_lat - координаты центра муниципалитета
municipal_district_center_lon - координаты центра муниципалитета
fid - идентификатор записи
geom - мультиполигон по муниципалитету (геометрия)
osm_ref - референс OpenStreetMap
osm_vers - версия OpenStreetMap
territory_id_1 - дополнительный id муниципалитета
year_from_1 - дополнительный год начала
year_to_1 - дополнительный год конца
source_nm - источник данных

Индекс доступности рынка | market_access_full
municipal_district_name_short - краткое название муниципалитета
oktmo - сквозной идентификатор муниципального округа
municipal_district_name - название муниципалитета
municipal_district_type - тип муниципалитета
municipal_district_status - статус субъекта
shape - форма
shape_linked_oktmo - связанный ОКТМО формы
municipal_district_center - центр муниципалитета (название населенного пункта)
source_rosstat - источник данных Росстат
year_from - год начала периода
year_to - год конца периода
territory_id - id муниципалитета у sberindex
change_id_from - id изменения откуда
change_id_to - id изменения куда
region_code - код региона
region_name - название региона
municipal_district_center_lat - координаты центра муниципалитета
municipal_district_center_lon - координаты центра муниципалитета
territory_id_1 - дополнительный id муниципалитета
market_access - индекс доступности рынка

Справочник от sberindex | dict_municipal_districts
municipal_district_name_short - краткое название муниципалитета
oktmo - сквозной идентификатор муниципального округа
municipal_district_name - название муниципалитета
municipal_district_type - тип муниципалитета
municipal_district_status - статус субъекта
shape - форма
shape_linked_oktmo - связанный ОКТМО формы
municipal_district_center - центр муниципалитета
source_rosstat - источник данных Росстат
year_from - год начала периода
year_to - год конца периода
territory_id - id муниципалитета у sberindex
change_id_from - id изменения откуда
change_id_to - id изменения куда
region_code - код региона
region_name - название региона
municipal_district_center_lat - координаты центра муниципалитета
municipal_district_center_lon - координаты центра муниципалитета

Отчет о потребительских расходах | consumption_full
date - Год и месяц
territory_id - id муниципалитета у sberindex
category - Категория потребительских расходов
value - Оценка средних безналичных потребительских расходов в текущем месяце жителей МО на основе моделей СберИндекса на транзакционных данных (руб.)
__index_level_0__ - индекс уровня
municipal_district_name_short - краткое название муниципалитета
oktmo - сквозной идентификатор муниципального округа
municipal_district_name - название муниципалитета
municipal_district_type - тип муниципалитета
municipal_district_status - статус субъекта
shape - форма
shape_linked_oktmo - связанный ОКТМО формы
municipal_district_center - центр муниципалитета
source_rosstat - источник данных Росстат
year_from - год начала периода
year_to - год конца периода
territory_id_1 - дополнительный id муниципалитета
change_id_from - id изменения откуда
change_id_to - id изменения куда
region_code - код региона
region_name - название региона
municipal_district_center_lat - координаты центра муниципалитета
municipal_district_center_lon - координаты центра муниципалитета 

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
MAX_RETRIES = 12
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
    
    # Проверяем директорию для базы данных (только если путь содержит директорию)
    db_dir = os.path.dirname(DATABASE_URL)
    if db_dir and not os.path.exists(db_dir):
        try:
            os.makedirs(db_dir, exist_ok=True)
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


def get_openai_settings() -> dict:
    """
    Получение настроек OpenAI из переменных окружения.
    
    Returns:
        Словарь с настройками OpenAI
    """
    return {
        'api_key': OPENAI_API_KEY,
        'base_url': OPENAI_BASE_URL,
        'model': OPENAI_MODEL,
        'temperature': OPENAI_TEMPERATURE,
        'max_tokens': OPENAI_MAX_TOKENS
    }