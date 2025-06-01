"""
Модуль для подключения к DuckDB и управления схемой данных.
Отвечает за создание таблиц, загрузку данных и выполнение SQL-запросов.
"""

import logging
import os
import duckdb
import pandas as pd
import re
from typing import Optional, Dict, Any, List
from pathlib import Path

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Менеджер базы данных DuckDB для SberIndexNavigator."""
    
    def __init__(self, database_path: str = "data/sber_index.db"):
        """
        Инициализация подключения к базе данных.
        
        Args:
            database_path: Путь к файлу базы данных DuckDB
        """
        self.database_path = database_path
        self.connection: Optional[duckdb.DuckDBPyConnection] = None
        self._connect()
        
    def _connect(self) -> None:
        """Создание подключения к базе данных."""
        try:
            # Создаем директорию для БД если не существует
            os.makedirs(os.path.dirname(self.database_path), exist_ok=True)
            
            self.connection = duckdb.connect(self.database_path)
            logger.info(f"Подключение к базе данных {self.database_path} установлено")
        except Exception as e:
            logger.error(f"Ошибка подключения к базе данных: {e}")
            raise
    
    def clean_sql_query(self, query: str) -> str:
        """
        Очистка SQL-запроса от markdown форматирования и других артефактов.
        
        Args:
            query: Сырой SQL-запрос, возможно с markdown форматированием
            
        Returns:
            Очищенный SQL-запрос
        """
        try:
            # Удаляем markdown code blocks
            query = re.sub(r'```sql\s*', '', query)
            query = re.sub(r'```\s*', '', query)
            
            # Удаляем дополнительные пробелы и переносы строк
            query = query.strip()
            
            # Удаляем комментарии вида "-- объяснение"
            lines = query.split('\n')
            cleaned_lines = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith('--'):
                    cleaned_lines.append(line)
            
            query = ' '.join(cleaned_lines)
            
            # Убираем лишние пробелы
            query = re.sub(r'\s+', ' ', query)
            
            # Убираем точку с запятой в конце если есть
            query = query.rstrip(';')
            
            logger.debug(f"Очищенный SQL-запрос: {query}")
            return query
            
        except Exception as e:
            logger.error(f"Ошибка очистки SQL-запроса: {e}")
            return query.strip()
    
    def execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Выполнение SQL-запроса и возврат результата как DataFrame.
        
        Args:
            query: SQL-запрос
            parameters: Параметры для запроса
            
        Returns:
            DataFrame с результатами запроса
        """
        try:
            # Очищаем запрос от markdown и артефактов
            clean_query = self.clean_sql_query(query)
            
            if parameters:
                result = self.connection.execute(clean_query, parameters).fetchdf()
            else:
                result = self.connection.execute(clean_query).fetchdf()
            
            logger.info(f"Запрос выполнен успешно, получено {len(result)} строк")
            return result
        except Exception as e:
            logger.error(f"Ошибка выполнения запроса: {e}")
            logger.error(f"Исходный запрос: {query}")
            logger.error(f"Очищенный запрос: {self.clean_sql_query(query)}")
            raise
    
    def create_tables(self) -> None:
        """Создание схемы таблиц для индексов Сбербанка."""
        try:
            # Таблица расходов по муниципалитетам
            self.connection.execute("""
                CREATE TABLE IF NOT EXISTS region_spending (
                    region VARCHAR,
                    region_code INTEGER,
                    month VARCHAR,
                    year INTEGER,
                    consumer_spending DOUBLE,
                    housing_index DOUBLE,
                    transport_accessibility DOUBLE,
                    market_accessibility DOUBLE
                )
            """)
            
            # Таблица демографических данных
            self.connection.execute("""
                CREATE TABLE IF NOT EXISTS demographics (
                    region VARCHAR,
                    region_code INTEGER,
                    population INTEGER,
                    age_median DOUBLE,
                    income_median DOUBLE,
                    unemployment_rate DOUBLE,
                    education_index DOUBLE
                )
            """)
            
            # Таблица транспортной доступности
            self.connection.execute("""
                CREATE TABLE IF NOT EXISTS transport_data (
                    region VARCHAR,
                    region_code INTEGER,
                    transport_score DOUBLE,
                    public_transport_coverage DOUBLE,
                    road_quality_index DOUBLE,
                    airport_accessibility DOUBLE,
                    railway_connectivity DOUBLE
                )
            """)
            
            logger.info("Схема таблиц создана успешно")
        except Exception as e:
            logger.error(f"Ошибка создания таблиц: {e}")
            raise
    
    def load_test_data(self) -> None:
        """Загрузка тестовых данных из CSV файлов."""
        try:
            data_dir = Path("data")
            
            # Загрузка данных по муниципалитетам
            if (data_dir / "test_region.csv").exists():
                df_region = pd.read_csv(data_dir / "test_region.csv")
                self.connection.register("region_df", df_region)
                # Очищаем таблицу перед загрузкой
                self.connection.execute("DELETE FROM region_spending")
                self.connection.execute("""
                    INSERT INTO region_spending 
                    SELECT * FROM region_df
                """)
                logger.info(f"Загружено {len(df_region)} записей в region_spending")
            
            # Загрузка демографических данных
            if (data_dir / "test_demography.csv").exists():
                df_demo = pd.read_csv(data_dir / "test_demography.csv")
                self.connection.register("demo_df", df_demo)
                # Очищаем таблицу перед загрузкой
                self.connection.execute("DELETE FROM demographics")
                self.connection.execute("""
                    INSERT INTO demographics
                    SELECT * FROM demo_df
                """)
                logger.info(f"Загружено {len(df_demo)} записей в demographics")
            
            # Загрузка данных по транспорту
            if (data_dir / "test_transport.csv").exists():
                df_transport = pd.read_csv(data_dir / "test_transport.csv")
                self.connection.register("transport_df", df_transport)
                # Очищаем таблицу перед загрузкой
                self.connection.execute("DELETE FROM transport_data")
                self.connection.execute("""
                    INSERT INTO transport_data
                    SELECT * FROM transport_df
                """)
                logger.info(f"Загружено {len(df_transport)} записей в transport_data")
                
        except Exception as e:
            logger.error(f"Ошибка загрузки тестовых данных: {e}")
            raise
    
    def get_table_info(self) -> Dict[str, List[str]]:
        """
        Получение информации о структуре таблиц.
        
        Returns:
            Словарь с именами таблиц и их колонками
        """
        try:
            tables_info = {}
            
            # Получаем список таблиц
            tables = self.connection.execute("SHOW TABLES").fetchdf()
            
            for table_name in tables['name']:
                columns = self.connection.execute(f"DESCRIBE {table_name}").fetchdf()
                tables_info[table_name] = columns['column_name'].tolist()
            
            return tables_info
        except Exception as e:
            logger.error(f"Ошибка получения информации о таблицах: {e}")
            return {}
    
    def validate_query(self, query: str) -> bool:
        """
        Валидация SQL-запроса без его выполнения.
        
        Args:
            query: SQL-запрос для валидации
            
        Returns:
            True если запрос валиден, False иначе
        """
        try:
            # Очищаем запрос перед валидацией
            clean_query = self.clean_sql_query(query)
            
            # Используем EXPLAIN для проверки синтаксиса
            self.connection.execute(f"EXPLAIN {clean_query}")
            return True
        except Exception as e:
            logger.warning(f"Невалидный запрос: {e}")
            logger.warning(f"Исходный запрос: {query}")
            logger.warning(f"Очищенный запрос: {self.clean_sql_query(query)}")
            return False
    
    def close(self) -> None:
        """Закрытие подключения к базе данных."""
        if self.connection:
            self.connection.close()
            logger.info("Подключение к базе данных закрыто")
    
    def __enter__(self):
        """Context manager enter."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# Singleton instance для использования в приложении
db_manager: Optional[DatabaseManager] = None


def get_database_manager() -> DatabaseManager:
    """
    Получение единственного экземпляра DatabaseManager.
    
    Returns:
        Экземпляр DatabaseManager
    """
    global db_manager
    if db_manager is None:
        db_manager = DatabaseManager()
    return db_manager


def initialize_database() -> None:
    """Инициализация базы данных с созданием таблиц и загрузкой данных."""
    try:
        db = get_database_manager()
        db.create_tables()
        db.load_test_data()
        logger.info("База данных инициализирована успешно")
    except Exception as e:
        logger.error(f"Ошибка инициализации базы данных: {e}")
        raise 