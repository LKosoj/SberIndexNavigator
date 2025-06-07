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

duckdb.install_extension('spatial')
duckdb.load_extension('spatial')

class DatabaseManager:
    """Менеджер базы данных DuckDB для SberIndexNavigator."""
    
    def __init__(self, database_path: Optional[str] = None):
        """
        Инициализация подключения к базе данных.
        
        Args:
            database_path: Путь к файлу базы данных DuckDB. Если None, используется значение из настроек.
        """
        if database_path is None:
            # Импортируем настройки и используем правильный путь к БД
            from src.config.settings import DATABASE_URL
            database_path = DATABASE_URL
            logger.debug(f"Получен путь из настроек: {repr(database_path)}")
            
            # Если переменная окружения пуста или None, используем default
            if not database_path:
                database_path = "data/sber_index_prod.db"
                logger.warning(f"DATABASE_URL пуста, используем default: {database_path}")
        
        # Нормализуем путь для избежания проблем с относительными путями
        self.database_path = os.path.abspath(database_path)
        logger.info(f"Инициализация DatabaseManager с путём: {self.database_path}")
        
        self.connection: Optional[duckdb.DuckDBPyConnection] = None
        self._connect()
        
    def _connect(self) -> None:
        """Создание подключения к базе данных."""
        try:
            # Проверяем что путь не пустой
            if not self.database_path or self.database_path.strip() == '':
                raise ValueError("Путь к базе данных не может быть пустым")
            
            # Создаем директорию для БД если не существует
            db_dir = os.path.dirname(self.database_path)
            if db_dir:  # Проверяем что dirname не пустой
                os.makedirs(db_dir, exist_ok=True)
                logger.debug(f"Директория создана/проверена: {db_dir}")
            
            # Проверяем существование файла
            if os.path.exists(self.database_path):
                logger.info(f"Файл базы данных найден: {self.database_path}")
            else:
                logger.warning(f"Файл базы данных не существует, будет создан: {self.database_path}")
            
            self.connection = duckdb.connect(self.database_path)
            logger.info(f"Подключение к базе данных {self.database_path} установлено")
            
        except Exception as e:
            logger.error(f"Ошибка подключения к базе данных: {e}")
            logger.error(f"Путь: {repr(self.database_path)}")
            logger.error(f"Текущая директория: {os.getcwd()}")
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
        
    def get_table_info(self) -> Dict[str, List[str]]:
        """
        Получение информации о структуре таблиц из ВСЕХ доступных схем.
        
        Returns:
            Словарь с именами таблиц и их колонками
        """
        try:
            tables_info = {}
            
            # Получаем таблицы из ВСЕХ доступных схем
            schemas_to_check = ['main', 'sber_index']
            
            for schema_name in schemas_to_check:
                try:
                    # Переключаемся на схему
                    self.connection.execute(f"USE {schema_name}")
                    
                    # Получаем список таблиц в этой схеме
                    tables = self.connection.execute("SHOW TABLES").fetchdf()
                    
                    logger.info(f"Обрабатываем схему {schema_name}: найдено {len(tables)} таблиц")
                    
                    for table_name in tables['name']:
                        # Используем полное имя таблицы с схемой для уникальности
                        # Но проверяем, не дублируется ли таблица
                        if table_name not in tables_info:
                            try:
                                columns = self.connection.execute(f"DESCRIBE {schema_name}.{table_name}").fetchdf()
                                tables_info[table_name] = columns['column_name'].tolist()
                                logger.debug(f"Добавлена таблица {table_name} из схемы {schema_name}")
                            except Exception as e:
                                logger.debug(f"Не удалось получить колонки для {schema_name}.{table_name}: {e}")
                        else:
                            logger.debug(f"Таблица {table_name} уже существует, пропускаем")
                            
                except Exception as e:
                    logger.warning(f"Не удалось обработать схему {schema_name}: {e}")
                    continue
            
            logger.info(f"Получена информация о {len(tables_info)} уникальных таблицах из всех схем")
            return tables_info
            
        except Exception as e:
            logger.error(f"Ошибка получения информации о таблицах: {e}")
            # Fallback к старому способу (только main схема)
            try:
                self.connection.execute("USE main")
                tables = self.connection.execute("SHOW TABLES").fetchdf()
                tables_info = {}
                for table_name in tables['name']:
                    columns = self.connection.execute(f"DESCRIBE {table_name}").fetchdf()
                    tables_info[table_name] = columns['column_name'].tolist()
                return tables_info
            except Exception as fallback_error:
                logger.error(f"Fallback также не сработал: {fallback_error}")
                return {}
    
    def get_database_summary(self) -> Dict[str, Any]:
        """
        Получение расширенной информации о базе данных из ВСЕХ доступных схем.
        
        Returns:
            Словарь с подробной информацией о таблицах, включая количество записей и описания
        """
        try:
            summary = {
                "tables": {},
                "total_tables": 0,
                "total_records": 0,
                "available_regions": []
            }
            
            # Получаем таблицы из ВСЕХ схем через наш обновленный метод
            all_tables_info = self.get_table_info()
            summary["total_tables"] = len(all_tables_info)
            
            # Расширенные описания таблиц для пользователей
            table_descriptions = {
                # Основные таблицы
                "all_data": "агрегированные данные по всем показателям",
                "employment_full": "данные о занятости населения",
                "retail_catering": "данные о розничной торговле и общественном питании",
                "organization_quantity": "количество организаций по отраслям",
                "production_quantity": "объемы производства товаров",
                "soc_people_quantity_payments_volume": "социальные выплаты и численность населения",
                "kom_sph": "коммунальная сфера",
                "selhoz": "сельскохозяйственные данные", 
                "selhoz_territory": "сельскохозяйственные территории",
                "v_rosstat_data": "данные Росстата",
                
                # Расширенные таблицы из sber_index схемы
                "dict_municipal_districts": "🏛️ справочник муниципальных районов",
                "bdmo_migration_full": "🚶 данные о миграции населения",
                "bdmo_population_full": "👥 демографические данные населения",
                "bdmo_salary_full": "💰 данные о заработной плате",
                "consumption_full": "🛒 данные о потреблении товаров и услуг",
                "connection_full": "🌐 данные о связности территорий",
                "market_access_full": "🏪 доступность рынков и услуг",
                "t_dict_municipal_districts_poly_full": "🗺️ геополигоны муниципальных районов"
            }
            
            # Переключаемся на схему sber_index для получения статистики
            try:
                self.connection.execute("USE sber_index")
            except:
                # Если sber_index недоступна, используем main
                self.connection.execute("USE main")
            
            for table_name, columns in all_tables_info.items():
                try:
                    # Пытаемся получить количество записей из текущей схемы
                    count_result = self.connection.execute(f"SELECT COUNT(*) as count FROM {table_name}").fetchdf()
                    record_count = int(count_result['count'].iloc[0])
                    summary["total_records"] += record_count
                    
                    # Сохраняем информацию о таблице
                    summary["tables"][table_name] = {
                        "columns": columns,
                        "record_count": record_count,
                        "description": table_descriptions.get(table_name, "данные")
                    }
                    
                except Exception as e:
                    # Если не удалось получить count, пытаемся из другой схемы
                    try:
                        # Пробуем с полным именем схемы
                        count_result = self.connection.execute(f"SELECT COUNT(*) as count FROM sber_index.{table_name}").fetchdf()
                        record_count = int(count_result['count'].iloc[0])
                        summary["total_records"] += record_count
                        
                        summary["tables"][table_name] = {
                            "columns": columns,
                            "record_count": record_count,
                            "description": table_descriptions.get(table_name, "данные")
                        }
                    except Exception as e2:
                        # Если и это не сработало, пропускаем
                        logger.debug(f"Не удалось получить count для таблицы {table_name}: {e2}")
                        continue
            
            # Получаем список уникальных муниципальных районов из справочника
            try:
                regions_query = """
                SELECT DISTINCT municipal_district_name_short 
                FROM dict_municipal_districts 
                WHERE municipal_district_name_short IS NOT NULL 
                ORDER BY municipal_district_name_short 
                """
                regions_result = self.connection.execute(regions_query).fetchdf()
                summary["available_regions"] = regions_result['municipal_district_name_short'].tolist()
                logger.info(f"Найдено {len(summary['available_regions'])} муниципальных районов в справочнике")
            except Exception as e1:
                logger.debug(f"Не удалось получить список регионов из dict_municipal_districts: {e1}")
                # Fallback к альтернативным источникам
                try:
                    regions_query = """
                    SELECT DISTINCT region 
                    FROM all_data 
                    WHERE region IS NOT NULL 
                    ORDER BY region 
                    LIMIT 50
                    """
                    regions_result = self.connection.execute(regions_query).fetchdf()
                    summary["available_regions"] = regions_result['region'].tolist()
                    logger.info(f"Fallback: найдено {len(summary['available_regions'])} регионов в all_data")
                except Exception as e2:
                    logger.debug(f"Fallback также не сработал: {e2}")
                    summary["available_regions"] = []
            
            logger.info(f"Получена информация о {summary['total_tables']} таблицах с {summary['total_records']} записями из всех схем")
            return summary
            
        except Exception as e:
            logger.error(f"Ошибка получения сводки базы данных: {e}")
            return {
                "tables": {},
                "total_tables": 0,
                "total_records": 0,
                "available_regions": []
            }
    
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
        logger.info("База данных инициализирована успешно")
    except Exception as e:
        logger.error(f"Ошибка инициализации базы данных: {e}")
        raise 