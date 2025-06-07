"""
Модуль для геокодинга городов и муниципалитетов.
Включает справочник координат и LLM-геокодинг.
"""

import logging
import pandas as pd
from typing import Optional, Tuple, Dict, Any
import re

logger = logging.getLogger(__name__)

# Справочник координат основных городов России
CITY_COORDINATES = {
    # Города федерального значения
    "москва": (55.7558, 37.6176),
    "санкт-петербург": (59.9311, 30.3609),
    "севастополь": (44.6054, 33.5221),
    
    # Столицы регионов и крупные города
    "казань": (55.8304, 49.0661),
    "город казань": (55.8304, 49.0661),
    "екатеринбург": (56.8431, 60.6454),
    "город екатеринбург": (56.8431, 60.6454),
    "новосибирск": (55.0084, 82.9357),
    "город новосибирск": (55.0084, 82.9357),
    "нижний новгород": (56.2965, 43.9361),
    "город нижний новгород": (56.2965, 43.9361),
    "челябинск": (55.1644, 61.4368),
    "город челябинск": (55.1644, 61.4368),
    "самара": (53.2001, 50.15),
    "город самара": (53.2001, 50.15),
    "омск": (54.9885, 73.3242),
    "город омск": (54.9885, 73.3242),
    "ростов-на-дону": (47.2357, 39.7015),
    "город ростов-на-дону": (47.2357, 39.7015),
    "уфа": (54.7388, 55.9721),
    "город уфа": (54.7388, 55.9721),
    "красноярск": (56.0184, 92.8672),
    "город красноярск": (56.0184, 92.8672),
    "воронеж": (51.6605, 39.2006),
    "город воронеж": (51.6605, 39.2006),
    "пермь": (58.0105, 56.2502),
    "город пермь": (58.0105, 56.2502),
    "волгоград": (48.708, 44.5133),
    "город волгоград": (48.708, 44.5133),
    "краснодар": (45.0355, 38.975),
    "город краснодар": (45.0355, 38.975),
    "саратов": (51.5924, 46.0348),
    "город саратов": (51.5924, 46.0348),
    "тюмень": (57.1522, 65.5272),
    "город тюмень": (57.1522, 65.5272),
    "тольятти": (53.5303, 49.3461),
    "город тольятти": (53.5303, 49.3461),
    "ижевск": (56.8527, 53.2112),
    "город ижевск": (56.8527, 53.2112),
    "барнаул": (53.3606, 83.7636),
    "город барнаул": (53.3606, 83.7636),
    "ульяновск": (54.3142, 48.4031),
    "город ульяновск": (54.3142, 48.4031),
    "иркутск": (52.2978, 104.2964),
    "город иркутск": (52.2978, 104.2964),
    "хабаровск": (48.4827, 135.0840),
    "город хабаровск": (48.4827, 135.0840),
    "ярославль": (57.6261, 39.8845),
    "город ярославль": (57.6261, 39.8845),
    "владивосток": (43.1056, 131.8735),
    "город владивосток": (43.1056, 131.8735),
    "махачкала": (42.9849, 47.5047),
    "город махачкала": (42.9849, 47.5047),
    "томск": (56.5018, 84.9776),
    "город томск": (56.5018, 84.9776),
    "оренбург": (51.7727, 55.0988),
    "город оренбург": (51.7727, 55.0988),
    "кемерово": (55.3331, 86.0827),
    "город кемерово": (55.3331, 86.0827),
    "новокузнецк": (53.7596, 87.1216),
    "город новокузнецк": (53.7596, 87.1216),
    "рязань": (54.6269, 39.6916),
    "город рязань": (54.6269, 39.6916),
    "астрахань": (46.3497, 48.0408),
    "город астрахань": (46.3497, 48.0408),
    "пенза": (53.1925, 45.0184),
    "город пенза": (53.1925, 45.0184),
    "липецк": (52.6031, 39.5708),
    "город липецк": (52.6031, 39.5708),
    "тула": (54.1961, 37.6182),
    "город тула": (54.1961, 37.6182),
    "киров": (58.5966, 49.6601),
    "город киров": (58.5966, 49.6601),
    "чебоксары": (56.1439, 47.2517),
    "город чебоксары": (56.1439, 47.2517),
    "калининград": (54.7065, 20.511),
    "город калининград": (54.7065, 20.511),
    "брянск": (53.2434, 34.3641),
    "город брянск": (53.2434, 34.3641),
    "курск": (51.7373, 36.1873),
    "город курск": (51.7373, 36.1873),
    "иваново": (56.9969, 40.9819),
    "город иваново": (56.9969, 40.9819),
    "магнитогорск": (53.4678, 59.0268),
    "город магнитогорск": (53.4678, 59.0268),
    "тверь": (56.8587, 35.9176),
    "город тверь": (56.8587, 35.9176),
    "ставрополь": (45.0428, 41.9691),
    "город ставрополь": (45.0428, 41.9691),
    "сочи": (43.6028, 39.7342),
    "город сочи": (43.6028, 39.7342),
}


class GeocodingService:
    """Сервис для определения координат городов."""
    
    def __init__(self):
        """Инициализация сервиса геокодинга."""
        logger.info("Инициализация сервиса геокодинга")
    
    def normalize_city_name(self, city_name: str) -> str:
        """
        Нормализация названия города для поиска в справочнике.
        
        Args:
            city_name: Название города
            
        Returns:
            Нормализованное название
        """
        if not city_name:
            return ""
        
        # Приводим к нижнему регистру
        normalized = city_name.lower().strip()
        
        # Удаляем лишние пробелы
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Стандартизируем некоторые сокращения
        normalized = normalized.replace('г. ', 'город ')
        normalized = normalized.replace('г.', 'город')
        
        # Убираем кавычки
        normalized = normalized.replace('"', '').replace("'", "")
        
        return normalized
    
    def get_coordinates_from_dict(self, city_name: str) -> Optional[Tuple[float, float]]:
        """
        Получение координат из встроенного справочника.
        
        Args:
            city_name: Название города
            
        Returns:
            Кортеж (широта, долгота) или None
        """
        normalized = self.normalize_city_name(city_name)
        
        # Прямой поиск
        if normalized in CITY_COORDINATES:
            coords = CITY_COORDINATES[normalized]
            logger.info(f"Найдены координаты для '{city_name}': {coords}")
            return coords
        
        # Поиск без префикса "город"
        if normalized.startswith('город '):
            short_name = normalized[6:]  # убираем "город "
            if short_name in CITY_COORDINATES:
                coords = CITY_COORDINATES[short_name]
                logger.info(f"Найдены координаты для '{city_name}' (без префикса): {coords}")
                return coords
        
        # Поиск с префиксом "город"
        if not normalized.startswith('город '):
            full_name = f"город {normalized}"
            if full_name in CITY_COORDINATES:
                coords = CITY_COORDINATES[full_name]
                logger.info(f"Найдены координаты для '{city_name}' (с префиксом): {coords}")
                return coords
        
        # Частичный поиск (для сложных названий)
        for dict_city, coords in CITY_COORDINATES.items():
            if normalized in dict_city or dict_city in normalized:
                logger.info(f"Найдены координаты для '{city_name}' (частичное совпадение с '{dict_city}'): {coords}")
                return coords
        
        logger.debug(f"Координаты для '{city_name}' не найдены в справочнике")
        return None
    
    def get_coordinates_via_llm(self, city_name: str, region_name: str = None) -> Optional[Tuple[float, float]]:
        """
        Получение координат через LLM (для неизвестных городов).
        
        Args:
            city_name: Название города
            region_name: Название региона (для уточнения)
            
        Returns:
            Кортеж (широта, долгота) или None
        """
        try:
            from src.config.settings import get_openai_settings
            from openai import OpenAI
            import os
            
            # Получаем настройки из конфига
            openai_settings = get_openai_settings()
            
            # Проверяем наличие настроек
            if not openai_settings.get('api_key') and not openai_settings.get('base_url'):
                logger.debug("LLM геокодинг недоступен: нет настроек OpenAI в конфиге")
                return None
            
            # Создаем клиента с настройками из конфига
            client_kwargs = {}
            if openai_settings.get('api_key'):
                client_kwargs['api_key'] = openai_settings['api_key']
            if openai_settings.get('base_url'):
                client_kwargs['base_url'] = openai_settings['base_url']
            
            client = OpenAI(**client_kwargs)
            
            # Формируем запрос
            query = f"Город: {city_name}"
            if region_name:
                query += f", Регион: {region_name}"
            
            prompt = f"""
Определи координаты (широта, долгота) для российского муниципалитета: {query}

Верни ТОЛЬКО числа в формате: широта,долгота
Например: 55.7558,37.6176

Если не знаешь точные координаты, верни: null
"""
            
            # Используем модель и настройки из конфига
            response = client.chat.completions.create(
                model=openai_settings.get('model', 'gpt-3.5-turbo'),
                messages=[{"role": "user", "content": prompt}],
                temperature=openai_settings.get('temperature', 0.1),
                max_tokens=50
            )
            
            result = response.choices[0].message.content.strip()
            
            if result == "null" or not result:
                logger.debug(f"LLM не смог определить координаты для '{city_name}'")
                return None
            
            # Парсим результат
            try:
                lat_str, lon_str = result.split(',')
                lat = float(lat_str.strip())
                lon = float(lon_str.strip())
                
                # Валидация координат для России
                if 41 <= lat <= 82 and 19 <= lon <= 180:
                    logger.info(f"LLM определил координаты для '{city_name}': ({lat}, {lon})")
                    return (lat, lon)
                else:
                    logger.warning(f"LLM вернул некорректные координаты для '{city_name}': ({lat}, {lon})")
                    return None
                    
            except (ValueError, IndexError) as e:
                logger.warning(f"Ошибка парсинга координат от LLM для '{city_name}': {result}, error: {e}")
                return None
                
        except Exception as e:
            logger.warning(f"Ошибка LLM геокодинга для '{city_name}': {e}")
            return None
    
    def geocode_city(self, city_name: str, region_name: str = None) -> Optional[Tuple[float, float]]:
        """
        Получение координат города с использованием всех доступных методов.
        
        Args:
            city_name: Название города
            region_name: Название региона
            
        Returns:
            Кортеж (широта, долгота) или None
        """
        if not city_name:
            return None
        
        # 1. Сначала ищем в справочнике
        coords = self.get_coordinates_from_dict(city_name)
        if coords:
            return coords
        
        # 2. Затем пробуем LLM геокодинг
        coords = self.get_coordinates_via_llm(city_name, region_name)
        if coords:
            return coords
        
        logger.info(f"Не удалось получить координаты для '{city_name}' ни одним методом")
        return None
    
    def enrich_dataframe_with_coordinates(self, df: pd.DataFrame, 
                                        city_column: str = 'municipality',
                                        region_column: str = 'region_name') -> pd.DataFrame:
        """
        Обогащение DataFrame координатами городов.
        
        Args:
            df: DataFrame с данными
            city_column: Название колонки с городами
            region_column: Название колонки с регионами
            
        Returns:
            DataFrame с добавленными координатами
        """
        if df.empty:
            return df
        
        # Проверяем наличие необходимых колонок
        if city_column not in df.columns:
            logger.warning(f"Колонка '{city_column}' не найдена в DataFrame")
            return df
        
        enriched_df = df.copy()
        
        # Проверяем, есть ли уже координаты
        has_lat = any('lat' in col.lower() for col in df.columns)
        has_lon = any('lon' in col.lower() for col in df.columns)
        
        if has_lat and has_lon:
            logger.debug("DataFrame уже содержит координаты")
            return enriched_df
        
        # Добавляем колонки для координат
        enriched_df['lat'] = None
        enriched_df['lon'] = None
        
        # Обрабатываем каждый уникальный город
        unique_cities = df[city_column].dropna().unique()
        geocoded_count = 0
        
        for city in unique_cities:
            region = None
            if region_column in df.columns:
                # Берем первый регион для этого города
                region_series = df[df[city_column] == city][region_column].dropna()
                if not region_series.empty:
                    region = region_series.iloc[0]
            
            coords = self.geocode_city(city, region)
            if coords:
                lat, lon = coords
                mask = enriched_df[city_column] == city
                enriched_df.loc[mask, 'lat'] = lat
                enriched_df.loc[mask, 'lon'] = lon
                geocoded_count += 1
        
        logger.info(f"Добавлены координаты для {geocoded_count} из {len(unique_cities)} уникальных городов")
        return enriched_df


# Singleton instance
_geocoding_service = None

def get_geocoding_service() -> GeocodingService:
    """Получение глобального экземпляра сервиса геокодинга."""
    global _geocoding_service
    if _geocoding_service is None:
        _geocoding_service = GeocodingService()
    return _geocoding_service 