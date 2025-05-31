"""
Модуль для создания интерактивных географических карт.
Использует Plotly для отображения данных на картах России.
"""

import logging
from typing import Dict, Any, Optional, List
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import json
from pathlib import Path
import time

from src.config.settings import (
    CHART_HEIGHT,
    CHART_WIDTH,
    MAP_DEFAULT_ZOOM,
    MAP_CENTER_LAT,
    MAP_CENTER_LON,
    COLOR_PALETTE
)

logger = logging.getLogger(__name__)


class MapCreator:
    """Создатель интерактивных карт для визуализации географических данных."""
    
    def __init__(self):
        """Инициализация создателя карт."""
        self.default_config = {
            "height": CHART_HEIGHT,
            "width": CHART_WIDTH
        }
        self.geo_data = self._load_geo_data()
    
    def _load_geo_data(self) -> Optional[Dict]:
        """
        Загрузка географических данных регионов.
        
        Returns:
            GeoJSON данные или None при ошибке
        """
        try:
            geo_file = Path("data/test_regions.geojson")
            if geo_file.exists():
                with open(geo_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                logger.warning("Файл с географическими данными не найден")
                return None
        except Exception as e:
            logger.error(f"Ошибка загрузки географических данных: {e}")
            return None
    
    def create_scatter_map(self, config: Dict[str, Any]) -> go.Figure:
        """
        Создание точечной карты с маркерами регионов.
        
        Args:
            config: Конфигурация карты
            
        Returns:
            Plotly Figure объект
        """
        try:
            df = config["data"]
            location_col = config.get("location_column", "region")
            value_col = config.get("value_column")
            title = config.get("title", "Географическое распределение")
            
            if not self.geo_data:
                return self._create_simple_scatter_map(config)
            
            # Создаем DataFrame с координатами
            geo_df = self._prepare_geo_dataframe(df, location_col, value_col)
            
            if geo_df.empty:
                raise ValueError("Не удалось сопоставить данные с координатами")
            
            fig = px.scatter_mapbox(
                geo_df,
                lat="lat",
                lon="lon",
                size=value_col if value_col else None,
                color=value_col if value_col else location_col,
                hover_name=location_col,
                hover_data={col: True for col in geo_df.columns if col not in ['lat', 'lon']},
                title=title,
                mapbox_style="open-street-map",
                zoom=MAP_DEFAULT_ZOOM,
                center={"lat": MAP_CENTER_LAT, "lon": MAP_CENTER_LON},
                **self.default_config
            )
            
            logger.info(f"Создана точечная карта: {title}")
            return fig
            
        except Exception as e:
            logger.error(f"Ошибка создания точечной карты: {e}")
            return self._create_error_map(str(e))
    
    def create_choropleth_map(self, config: Dict[str, Any]) -> go.Figure:
        """
        Создание хороплетной карты (заливка регионов по значениям).
        
        Args:
            config: Конфигурация карты
            
        Returns:
            Plotly Figure объект
        """
        try:
            df = config["data"]
            location_col = config.get("location_column", "region")
            value_col = config.get("value_column")
            title = config.get("title", "Тематическая карта")
            
            if not value_col:
                raise ValueError("Не указана колонка со значениями для раскраски")
            
            # Для простоты используем scatter map с размером маркеров
            # В реальном проекте здесь была бы полноценная хороплетная карта
            return self.create_scatter_map(config)
            
        except Exception as e:
            logger.error(f"Ошибка создания хороплетной карты: {e}")
            return self._create_error_map(str(e))
    
    def create_bubble_map(self, config: Dict[str, Any]) -> go.Figure:
        """
        Создание пузырьковой карты с размером пузырьков по значениям.
        
        Args:
            config: Конфигурация карты
            
        Returns:
            Plotly Figure объект
        """
        try:
            df = config["data"]
            location_col = config.get("location_column", "region")
            value_col = config.get("value_column")
            color_col = config.get("color_column")
            title = config.get("title", "Пузырьковая карта")
            
            geo_df = self._prepare_geo_dataframe(df, location_col, value_col)
            
            if geo_df.empty:
                raise ValueError("Не удалось сопоставить данные с координатами")
            
            # Нормализуем размеры пузырьков
            if value_col and value_col in geo_df.columns:
                max_val = geo_df[value_col].max()
                min_val = geo_df[value_col].min()
                geo_df['bubble_size'] = 10 + (geo_df[value_col] - min_val) / (max_val - min_val) * 40
            else:
                geo_df['bubble_size'] = 20
            
            fig = go.Figure()
            
            # Добавляем пузырьки
            fig.add_trace(go.Scattermapbox(
                lat=geo_df["lat"],
                lon=geo_df["lon"],
                mode='markers',
                marker=dict(
                    size=geo_df['bubble_size'],
                    color=geo_df[color_col] if color_col and color_col in geo_df.columns else geo_df[value_col] if value_col else 'blue',
                    colorscale='Viridis',
                    showscale=True,
                    sizemode='diameter'
                ),
                text=geo_df[location_col],
                hovertemplate='<b>%{text}</b><br>' + 
                            f'{value_col}: %{{marker.color}}<extra></extra>' if value_col else '<b>%{text}</b><extra></extra>',
                name=""
            ))
            
            fig.update_layout(
                title=title,
                mapbox=dict(
                    style="open-street-map",
                    zoom=MAP_DEFAULT_ZOOM,
                    center={"lat": MAP_CENTER_LAT, "lon": MAP_CENTER_LON}
                ),
                **self.default_config
            )
            
            logger.info(f"Создана пузырьковая карта: {title}")
            return fig
            
        except Exception as e:
            logger.error(f"Ошибка создания пузырьковой карты: {e}")
            return self._create_error_map(str(e))
    
    def _prepare_geo_dataframe(self, df: pd.DataFrame, location_col: str, value_col: Optional[str]) -> pd.DataFrame:
        """
        Подготовка DataFrame с географическими координатами.
        
        Args:
            df: Исходный DataFrame
            location_col: Колонка с названиями регионов
            value_col: Колонка со значениями
            
        Returns:
            DataFrame с координатами
        """
        try:
            if not self.geo_data:
                return pd.DataFrame()
            
            # Создаем словарь координат из GeoJSON
            coords_dict = {}
            for feature in self.geo_data.get('features', []):
                region_name = feature['properties']['region']
                coords = feature['geometry']['coordinates']
                coords_dict[region_name] = {
                    'lat': coords[1],
                    'lon': coords[0]
                }
            
            # Объединяем с исходными данными
            geo_rows = []
            for _, row in df.iterrows():
                region = row[location_col]
                if region in coords_dict:
                    geo_row = row.to_dict()
                    geo_row.update(coords_dict[region])
                    geo_rows.append(geo_row)
            
            return pd.DataFrame(geo_rows)
            
        except Exception as e:
            logger.error(f"Ошибка подготовки географических данных: {e}")
            return pd.DataFrame()
    
    def _create_simple_scatter_map(self, config: Dict[str, Any]) -> go.Figure:
        """
        Создание простой карты без географических данных.
        
        Args:
            config: Конфигурация карты
            
        Returns:
            Простая карта России
        """
        fig = go.Figure()
        
        fig.add_trace(go.Scattermapbox(
            lat=[MAP_CENTER_LAT],
            lon=[MAP_CENTER_LON],
            mode='markers',
            marker=dict(size=10, color='red'),
            text=["Центр России"],
            name=""
        ))
        
        fig.update_layout(
            title="Карта России (географические данные недоступны)",
            mapbox=dict(
                style="open-street-map",
                zoom=MAP_DEFAULT_ZOOM,
                center={"lat": MAP_CENTER_LAT, "lon": MAP_CENTER_LON}
            ),
            **self.default_config
        )
        
        return fig
    
    def _create_error_map(self, error_message: str) -> go.Figure:
        """
        Создание карты с сообщением об ошибке.
        
        Args:
            error_message: Сообщение об ошибке
            
        Returns:
            Карта с сообщением об ошибке
        """
        fig = go.Figure()
        
        fig.add_annotation(
            text=f"Ошибка создания карты:<br>{error_message}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="red")
        )
        
        fig.update_layout(
            title="Ошибка создания карты",
            **self.default_config
        )
        
        return fig
    
    def display_map(self, map_type: str, config: Dict[str, Any]) -> None:
        """
        Отображение карты в Streamlit.
        
        Args:
            map_type: Тип карты
            config: Конфигурация карты
        """
        try:
            map_methods = {
                "scatter": self.create_scatter_map,
                "choropleth": self.create_choropleth_map,
                "bubble": self.create_bubble_map
            }
            
            if map_type not in map_methods:
                st.error(f"Неподдерживаемый тип карты: {map_type}")
                return
            
            fig = map_methods[map_type](config)
            
            # Генерируем уникальный ключ для карты
            unique_key = f"map_{map_type}_{int(time.time() * 1000000)}"
            
            st.plotly_chart(fig, use_container_width=True, key=unique_key)
            
        except Exception as e:
            logger.error(f"Ошибка отображения карты: {e}")
            st.error(f"Ошибка отображения карты: {e}")


# Singleton instance
map_creator: Optional[MapCreator] = None


def get_map_creator() -> MapCreator:
    """
    Получение единственного экземпляра MapCreator.
    
    Returns:
        Экземпляр MapCreator
    """
    global map_creator
    if map_creator is None:
        map_creator = MapCreator()
    return map_creator 