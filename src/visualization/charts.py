"""
Модуль для создания различных типов графиков и диаграмм.
Использует Plotly для интерактивных визуализаций.
"""

import logging
from typing import Dict, Any, Optional, List
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import time

from src.config.settings import (
    CHART_HEIGHT,
    CHART_WIDTH,
    COLOR_PALETTE
)

logger = logging.getLogger(__name__)


class ChartCreator:
    """Создатель графиков для визуализации данных индексов Сбербанка."""
    
    def __init__(self):
        """Инициализация создателя графиков."""
        self.default_config = {
            "height": CHART_HEIGHT,
            "width": CHART_WIDTH,
            "color_discrete_sequence": COLOR_PALETTE
        }
    
    def create_line_chart(self, config: Dict[str, Any]) -> go.Figure:
        """
        Создание линейного графика для временных рядов.
        
        Args:
            config: Конфигурация графика
            
        Returns:
            Plotly Figure объект
        """
        try:
            df = config["data"]
            x_col = config.get("x_column")
            y_col = config.get("y_column")
            color_col = config.get("color_column")
            title = config.get("title", "Динамика показателей")
            
            if not x_col or not y_col:
                raise ValueError("Не указаны колонки для осей X и Y")
            
            fig = px.line(
                df,
                x=x_col,
                y=y_col,
                color=color_col,
                title=title,
                **self.default_config
            )
            
            # Настройка осей
            fig.update_xaxes(title_text=x_col)
            fig.update_yaxes(title_text=y_col)
            
            # Настройка макета
            fig.update_layout(
                showlegend=True if color_col else False,
                hovermode='x unified'
            )
            
            logger.info(f"Создан линейный график: {title}")
            return fig
            
        except Exception as e:
            logger.error(f"Ошибка создания линейного графика: {e}")
            return self._create_error_chart(str(e))
    
    def create_bar_chart(self, config: Dict[str, Any]) -> go.Figure:
        """
        Создание столбчатой диаграммы для сравнения значений.
        
        Args:
            config: Конфигурация графика
            
        Returns:
            Plotly Figure объект
        """
        try:
            df = config["data"]
            x_col = config.get("x_column")
            y_col = config.get("y_column")
            color_col = config.get("color_column")
            title = config.get("title", "Сравнение показателей")
            
            if not x_col or not y_col:
                raise ValueError("Не указаны колонки для осей X и Y")
            
            # Сортируем данные по убыванию для лучшего отображения
            df_sorted = df.sort_values(y_col, ascending=False)
            
            fig = px.bar(
                df_sorted,
                x=x_col,
                y=y_col,
                color=color_col if color_col else x_col,
                title=title,
                **self.default_config
            )
            
            # Настройка осей
            fig.update_xaxes(title_text=x_col, tickangle=45)
            fig.update_yaxes(title_text=y_col)
            
            # Добавляем значения на столбцы
            fig.update_traces(texttemplate='%{y}', textposition='outside')
            
            logger.info(f"Создана столбчатая диаграмма: {title}")
            return fig
            
        except Exception as e:
            logger.error(f"Ошибка создания столбчатой диаграммы: {e}")
            return self._create_error_chart(str(e))
    
    def create_scatter_plot(self, config: Dict[str, Any]) -> go.Figure:
        """
        Создание точечной диаграммы для анализа корреляций.
        
        Args:
            config: Конфигурация графика
            
        Returns:
            Plotly Figure объект
        """
        try:
            df = config["data"]
            x_col = config.get("x_column")
            y_col = config.get("y_column")
            color_col = config.get("color_column")
            size_col = config.get("size_column")
            title = config.get("title", "Анализ зависимостей")
            
            if not x_col or not y_col:
                raise ValueError("Не указаны колонки для осей X и Y")
            
            fig = px.scatter(
                df,
                x=x_col,
                y=y_col,
                color=color_col,
                size=size_col,
                title=title,
                hover_data=df.columns.tolist(),
                **self.default_config
            )
            
            # Добавляем линию тренда
            fig.add_traces(
                px.scatter(df, x=x_col, y=y_col, trendline="ols").data[1:]
            )
            
            # Настройка осей
            fig.update_xaxes(title_text=x_col)
            fig.update_yaxes(title_text=y_col)
            
            logger.info(f"Создана точечная диаграмма: {title}")
            return fig
            
        except Exception as e:
            logger.error(f"Ошибка создания точечной диаграммы: {e}")
            return self._create_error_chart(str(e))
    
    def create_histogram(self, config: Dict[str, Any]) -> go.Figure:
        """
        Создание гистограммы для анализа распределения.
        
        Args:
            config: Конфигурация графика
            
        Returns:
            Plotly Figure объект
        """
        try:
            df = config["data"]
            x_col = config.get("x_column")
            color_col = config.get("color_column")
            title = config.get("title", "Распределение значений")
            
            if not x_col:
                raise ValueError("Не указана колонка для анализа")
            
            fig = px.histogram(
                df,
                x=x_col,
                color=color_col,
                title=title,
                nbins=20,
                **self.default_config
            )
            
            # Настройка осей
            fig.update_xaxes(title_text=x_col)
            fig.update_yaxes(title_text="Количество")
            
            logger.info(f"Создана гистограмма: {title}")
            return fig
            
        except Exception as e:
            logger.error(f"Ошибка создания гистограммы: {e}")
            return self._create_error_chart(str(e))
    
    def create_box_plot(self, config: Dict[str, Any]) -> go.Figure:
        """
        Создание коробчатой диаграммы для анализа распределения.
        
        Args:
            config: Конфигурация графика
            
        Returns:
            Plotly Figure объект
        """
        try:
            df = config["data"]
            x_col = config.get("x_column")
            y_col = config.get("y_column")
            title = config.get("title", "Анализ распределения")
            
            if not y_col:
                raise ValueError("Не указана колонка для анализа")
            
            fig = px.box(
                df,
                x=x_col,
                y=y_col,
                title=title,
                **self.default_config
            )
            
            # Настройка осей
            if x_col:
                fig.update_xaxes(title_text=x_col)
            fig.update_yaxes(title_text=y_col)
            
            logger.info(f"Создана коробчатая диаграмма: {title}")
            return fig
            
        except Exception as e:
            logger.error(f"Ошибка создания коробчатой диаграммы: {e}")
            return self._create_error_chart(str(e))
    
    def create_heatmap(self, config: Dict[str, Any]) -> go.Figure:
        """
        Создание тепловой карты для корреляционного анализа.
        
        Args:
            config: Конфигурация графика
            
        Returns:
            Plotly Figure объект
        """
        try:
            df = config["data"]
            title = config.get("title", "Корреляционная матрица")
            
            # Выбираем только числовые колонки
            numeric_df = df.select_dtypes(include=['number'])
            
            if numeric_df.empty:
                raise ValueError("Нет числовых данных для корреляционного анализа")
            
            # Вычисляем корреляционную матрицу
            corr_matrix = numeric_df.corr()
            
            fig = px.imshow(
                corr_matrix,
                title=title,
                color_continuous_scale='RdBu',
                aspect='auto',
                **self.default_config
            )
            
            # Добавляем значения корреляции на карту
            fig.update_traces(
                text=corr_matrix.round(2).values,
                texttemplate="%{text}",
                textfont={"size": 10}
            )
            
            logger.info(f"Создана тепловая карта: {title}")
            return fig
            
        except Exception as e:
            logger.error(f"Ошибка создания тепловой карты: {e}")
            return self._create_error_chart(str(e))
    
    def _create_error_chart(self, error_message: str) -> go.Figure:
        """
        Создание графика с сообщением об ошибке.
        
        Args:
            error_message: Сообщение об ошибке
            
        Returns:
            Plotly Figure с сообщением об ошибке
        """
        fig = go.Figure()
        fig.add_annotation(
            text=f"Ошибка создания графика:<br>{error_message}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="red")
        )
        fig.update_layout(
            title="Ошибка визуализации",
            height=self.default_config["height"],
            width=self.default_config["width"]
        )
        return fig
    
    def display_chart(self, chart_type: str, config: Dict[str, Any]) -> None:
        """
        Отображение графика в Streamlit.
        
        Args:
            chart_type: Тип графика
            config: Конфигурация графика
        """
        try:
            chart_methods = {
                "line": self.create_line_chart,
                "bar": self.create_bar_chart,
                "scatter": self.create_scatter_plot,
                "histogram": self.create_histogram,
                "box": self.create_box_plot,
                "heatmap": self.create_heatmap
            }
            
            if chart_type not in chart_methods:
                st.error(f"Неподдерживаемый тип графика: {chart_type}")
                return
            
            fig = chart_methods[chart_type](config)
            
            # Генерируем уникальный ключ на основе типа и времени
            unique_key = f"chart_{chart_type}_{int(time.time() * 1000000)}"
            
            st.plotly_chart(fig, use_container_width=True, key=unique_key)
            
        except Exception as e:
            logger.error(f"Ошибка отображения графика: {e}")
            st.error(f"Ошибка отображения графика: {e}")


# Singleton instance
chart_creator: Optional[ChartCreator] = None


def get_chart_creator() -> ChartCreator:
    """
    Получение единственного экземпляра ChartCreator.
    
    Returns:
        Экземпляр ChartCreator
    """
    global chart_creator
    if chart_creator is None:
        chart_creator = ChartCreator()
    return chart_creator 