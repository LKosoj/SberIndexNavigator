"""
Модуль для экспорта результатов анализа в PDF формат.
Создание красивых PDF отчетов с вопросами, ответами и визуализациями.
"""

import io
import base64
import os
from datetime import datetime
from typing import Dict, Any, Optional, List
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import logging

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

logger = logging.getLogger(__name__)


def register_cyrillic_fonts():
    """Регистрация шрифтов с поддержкой кириллицы."""
    try:
        # Проверяем, уже ли зарегистрированы наши шрифты
        registered_font_names = [f.fontName for f in pdfmetrics._fonts.values()]
        if 'CyrillicSans' in registered_font_names:
            logger.info("Шрифты уже зарегистрированы")
            return {
                'sans': 'CyrillicSans',
                'serif': 'CyrillicSerif' if 'CyrillicSerif' in registered_font_names else 'Times-Roman',
                'mono': 'Courier'
            }
        
        # Пытаемся найти и зарегистрировать системные шрифты
        system_fonts = [
            # macOS шрифты
            '/Library/Fonts/Arial.ttf',
            '/System/Library/Fonts/Helvetica.ttc',
            '/System/Library/Fonts/Times.ttc',
            '/Library/Fonts/Times New Roman.ttf',
            
            # Windows шрифты
            'C:/Windows/Fonts/arial.ttf',
            'C:/Windows/Fonts/times.ttf',
            'C:/Windows/Fonts/calibri.ttf',
            
            # Linux шрифты
            '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
            '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
        ]
        
        fonts_registered = []
        
        # Регистрируем первый найденный шрифт для каждого типа
        for font_path in system_fonts:
            if os.path.exists(font_path):
                try:
                    if 'arial' in font_path.lower() or 'helvetica' in font_path.lower():
                        pdfmetrics.registerFont(TTFont('CyrillicSans', font_path))
                        fonts_registered.append('CyrillicSans')
                        logger.info(f"Зарегистрирован шрифт CyrillicSans: {font_path}")
                    
                    elif 'times' in font_path.lower():
                        pdfmetrics.registerFont(TTFont('CyrillicSerif', font_path))
                        fonts_registered.append('CyrillicSerif')
                        logger.info(f"Зарегистрирован шрифт CyrillicSerif: {font_path}")
                    
                    elif 'dejavu' in font_path.lower() or 'liberation' in font_path.lower():
                        pdfmetrics.registerFont(TTFont('CyrillicSans', font_path))
                        fonts_registered.append('CyrillicSans')
                        logger.info(f"Зарегистрирован шрифт CyrillicSans: {font_path}")
                            
                except Exception as e:
                    logger.warning(f"Не удалось зарегистрировать шрифт {font_path}: {e}")
                    continue
                
                # Если уже зарегистрировали оба типа шрифтов, выходим
                if len(fonts_registered) >= 2:
                    break
        
        # Если не удалось найти системные шрифты, используем встроенные шрифты
        if not fonts_registered:
            logger.warning("Системные шрифты не найдены, используем встроенные шрифты")
            return {'sans': 'Helvetica', 'serif': 'Times-Roman', 'mono': 'Courier'}
        
        # Возвращаем доступные шрифты
        available_fonts = {
            'sans': 'CyrillicSans' if 'CyrillicSans' in fonts_registered else 'Helvetica',
            'serif': 'CyrillicSerif' if 'CyrillicSerif' in fonts_registered else 'Times-Roman',
            'mono': 'Courier'
        }
        
        logger.info(f"Доступные шрифты: {available_fonts}")
        return available_fonts
        
    except Exception as e:
        logger.error(f"Ошибка регистрации шрифтов: {e}")
        # Возвращаем встроенные шрифты как fallback
        return {'sans': 'Helvetica', 'serif': 'Times-Roman', 'mono': 'Courier'}


class PDFReportGenerator:
    """Генератор PDF отчетов для результатов анализа."""
    
    def __init__(self):
        """Инициализация генератора PDF."""
        self.styles = getSampleStyleSheet()
        
        # Регистрируем шрифты с поддержкой кириллицы
        self.fonts = register_cyrillic_fonts()
        
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Настройка пользовательских стилей для PDF."""
        # Стиль для заголовка
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Title'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#1f4e79'),
            fontName=self.fonts['sans']
        )
        
        # Стиль для подзаголовков
        self.heading_style = ParagraphStyle(
            'CustomHeading',
            parent=self.styles['Heading1'],
            fontSize=16,
            spaceAfter=12,
            spaceBefore=20,
            textColor=colors.HexColor('#2e75b6'),
            fontName=self.fonts['sans']
        )
        
        # Стиль для вопросов
        self.question_style = ParagraphStyle(
            'QuestionStyle',
            parent=self.styles['Normal'],
            fontSize=14,
            spaceAfter=10,
            spaceBefore=10,
            textColor=colors.HexColor('#d73527'),
            fontName=self.fonts['sans']
        )
        
        # Стиль для ответов
        self.answer_style = ParagraphStyle(
            'AnswerStyle',
            parent=self.styles['Normal'],
            fontSize=12,
            spaceAfter=15,
            alignment=TA_JUSTIFY,
            fontName=self.fonts['sans']
        )
        
        # Стиль для метаданных
        self.meta_style = ParagraphStyle(
            'MetaStyle',
            parent=self.styles['Normal'],
            fontSize=10,
            textColor=colors.grey,
            spaceAfter=5,
            fontName=self.fonts['sans']
        )
        
        # Стиль для кода
        self.code_style = ParagraphStyle(
            'CodeStyle',
            parent=self.styles['Normal'],
            fontSize=10,
            fontName=self.fonts['mono'],
            leftIndent=20,
            spaceAfter=10,
            textColor=colors.HexColor('#333333')
        )
    
    def generate_report(
        self,
        question: str,
        answer: str,
        data: Optional[pd.DataFrame] = None,
        sql_query: Optional[str] = None,
        visualization_config: Optional[Dict[str, Any]] = None,
        visualization_configs: Optional[List[Dict[str, Any]]] = None,
        analysis_result: Optional[Dict[str, Any]] = None
    ) -> bytes:
        """
        Генерация PDF отчета.
        
        Args:
            question: Вопрос пользователя
            answer: Ответ системы
            data: Данные таблицы
            sql_query: SQL запрос
            visualization_config: Конфигурация визуализации (одна)
            visualization_configs: Список конфигураций визуализации (множественные)
            analysis_result: Результаты анализа данных
            
        Returns:
            Байты PDF файла
        """
        try:
            # Создаем буфер для PDF
            buffer = io.BytesIO()
            
            # Создаем документ
            doc = SimpleDocTemplate(
                buffer,
                pagesize=A4,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=18
            )
            
            # Элементы документа
            story = []
            
            # Заголовок
            story.append(Paragraph("SberIndexNavigator", self.title_style))
            story.append(Paragraph("Отчет по анализу данных", self.heading_style))
            story.append(Spacer(1, 20))
            
            # Метаданные
            timestamp = datetime.now().strftime("%d.%m.%Y %H:%M:%S")
            story.append(Paragraph(f"Дата создания: {timestamp}", self.meta_style))
            story.append(Spacer(1, 20))
            
            # Вопрос
            story.append(Paragraph("Вопрос:", self.heading_style))
            story.append(Paragraph(question, self.question_style))
            story.append(Spacer(1, 15))
            
            # Ответ
            story.append(Paragraph("Анализ:", self.heading_style))
            story.append(Paragraph(answer, self.answer_style))
            story.append(Spacer(1, 15))
            
            # SQL запрос (если есть)
            if sql_query:
                story.append(Paragraph("SQL запрос:", self.heading_style))
                sql_paragraph = Paragraph(sql_query, self.code_style)
                story.append(sql_paragraph)
                story.append(Spacer(1, 15))
            
            # Таблица данных (если есть)
            if data is not None and not data.empty:
                story.append(Paragraph("Данные:", self.heading_style))
                table = self._create_data_table(data)
                story.append(table)
                story.append(Spacer(1, 15))
            
            # Результаты анализа (если есть)
            if analysis_result and analysis_result.get("success", False):
                story.append(Paragraph("Интеллектуальный анализ:", self.heading_style))
                
                # AI инсайты
                ai_insights = analysis_result.get("ai_insights", {})
                if ai_insights and "error" not in ai_insights:
                    # Ключевые выводы
                    key_insights = ai_insights.get("key_insights", [])
                    if key_insights:
                        story.append(Paragraph("Ключевые выводы:", self.heading_style))
                        for i, insight in enumerate(key_insights, 1):
                            story.append(Paragraph(f"• {insight}", self.answer_style))
                        story.append(Spacer(1, 10))
                    
                    # Статистические данные
                    statistics = ai_insights.get("statistics", {})
                    if statistics:
                        story.append(Paragraph("Статистические показатели:", self.heading_style))
                        for key, value in statistics.items():
                            if isinstance(value, (int, float)):
                                story.append(Paragraph(f"• {key}: {value:.2f}", self.answer_style))
                            else:
                                story.append(Paragraph(f"• {key}: {value}", self.answer_style))
                        story.append(Spacer(1, 10))
                
                # Рекомендации
                recommendations = analysis_result.get("recommendations", [])
                if recommendations:
                    story.append(Paragraph("Рекомендации:", self.heading_style))
                    for i, rec in enumerate(recommendations, 1):
                        story.append(Paragraph(f"{i}. {rec}", self.answer_style))
                    story.append(Spacer(1, 15))
            
            # Визуализации (если есть)
            visualizations_to_process = []
            
            # Сначала добавляем одиночную визуализацию (для обратной совместимости)
            if visualization_config:
                visualizations_to_process.append(visualization_config)
            
            # Затем добавляем множественные визуализации
            if visualization_configs:
                visualizations_to_process.extend(visualization_configs)
            
            if visualizations_to_process:
                story.append(PageBreak())
                story.append(Paragraph("Визуализации:", self.heading_style))
                
                for i, viz_config in enumerate(visualizations_to_process, 1):
                    if viz_config:
                        # Добавляем заголовок для каждой визуализации
                        if len(visualizations_to_process) > 1:
                            viz_title = viz_config.get('title', f'График {i}')
                            story.append(Paragraph(f"{i}. {viz_title}", self.heading_style))
                        
                        chart_image = self._create_chart_image(viz_config)
                        if chart_image:
                            story.append(chart_image)
                            story.append(Spacer(1, 15))
                        
                        # Разделитель между графиками (кроме последнего)
                        if i < len(visualizations_to_process):
                            story.append(Spacer(1, 10))
            
            # Футер
            story.append(Spacer(1, 30))
            story.append(Paragraph(
                "Отчет сгенерирован автоматически системой SberIndexNavigator",
                self.meta_style
            ))
            
            # Строим PDF
            doc.build(story)
            
            # Возвращаем содержимое
            buffer.seek(0)
            return buffer.getvalue()
            
        except Exception as e:
            logger.error(f"Ошибка генерации PDF отчета: {e}")
            raise
    
    def _create_data_table(self, data: pd.DataFrame) -> Table:
        """
        Создание таблицы для отображения данных.
        
        Args:
            data: DataFrame с данными
            
        Returns:
            Table объект для ReportLab
        """
        try:
            # Ограничиваем количество строк для отображения
            max_rows = 50
            if len(data) > max_rows:
                display_data = data.head(max_rows)
                truncated = True
            else:
                display_data = data
                truncated = False
            
            # Подготавливаем данные для таблицы
            table_data = []
            
            # Заголовки
            headers = list(display_data.columns)
            table_data.append(headers)
            
            # Данные
            for _, row in display_data.iterrows():
                formatted_row = []
                for value in row:
                    if pd.isna(value):
                        formatted_row.append("-")
                    elif isinstance(value, float):
                        formatted_row.append(f"{value:.2f}")
                    else:
                        formatted_row.append(str(value))
                table_data.append(formatted_row)
            
            # Создаем таблицу
            table = Table(table_data)
            
            # Стилизация таблицы с правильными шрифтами
            style = TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4472C4')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), self.fonts['sans']),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('FONTNAME', (0, 1), (-1, -1), self.fonts['sans']),
                ('FONTSIZE', (0, 1), (-1, -1), 8),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
            ])
            
            table.setStyle(style)
            
            # Добавляем примечание об ограничении строк
            if truncated:
                note = Paragraph(
                    f"Показаны первые {max_rows} строк из {len(data)} общих строк.",
                    self.meta_style
                )
                return [table, Spacer(1, 10), note]
            
            return table
            
        except Exception as e:
            logger.error(f"Ошибка создания таблицы данных: {e}")
            return Paragraph("Ошибка отображения данных", self.answer_style)
    
    def _create_chart_image(self, config: Dict[str, Any]) -> Optional[Image]:
        """
        Создание изображения графика для PDF.
        
        Args:
            config: Конфигурация визуализации
            
        Returns:
            Image объект для ReportLab или None
        """
        try:
            chart_type = config.get("type", "table")
            
            if chart_type == "table":
                return None
            
            # Создаем простой график на основе данных
            data = config.get("data")
            if data is None or data.empty:
                return None
            
            # Создаем Plotly график
            fig = self._create_plotly_chart(config)
            
            if fig is None:
                return None
            
            # Конвертируем в изображение
            img_bytes = pio.to_image(fig, format="png", width=600, height=400)
            
            # Создаем Image объект
            img_buffer = io.BytesIO(img_bytes)
            img = Image(img_buffer, width=6*inch, height=4*inch)
            
            return img
            
        except Exception as e:
            logger.error(f"Ошибка создания изображения графика: {e}")
            return None
    
    def _create_plotly_chart(self, config: Dict[str, Any]) -> Optional[go.Figure]:
        """
        Создание Plotly графика на основе конфигурации.
        
        Args:
            config: Конфигурация визуализации
            
        Returns:
            Plotly Figure или None
        """
        try:
            chart_type = config.get("type", "bar")
            data = config.get("data")
            title = config.get("title", "График")
            
            if data is None or data.empty:
                return None
            
            fig = go.Figure()
            
            if chart_type == "bar":
                x_col = config.get("x_column") or data.columns[0]
                y_col = config.get("y_column") or data.columns[1]
                
                fig.add_trace(go.Bar(
                    x=data[x_col],
                    y=data[y_col],
                    name=y_col
                ))
                
            elif chart_type == "line":
                x_col = config.get("x_column") or data.columns[0]
                y_col = config.get("y_column") or data.columns[1]
                
                fig.add_trace(go.Scatter(
                    x=data[x_col],
                    y=data[y_col],
                    mode='lines+markers',
                    name=y_col
                ))
                
            elif chart_type == "pie":
                labels_col = config.get("labels_column") or data.columns[0]
                values_col = config.get("values_column") or data.columns[1]
                
                fig.add_trace(go.Pie(
                    labels=data[labels_col],
                    values=data[values_col]
                ))
            
            else:
                # По умолчанию bar chart
                x_col = data.columns[0]
                y_col = data.columns[1] if len(data.columns) > 1 else data.columns[0]
                
                fig.add_trace(go.Bar(
                    x=data[x_col],
                    y=data[y_col]
                ))
            
            # Настройка внешнего вида с поддержкой кириллицы
            fig.update_layout(
                title=title,
                title_font_size=16,
                font_size=12,
                height=400,
                width=600,
                margin=dict(l=50, r=50, t=80, b=50),
                font_family="Arial, sans-serif"  # Шрифт с поддержкой кириллицы
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Ошибка создания Plotly графика: {e}")
            return None


# Глобальный экземпляр генератора PDF для кэширования
_pdf_generator = None

def get_pdf_generator():
    """Получение кэшированного экземпляра PDFReportGenerator."""
    global _pdf_generator
    if _pdf_generator is None:
        _pdf_generator = PDFReportGenerator()
    return _pdf_generator


def generate_qa_pdf(
    question: str,
    answer: str,
    data: Optional[pd.DataFrame] = None,
    sql_query: Optional[str] = None,
    visualization_config: Optional[Dict[str, Any]] = None,
    visualization_configs: Optional[List[Dict[str, Any]]] = None,
    analysis_result: Optional[Dict[str, Any]] = None
) -> bytes:
    """
    Быстрая функция для генерации PDF отчета вопрос-ответ.
    
    Args:
        question: Вопрос пользователя
        answer: Ответ системы
        data: Данные таблицы
        sql_query: SQL запрос
        visualization_config: Конфигурация визуализации (одна)
        visualization_configs: Список конфигураций визуализации (множественные)
        analysis_result: Результаты анализа данных
        
    Returns:
        Байты PDF файла
    """
    generator = get_pdf_generator()
    return generator.generate_report(
        question=question,
        answer=answer,
        data=data,
        sql_query=sql_query,
        visualization_config=visualization_config,
        visualization_configs=visualization_configs,
        analysis_result=analysis_result
    )


def generate_full_history_pdf(qa_pairs: List[Dict[str, Any]]) -> bytes:
    """
    Генерация PDF отчета для полной истории с отдельными секциями для каждого Q&A.
    
    Args:
        qa_pairs: Список словарей с парами вопрос-ответ
        
    Returns:
        Байты PDF файла
    """
    try:
        generator = get_pdf_generator()
        
        # Создаем буфер для PDF
        buffer = io.BytesIO()
        
        # Создаем документ
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18
        )
        
        # Элементы документа
        story = []
        
        # Общий заголовок
        story.append(Paragraph("SberIndexNavigator", generator.title_style))
        story.append(Paragraph("Полная история анализа данных", generator.heading_style))
        story.append(Spacer(1, 20))
        
        # Метаданные
        timestamp = datetime.now().strftime("%d.%m.%Y %H:%M:%S")
        story.append(Paragraph(f"Дата создания: {timestamp}", generator.meta_style))
        story.append(Paragraph(f"Количество вопросов: {len(qa_pairs)}", generator.meta_style))
        story.append(Spacer(1, 30))
        
        # Создаем отдельную секцию для каждого Q&A
        for i, pair in enumerate(qa_pairs, 1):
            # Заголовок секции
            story.append(Paragraph(f"Вопрос {i} из {len(qa_pairs)}", generator.heading_style))
            story.append(Spacer(1, 10))
            
            # Вопрос
            story.append(Paragraph("Вопрос:", generator.heading_style))
            story.append(Paragraph(pair['question'], generator.question_style))
            story.append(Spacer(1, 15))
            
            # Ответ
            story.append(Paragraph("Ответ:", generator.heading_style))
            story.append(Paragraph(pair['answer'], generator.answer_style))
            story.append(Spacer(1, 15))
            
            # SQL запрос (если есть)
            if pair['sql_query']:
                story.append(Paragraph("SQL запрос:", generator.heading_style))
                story.append(Paragraph(pair['sql_query'], generator.code_style))
                story.append(Spacer(1, 15))
            
            # Таблица данных (если есть)
            if pair['data'] is not None and not pair['data'].empty:
                story.append(Paragraph("Данные:", generator.heading_style))
                table = generator._create_data_table(pair['data'])
                story.append(table)
                story.append(Spacer(1, 15))
            
            # Визуализация (если есть)
            if pair['visualization']:
                story.append(Paragraph("Визуализация:", generator.heading_style))
                chart_image = generator._create_chart_image(pair['visualization'])
                if chart_image:
                    story.append(chart_image)
                    story.append(Spacer(1, 15))
            
            # Результаты анализа (если есть)
            analysis_result = pair.get('analysis')
            if analysis_result and analysis_result.get("success", False):
                story.append(Paragraph("Интеллектуальный анализ:", generator.heading_style))
                
                # AI инсайты
                ai_insights = analysis_result.get("ai_insights", {})
                if ai_insights and "error" not in ai_insights:
                    # Ключевые выводы
                    key_insights = ai_insights.get("key_insights", [])
                    if key_insights:
                        story.append(Paragraph("Ключевые выводы:", generator.heading_style))
                        for insight in key_insights:
                            story.append(Paragraph(f"• {insight}", generator.answer_style))
                        story.append(Spacer(1, 10))
                    
                    # Статистические данные
                    statistics = ai_insights.get("statistics", {})
                    if statistics:
                        story.append(Paragraph("Статистические показатели:", generator.heading_style))
                        for key, value in statistics.items():
                            if isinstance(value, (int, float)):
                                story.append(Paragraph(f"• {key}: {value:.2f}", generator.answer_style))
                            else:
                                story.append(Paragraph(f"• {key}: {value}", generator.answer_style))
                        story.append(Spacer(1, 10))
                
                # Рекомендации
                recommendations = analysis_result.get("recommendations", [])
                if recommendations:
                    story.append(Paragraph("Рекомендации:", generator.heading_style))
                    for j, rec in enumerate(recommendations, 1):
                        story.append(Paragraph(f"{j}. {rec}", generator.answer_style))
                    story.append(Spacer(1, 15))
            
            # Разделитель между секциями (кроме последней)
            if i < len(qa_pairs):
                story.append(Spacer(1, 30))
                story.append(Paragraph("═" * 80, generator.meta_style))
                story.append(PageBreak())
        
        # Футер
        story.append(Spacer(1, 30))
        story.append(Paragraph(
            "Отчет сгенерирован автоматически системой SberIndexNavigator",
            generator.meta_style
        ))
        
        # Строим PDF
        doc.build(story)
        
        # Возвращаем содержимое
        buffer.seek(0)
        return buffer.getvalue()
        
    except Exception as e:
        logger.error(f"Ошибка генерации полного PDF отчета: {e}")
        raise 