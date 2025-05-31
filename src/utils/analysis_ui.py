"""
Утилиты для отображения результатов анализа данных в Streamlit.
Компоненты для красивого представления insights и рекомендаций.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class AnalysisUIRenderer:
    """Рендерер для отображения результатов анализа данных."""
    
    def __init__(self):
        """Инициализация рендерера."""
        # Эмодзи для разных типов анализа
        self.analysis_icons = {
            "descriptive": "📊",
            "comparative": "🏆", 
            "correlation": "🔗",
            "forecasting": "🔮",
            "error": "❌"
        }
        
        # Цветовая схема для разных элементов
        self.colors = {
            "insights": "#1f77b4",
            "recommendations": "#2ca02c", 
            "risks": "#d62728",
            "opportunities": "#ff7f0e",
            "leaders": "#2ca02c",
            "outsiders": "#d62728"
        }
    
    def render_analysis_results(self, analysis_result: Dict[str, Any]) -> None:
        """
        Отображение полных результатов анализа.
        
        Args:
            analysis_result: Результаты анализа от AnalysisAgent
        """
        try:
            if not analysis_result.get("success", False):
                st.error(f"❌ Ошибка анализа: {analysis_result.get('error', 'Неизвестная ошибка')}")
                return
            
            analysis_type = analysis_result.get("analysis_type", "descriptive")
            ai_insights = analysis_result.get("ai_insights", {})
            
            # Заголовок с иконкой типа анализа
            icon = self.analysis_icons.get(analysis_type, "📊")
            analysis_name = self._get_analysis_name(analysis_type)
            
            st.markdown(f"## {icon} {analysis_name}")
            
            # Отображение AI инсайтов
            if ai_insights and "error" not in ai_insights:
                self._render_ai_insights(ai_insights, analysis_type)
            
            # Отображение статистического анализа
            if "statistical_analysis" in analysis_result:
                self._render_statistical_summary(analysis_result["statistical_analysis"])
            
            # Отображение качества данных
            if "data_quality" in analysis_result:
                self._render_data_quality(analysis_result["data_quality"])
            
            # Отображение итоговых рекомендаций
            if "recommendations" in analysis_result and analysis_result["recommendations"]:
                self._render_recommendations(analysis_result["recommendations"])
            
        except Exception as e:
            logger.error(f"Ошибка отображения результатов анализа: {e}")
            st.error(f"Ошибка отображения: {e}")
    
    def _get_analysis_name(self, analysis_type: str) -> str:
        """Получение человеко-читаемого названия типа анализа."""
        names = {
            "descriptive": "Описательный анализ данных",
            "comparative": "Сравнительный анализ",
            "correlation": "Корреляционный анализ",
            "forecasting": "Прогнозный анализ"
        }
        return names.get(analysis_type, "Анализ данных")
    
    def _render_ai_insights(self, ai_insights: Dict[str, Any], analysis_type: str) -> None:
        """Отображение AI инсайтов в зависимости от типа анализа."""
        if analysis_type == "descriptive":
            self._render_descriptive_insights(ai_insights)
        elif analysis_type == "comparative":
            self._render_comparative_insights(ai_insights)
        elif analysis_type == "correlation":
            self._render_correlation_insights(ai_insights)
        elif analysis_type == "forecasting":
            self._render_forecasting_insights(ai_insights)
    
    def _render_descriptive_insights(self, insights: Dict[str, Any]) -> None:
        """Отображение результатов описательного анализа."""
        # Ключевые инсайты
        if "key_insights" in insights and insights["key_insights"]:
            st.subheader("🔍 Ключевые инсайты")
            for insight in insights["key_insights"]:
                st.markdown(f"• {insight}")
        
        # Статистическое резюме
        if "statistical_summary" in insights:
            st.subheader("📈 Статистическое резюме")
            summary = insights["statistical_summary"]
            
            col1, col2 = st.columns(2)
            with col1:
                if "mean" in summary:
                    st.info(f"**Среднее значение**: {summary['mean']}")
                if "median" in summary:
                    st.info(f"**Медиана**: {summary['median']}")
            
            with col2:
                if "std" in summary:
                    st.info(f"**Стандартное отклонение**: {summary['std']}")
                if "min_max" in summary:
                    st.info(f"**Диапазон**: {summary['min_max']}")
        
        # Тренды
        if "trends" in insights and insights["trends"]:
            st.subheader("📊 Выявленные тренды")
            for trend in insights["trends"]:
                st.success(f"📈 {trend}")
        
        # Аномалии
        if "anomalies" in insights and insights["anomalies"]:
            st.subheader("⚠️ Выявленные аномалии")
            for anomaly in insights["anomalies"]:
                st.warning(f"🚨 {anomaly}")
        
        # Бизнес-импликации
        if "business_implications" in insights and insights["business_implications"]:
            st.subheader("💼 Бизнес-импликации")
            for implication in insights["business_implications"]:
                st.markdown(f"💡 {implication}")
    
    def _render_comparative_insights(self, insights: Dict[str, Any]) -> None:
        """Отображение результатов сравнительного анализа."""
        col1, col2 = st.columns(2)
        
        # Лидеры
        with col1:
            if "leaders" in insights and insights["leaders"]:
                st.subheader("🏆 Лидеры")
                for leader in insights["leaders"]:
                    st.success(f"🥇 {leader}")
        
        # Аутсайдеры
        with col2:
            if "outsiders" in insights and insights["outsiders"]:
                st.subheader("📉 Аутсайдеры")
                for outsider in insights["outsiders"]:
                    st.error(f"📊 {outsider}")
        
        # Ключевые различия
        if "key_differences" in insights and insights["key_differences"]:
            st.subheader("🔍 Ключевые различия")
            for diff in insights["key_differences"]:
                st.info(f"🔄 {diff}")
        
        # Факторы успеха
        if "success_factors" in insights and insights["success_factors"]:
            st.subheader("⭐ Факторы успеха")
            for factor in insights["success_factors"]:
                st.success(f"✅ {factor}")
        
        # Области для улучшения
        if "improvement_areas" in insights and insights["improvement_areas"]:
            st.subheader("🎯 Области для улучшения")
            for area in insights["improvement_areas"]:
                st.warning(f"🔧 {area}")
    
    def _render_correlation_insights(self, insights: Dict[str, Any]) -> None:
        """Отображение результатов корреляционного анализа."""
        # Сильные корреляции
        if "strong_correlations" in insights and insights["strong_correlations"]:
            st.subheader("🔗 Сильные корреляции")
            
            for corr in insights["strong_correlations"]:
                variables = " ↔ ".join(corr.get("variables", []))
                correlation_value = corr.get("correlation", 0)
                interpretation = corr.get("interpretation", "")
                
                # Цвет в зависимости от силы корреляции
                if abs(correlation_value) >= 0.8:
                    st.success(f"**{variables}** (r = {correlation_value:.3f})")
                else:
                    st.info(f"**{variables}** (r = {correlation_value:.3f})")
                
                if interpretation:
                    st.markdown(f"  💡 {interpretation}")
        
        # Причинно-следственные связи
        if "causal_relationships" in insights and insights["causal_relationships"]:
            st.subheader("🎯 Причинно-следственные связи")
            for relationship in insights["causal_relationships"]:
                st.info(f"🔄 {relationship}")
        
        # Неожиданные находки
        if "unexpected_findings" in insights and insights["unexpected_findings"]:
            st.subheader("🤔 Неожиданные находки")
            for finding in insights["unexpected_findings"]:
                st.warning(f"❗ {finding}")
        
        # Практические инсайты
        if "actionable_insights" in insights and insights["actionable_insights"]:
            st.subheader("⚡ Практические инсайты")
            for insight in insights["actionable_insights"]:
                st.success(f"💡 {insight}")
    
    def _render_forecasting_insights(self, insights: Dict[str, Any]) -> None:
        """Отображение результатов прогнозного анализа."""
        # Будущие тренды
        if "future_trends" in insights and insights["future_trends"]:
            st.subheader("🔮 Прогнозируемые тренды")
            for trend in insights["future_trends"]:
                st.info(f"📈 {trend}")
        
        col1, col2 = st.columns(2)
        
        # Возможности роста
        with col1:
            if "growth_opportunities" in insights and insights["growth_opportunities"]:
                st.subheader("🚀 Возможности роста")
                for opportunity in insights["growth_opportunities"]:
                    st.success(f"💹 {opportunity}")
        
        # Потенциальные риски
        with col2:
            if "potential_risks" in insights and insights["potential_risks"]:
                st.subheader("⚠️ Потенциальные риски")
                for risk in insights["potential_risks"]:
                    st.error(f"🚨 {risk}")
        
        # Стратегические рекомендации
        if "strategic_recommendations" in insights and insights["strategic_recommendations"]:
            st.subheader("📋 Стратегические рекомендации")
            for recommendation in insights["strategic_recommendations"]:
                st.success(f"💼 {recommendation}")
        
        # Временные рамки и уверенность
        col1, col2 = st.columns(2)
        with col1:
            if "timeline" in insights:
                st.metric("📅 Временные рамки", insights["timeline"])
        
        with col2:
            if "confidence_level" in insights:
                st.metric("🎯 Уровень уверенности", insights["confidence_level"])
    
    def _render_statistical_summary(self, stats: Dict[str, Any]) -> None:
        """Отображение статистического резюме."""
        with st.expander("📊 Детальная статистика", expanded=False):
            
            # Базовая статистика
            if "basic_stats" in stats and stats["basic_stats"]:
                st.subheader("📈 Базовая статистика")
                basic_stats_df = pd.DataFrame(stats["basic_stats"])
                st.dataframe(basic_stats_df.round(3), use_container_width=True)
            
            # Корреляционная матрица
            if "correlation_matrix" in stats and stats["correlation_matrix"]:
                st.subheader("🔗 Корреляционная матрица")
                corr_df = pd.DataFrame(stats["correlation_matrix"])
                
                # Создаем тепловую карту
                fig = px.imshow(
                    corr_df,
                    title="Корреляционная матрица",
                    color_continuous_scale="RdBu",
                    aspect="auto",
                    height=400
                )
                fig.update_traces(
                    text=corr_df.round(2).values,
                    texttemplate="%{text}",
                    textfont={"size": 10}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Выбросы
            if "outliers" in stats and stats["outliers"]:
                st.subheader("🚨 Выявленные выбросы")
                for column, outlier_values in stats["outliers"].items():
                    st.warning(f"**{column}**: {len(outlier_values)} выбросов")
                    if len(outlier_values) <= 10:  # Показываем значения только если их немного
                        st.text(f"Значения: {', '.join(map(str, outlier_values))}")
            
            # Кластерный анализ
            if "clustering" in stats and stats["clustering"]:
                self._render_clustering_results(stats["clustering"])
    
    def _render_clustering_results(self, clustering: Dict[str, Any]) -> None:
        """Отображение результатов кластерного анализа."""
        st.subheader("🎯 Кластерный анализ")
        
        n_clusters = clustering.get("n_clusters", 0)
        st.info(f"Выявлено **{n_clusters}** кластеров")
        
        if "cluster_analysis" in clustering:
            for cluster_name, cluster_info in clustering["cluster_analysis"].items():
                cluster_num = cluster_name.split("_")[1]
                size = cluster_info.get("size", 0)
                characteristics = cluster_info.get("characteristics", "Нет описания")
                
                with st.expander(f"Кластер {cluster_num} ({size} объектов)"):
                    st.markdown(f"**Характеристики**: {characteristics}")
                    
                    # Средние значения кластера
                    if "mean_values" in cluster_info:
                        mean_values = cluster_info["mean_values"]
                        if mean_values:
                            st.markdown("**Средние значения:**")
                            for metric, value in mean_values.items():
                                if isinstance(value, (int, float)):
                                    st.metric(metric, f"{value:.2f}")
    
    def _render_data_quality(self, quality: Dict[str, Any]) -> None:
        """Отображение информации о качестве данных."""
        with st.expander("🔍 Качество данных", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                score = quality.get("quality_score", 0)
                st.metric("📊 Оценка качества", f"{score}%")
            
            with col2:
                level = quality.get("quality_level", "Неизвестно")
                st.metric("🏆 Уровень качества", level)
            
            with col3:
                missing = quality.get("missing_percentage", 0)
                st.metric("❌ Пропущенные данные", f"{missing}%")
            
            # Проблемы в данных
            if "issues" in quality and quality["issues"]:
                st.subheader("⚠️ Выявленные проблемы")
                for issue in quality["issues"]:
                    st.warning(f"🔧 {issue}")
    
    def _render_recommendations(self, recommendations: List[str]) -> None:
        """Отображение итоговых рекомендаций."""
        st.subheader("💡 Итоговые рекомендации")
        
        for i, recommendation in enumerate(recommendations, 1):
            st.success(f"**{i}.** {recommendation}")
        
        # Кнопка экспорта рекомендаций
        if st.button("📋 Скопировать рекомендации"):
            recommendations_text = "\n".join([f"{i}. {rec}" for i, rec in enumerate(recommendations, 1)])
            st.code(recommendations_text, language=None)
            st.success("Рекомендации готовы для копирования!")


# Singleton instance
analysis_ui_renderer: AnalysisUIRenderer = None


def get_analysis_ui_renderer() -> AnalysisUIRenderer:
    """
    Получение единственного экземпляра AnalysisUIRenderer.
    
    Returns:
        Экземпляр AnalysisUIRenderer
    """
    global analysis_ui_renderer
    if analysis_ui_renderer is None:
        analysis_ui_renderer = AnalysisUIRenderer()
    return analysis_ui_renderer


def render_analysis_quick_summary(insights: Dict[str, Any]) -> None:
    """
    Быстрое отображение краткого резюме анализа.
    
    Args:
        insights: AI инсайты для краткого отображения
    """
    if "key_insights" in insights and insights["key_insights"]:
        with st.container():
            st.markdown("### 🔍 Главные выводы")
            for insight in insights["key_insights"][:3]:  # Показываем только первые 3
                st.info(f"💡 {insight}")
    
    if "recommendations" in insights and insights["recommendations"]:
        with st.container():
            st.markdown("### 💡 Ключевые рекомендации")
            for rec in insights["recommendations"][:2]:  # Показываем только первые 2
                st.success(f"✅ {rec}") 