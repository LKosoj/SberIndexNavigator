"""
Специализированные инструменты анализа для SmartAnalysisAgent.

Содержит:
- StatisticalTool - расширенная статистика
- CorrelationTool - анализ взаимосвязей  
- TrendTool - выявление трендов
- AnomalyTool - поиск аномалий
- ComparisonTool - сравнительный анализ
"""

import logging
import json
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.config.settings import (
    OPENAI_API_KEY,
    OPENAI_BASE_URL, 
    OPENAI_MODEL,
    OPENAI_TEMPERATURE
)

logger = logging.getLogger(__name__)


class StatisticalTool(BaseTool):
    """Инструмент для расширенного статистического анализа."""
    
    name: str = "statistical_analysis"
    description: str = "Проводит углубленный статистический анализ данных"
    
    def _run(self, data: pd.DataFrame, question: str = "") -> Dict[str, Any]:
        """Выполнение статистического анализа."""
        try:
            result = {
                "analysis_type": "statistical",
                "basic_stats": self._calculate_basic_stats(data),
                "distribution_analysis": self._analyze_distributions(data),
                "outlier_detection": self._detect_outliers(data),
                "normality_tests": self._test_normality(data),
                "recommendations": []
            }
            
            # Добавляем рекомендации на основе анализа
            result["recommendations"] = self._generate_statistical_recommendations(result)
            
            logger.info(f"Статистический анализ завершен для {data.shape[0]} строк, {data.shape[1]} колонок")
            return result
            
        except Exception as e:
            logger.error(f"Ошибка статистического анализа: {e}")
            return {"error": str(e), "analysis_type": "statistical"}
    
    def _calculate_basic_stats(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Расчет базовой статистики."""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return {"message": "Нет числовых колонок для анализа"}
        
        stats_dict = {}
        for col in numeric_cols:
            stats_dict[col] = {
                "mean": float(data[col].mean()),
                "median": float(data[col].median()),
                "std": float(data[col].std()),
                "min": float(data[col].min()),
                "max": float(data[col].max()),
                "q25": float(data[col].quantile(0.25)),
                "q75": float(data[col].quantile(0.75)),
                "skewness": float(data[col].skew()),
                "kurtosis": float(data[col].kurtosis()),
                "coefficient_of_variation": float(data[col].std() / data[col].mean()) if data[col].mean() != 0 else 0
            }
        
        return stats_dict
    
    def _analyze_distributions(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Анализ распределений."""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        distributions = {}
        
        for col in numeric_cols:
            col_data = data[col].dropna()
            if len(col_data) < 3:
                continue
                
            # Определяем тип распределения
            dist_type = "unknown"
            if abs(col_data.skew()) < 0.5:
                dist_type = "normal-like"
            elif col_data.skew() > 1:
                dist_type = "right-skewed"
            elif col_data.skew() < -1:
                dist_type = "left-skewed"
            else:
                dist_type = "moderately_skewed"
            
            distributions[col] = {
                "type": dist_type,
                "skewness": float(col_data.skew()),
                "kurtosis": float(col_data.kurtosis()),
                "unique_values": int(col_data.nunique()),
                "missing_percent": float((len(data) - len(col_data)) / len(data) * 100)
            }
        
        return distributions
    
    def _detect_outliers(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Обнаружение выбросов."""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        outliers = {}
        
        for col in numeric_cols:
            col_data = data[col].dropna()
            if len(col_data) < 4:
                continue
            
            # IQR метод
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outlier_indices = data[(data[col] < lower_bound) | (data[col] > upper_bound)].index.tolist()
            
            # Z-score метод
            z_scores = np.abs(stats.zscore(col_data))
            z_outliers = data[z_scores > 3].index.tolist() if len(z_scores) > 0 else []
            
            outliers[col] = {
                "iqr_outliers": len(outlier_indices),
                "iqr_outlier_indices": outlier_indices[:10],  # Показываем только первые 10
                "zscore_outliers": len(z_outliers),
                "outlier_percentage": float(len(outlier_indices) / len(data) * 100),
                "bounds": {
                    "lower": float(lower_bound),
                    "upper": float(upper_bound)
                }
            }
        
        return outliers
    
    def _test_normality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Тесты на нормальность."""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        normality_tests = {}
        
        for col in numeric_cols:
            col_data = data[col].dropna()
            if len(col_data) < 8:  # Минимум для Shapiro-Wilk
                continue
            
            try:
                # Shapiro-Wilk тест (для небольших выборок)
                if len(col_data) <= 5000:
                    shapiro_stat, shapiro_p = stats.shapiro(col_data)
                    is_normal_shapiro = shapiro_p > 0.05
                else:
                    shapiro_stat, shapiro_p, is_normal_shapiro = None, None, None
                
                # Kolmogorov-Smirnov тест
                ks_stat, ks_p = stats.kstest(col_data, 'norm', args=(col_data.mean(), col_data.std()))
                is_normal_ks = ks_p > 0.05
                
                normality_tests[col] = {
                    "shapiro_wilk": {
                        "statistic": float(shapiro_stat) if shapiro_stat else None,
                        "p_value": float(shapiro_p) if shapiro_p else None,
                        "is_normal": is_normal_shapiro
                    },
                    "kolmogorov_smirnov": {
                        "statistic": float(ks_stat),
                        "p_value": float(ks_p),
                        "is_normal": is_normal_ks
                    },
                    "visual_indicators": {
                        "skewness_normal": abs(col_data.skew()) < 0.5,
                        "kurtosis_normal": abs(col_data.kurtosis()) < 3
                    }
                }
                
            except Exception as e:
                logger.warning(f"Ошибка теста нормальности для {col}: {e}")
                normality_tests[col] = {"error": str(e)}
        
        return normality_tests
    
    def _generate_statistical_recommendations(self, analysis_result: Dict[str, Any]) -> List[str]:
        """Генерация рекомендаций на основе статистического анализа."""
        recommendations = []
        
        # Анализ выбросов
        outliers = analysis_result.get("outlier_detection", {})
        for col, outlier_info in outliers.items():
            if outlier_info.get("outlier_percentage", 0) > 5:
                recommendations.append(f"В колонке '{col}' обнаружено {outlier_info['outlier_percentage']:.1f}% выбросов. Рекомендуется проверить данные.")
        
        # Анализ распределений
        distributions = analysis_result.get("distribution_analysis", {})
        for col, dist_info in distributions.items():
            if abs(dist_info.get("skewness", 0)) > 2:
                recommendations.append(f"Колонка '{col}' имеет сильную асимметрию (skewness={dist_info['skewness']:.2f}). Рассмотрите логарифмическое преобразование.")
        
        # Анализ пропущенных значений
        for col, dist_info in distributions.items():
            if dist_info.get("missing_percent", 0) > 20:
                recommendations.append(f"В колонке '{col}' {dist_info['missing_percent']:.1f}% пропущенных значений. Требуется стратегия обработки.")
        
        return recommendations[:5]  # Ограничиваем количество рекомендаций


class CorrelationTool(BaseTool):
    """Инструмент для анализа корреляций и взаимосвязей."""
    
    name: str = "correlation_analysis"
    description: str = "Анализирует корреляции и взаимосвязи между переменными"
    
    def _run(self, data: pd.DataFrame, question: str = "") -> Dict[str, Any]:
        """Выполнение корреляционного анализа."""
        try:
            result = {
                "analysis_type": "correlation",
                "correlation_matrix": self._calculate_correlations(data),
                "significant_correlations": self._find_significant_correlations(data),
                "correlation_patterns": self._identify_correlation_patterns(data),
                "recommendations": []
            }
            
            result["recommendations"] = self._generate_correlation_recommendations(result)
            
            logger.info(f"Корреляционный анализ завершен для {data.shape[1]} переменных")
            return result
            
        except Exception as e:
            logger.error(f"Ошибка корреляционного анализа: {e}")
            return {"error": str(e), "analysis_type": "correlation"}
    
    def _calculate_correlations(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Расчет различных типов корреляций."""
        numeric_data = data.select_dtypes(include=[np.number])
        
        if numeric_data.shape[1] < 2:
            return {"message": "Недостаточно числовых колонок для корреляционного анализа"}
        
        # Pearson корреляция
        pearson_corr = numeric_data.corr(method='pearson')
        
        # Spearman корреляция (ранговая)
        spearman_corr = numeric_data.corr(method='spearman')
        
        # Kendall корреляция
        kendall_corr = numeric_data.corr(method='kendall')
        
        return {
            "pearson": pearson_corr.to_dict(),
            "spearman": spearman_corr.to_dict(),
            "kendall": kendall_corr.to_dict(),
            "variables": list(numeric_data.columns)
        }
    
    def _find_significant_correlations(self, data: pd.DataFrame, threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Поиск значимых корреляций."""
        numeric_data = data.select_dtypes(include=[np.number])
        correlation_matrix = numeric_data.corr(method='pearson')
        
        significant_correlations = []
        
        for i, var1 in enumerate(correlation_matrix.columns):
            for j, var2 in enumerate(correlation_matrix.columns):
                if i < j:  # Избегаем дублирования
                    corr_value = correlation_matrix.loc[var1, var2]
                    if abs(corr_value) >= threshold and not pd.isna(corr_value):
                        # Вычисляем p-value
                        try:
                            corr_coeff, p_value = stats.pearsonr(data[var1].dropna(), data[var2].dropna())
                            significance = "высокая" if abs(corr_value) >= 0.7 else "средняя" if abs(corr_value) >= 0.5 else "слабая"
                            direction = "положительная" if corr_value > 0 else "отрицательная"
                            
                            significant_correlations.append({
                                "variable1": var1,
                                "variable2": var2,
                                "correlation": float(corr_value),
                                "p_value": float(p_value),
                                "significance": significance,
                                "direction": direction,
                                "interpretation": f"{direction.capitalize()} {significance} связь между {var1} и {var2}"
                            })
                        except Exception as e:
                            logger.warning(f"Ошибка расчета p-value для {var1}-{var2}: {e}")
        
        # Сортируем по убыванию силы связи
        significant_correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)
        return significant_correlations[:10]  # Топ-10 корреляций
    
    def _identify_correlation_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Выявление паттернов в корреляциях."""
        numeric_data = data.select_dtypes(include=[np.number])
        correlation_matrix = numeric_data.corr(method='pearson')
        
        patterns = {
            "highly_correlated_groups": [],
            "correlation_clusters": [],
            "isolation_variables": [],
            "multicollinearity_warning": []
        }
        
        # Поиск групп высоко коррелированных переменных
        high_corr_threshold = 0.8
        processed_vars = set()
        
        for var in correlation_matrix.columns:
            if var in processed_vars:
                continue
                
            high_corr_vars = [var]
            for other_var in correlation_matrix.columns:
                if (var != other_var and 
                    abs(correlation_matrix.loc[var, other_var]) >= high_corr_threshold and
                    not pd.isna(correlation_matrix.loc[var, other_var])):
                    high_corr_vars.append(other_var)
            
            if len(high_corr_vars) > 2:
                patterns["highly_correlated_groups"].append({
                    "variables": high_corr_vars,
                    "average_correlation": float(np.mean([
                        abs(correlation_matrix.loc[v1, v2]) 
                        for i, v1 in enumerate(high_corr_vars) 
                        for j, v2 in enumerate(high_corr_vars) 
                        if i < j
                    ]))
                })
                processed_vars.update(high_corr_vars)
        
        # Поиск изолированных переменных (слабо связанных с другими)
        for var in correlation_matrix.columns:
            max_corr = max([abs(correlation_matrix.loc[var, other_var]) 
                           for other_var in correlation_matrix.columns 
                           if var != other_var and not pd.isna(correlation_matrix.loc[var, other_var])])
            
            if max_corr < 0.3:
                patterns["isolation_variables"].append({
                    "variable": var,
                    "max_correlation": float(max_corr)
                })
        
        # Предупреждения о мультиколлинеарности
        for var in correlation_matrix.columns:
            high_correlations = [other_var for other_var in correlation_matrix.columns
                               if (var != other_var and 
                                   abs(correlation_matrix.loc[var, other_var]) >= 0.9 and
                                   not pd.isna(correlation_matrix.loc[var, other_var]))]
            
            if high_correlations:
                patterns["multicollinearity_warning"].append({
                    "variable": var,
                    "highly_correlated_with": high_correlations
                })
        
        return patterns
    
    def _generate_correlation_recommendations(self, analysis_result: Dict[str, Any]) -> List[str]:
        """Генерация рекомендаций на основе корреляционного анализа."""
        recommendations = []
        
        # Рекомендации по значимым корреляциям
        significant_corrs = analysis_result.get("significant_correlations", [])
        if significant_corrs:
            top_corr = significant_corrs[0]
            recommendations.append(
                f"Обнаружена {top_corr['significance']} {top_corr['direction']} связь между "
                f"'{top_corr['variable1']}' и '{top_corr['variable2']}' (r={top_corr['correlation']:.3f})"
            )
        
        # Рекомендации по мультиколлинеарности
        patterns = analysis_result.get("correlation_patterns", {})
        multicollinearity = patterns.get("multicollinearity_warning", [])
        if multicollinearity:
            recommendations.append(
                f"Обнаружена мультиколлинеарность в переменных: {', '.join([w['variable'] for w in multicollinearity[:3]])}. "
                "Рассмотрите удаление избыточных переменных."
            )
        
        # Рекомендации по изолированным переменным
        isolated = patterns.get("isolation_variables", [])
        if isolated:
            recommendations.append(
                f"Переменные {', '.join([v['variable'] for v in isolated[:2]])} слабо связаны с другими. "
                "Проверьте их релевантность для анализа."
            )
        
        return recommendations[:5] 


class TrendTool(BaseTool):
    """Инструмент для выявления трендов в данных."""
    
    name: str = "trend_analysis"
    description: str = "Выявляет тренды и временные паттерны в данных"
    
    def _run(self, data: pd.DataFrame, question: str = "") -> Dict[str, Any]:
        """Выполнение анализа трендов."""
        try:
            result = {
                "analysis_type": "trend",
                "temporal_trends": self._analyze_temporal_trends(data),
                "monotonic_trends": self._find_monotonic_trends(data),
                "trend_strength": self._calculate_trend_strength(data),
                "seasonal_patterns": self._detect_seasonal_patterns(data),
                "recommendations": []
            }
            
            result["recommendations"] = self._generate_trend_recommendations(result)
            
            logger.info(f"Анализ трендов завершен для {data.shape[1]} переменных")
            return result
            
        except Exception as e:
            logger.error(f"Ошибка анализа трендов: {e}")
            return {"error": str(e), "analysis_type": "trend"}
    
    def _analyze_temporal_trends(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Анализ временных трендов."""
        temporal_trends = {}
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        # Попытаемся найти временные колонки
        datetime_cols = data.select_dtypes(include=['datetime64', 'object']).columns
        time_col = None
        
        for col in datetime_cols:
            try:
                pd.to_datetime(data[col])
                time_col = col
                break
            except:
                continue
        
        if time_col:
            # Если есть временная колонка
            for col in numeric_cols:
                try:
                    time_data = pd.to_datetime(data[time_col])
                    y_data = data[col].dropna()
                    
                    if len(y_data) >= 3:
                        # Линейная регрессия для тренда
                        x_numeric = np.arange(len(y_data))
                        slope, intercept, r_value, p_value, std_err = stats.linregress(x_numeric, y_data)
                        
                        trend_direction = "возрастающий" if slope > 0 else "убывающий" if slope < 0 else "стабильный"
                        trend_strength = "сильный" if abs(r_value) > 0.7 else "умеренный" if abs(r_value) > 0.3 else "слабый"
                        
                        temporal_trends[col] = {
                            "slope": float(slope),
                            "r_squared": float(r_value ** 2),
                            "p_value": float(p_value),
                            "direction": trend_direction,
                            "strength": trend_strength,
                            "significant": p_value < 0.05
                        }
                except Exception as e:
                    logger.warning(f"Ошибка анализа тренда для {col}: {e}")
        else:
            # Если нет явной временной колонки, анализируем как последовательность
            for col in numeric_cols:
                y_data = data[col].dropna()
                if len(y_data) >= 3:
                    x_numeric = np.arange(len(y_data))
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x_numeric, y_data)
                    
                    temporal_trends[col] = {
                        "slope": float(slope),
                        "r_squared": float(r_value ** 2),
                        "p_value": float(p_value),
                        "note": "Анализ по порядку строк (нет временной колонки)"
                    }
        
        return temporal_trends
    
    def _find_monotonic_trends(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Поиск монотонных трендов."""
        monotonic_trends = {}
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            col_data = data[col].dropna()
            if len(col_data) >= 3:
                # Проверяем монотонность
                is_increasing = col_data.is_monotonic_increasing
                is_decreasing = col_data.is_monotonic_decreasing
                
                # Рассчитываем степень монотонности
                if len(col_data) > 1:
                    diffs = np.diff(col_data)
                    increasing_steps = np.sum(diffs > 0)
                    decreasing_steps = np.sum(diffs < 0)
                    stable_steps = np.sum(diffs == 0)
                    
                    monotonic_trends[col] = {
                        "is_strictly_increasing": is_increasing and not any(diffs == 0),
                        "is_strictly_decreasing": is_decreasing and not any(diffs == 0),
                        "is_non_decreasing": is_increasing,
                        "is_non_increasing": is_decreasing,
                        "increasing_percentage": float(increasing_steps / len(diffs) * 100),
                        "decreasing_percentage": float(decreasing_steps / len(diffs) * 100),
                        "stable_percentage": float(stable_steps / len(diffs) * 100)
                    }
        
        return monotonic_trends
    
    def _calculate_trend_strength(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Расчет силы тренда."""
        trend_strength = {}
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            col_data = data[col].dropna()
            if len(col_data) >= 4:
                # Расчет различных метрик силы тренда
                x = np.arange(len(col_data))
                
                # Коэффициент детерминации
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, col_data)
                r_squared = r_value ** 2
                
                # Средняя абсолютная ошибка тренда
                trend_line = slope * x + intercept
                mae = np.mean(np.abs(col_data - trend_line))
                
                # Относительная ошибка
                relative_error = mae / np.mean(np.abs(col_data)) if np.mean(np.abs(col_data)) != 0 else 0
                
                trend_strength[col] = {
                    "r_squared": float(r_squared),
                    "slope_significance": float(p_value),
                    "mean_absolute_error": float(mae),
                    "relative_error": float(relative_error),
                    "trend_quality": "отличный" if r_squared > 0.8 else "хороший" if r_squared > 0.6 else "удовлетворительный" if r_squared > 0.3 else "слабый"
                }
        
        return trend_strength
    
    def _detect_seasonal_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Обнаружение сезонных паттернов (упрощенная версия)."""
        seasonal_patterns = {}
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            col_data = data[col].dropna()
            if len(col_data) >= 12:  # Минимум год данных
                try:
                    # Простой анализ цикличности через автокорреляцию
                    max_lag = min(len(col_data) // 4, 24)  # Максимум 24 лага или четверть данных
                    
                    autocorrelations = []
                    for lag in range(1, max_lag + 1):
                        if len(col_data) > lag:
                            corr = np.corrcoef(col_data[:-lag], col_data[lag:])[0, 1]
                            if not np.isnan(corr):
                                autocorrelations.append((lag, abs(corr)))
                    
                    if autocorrelations:
                        # Находим наиболее значимые периоды
                        autocorrelations.sort(key=lambda x: x[1], reverse=True)
                        top_periods = autocorrelations[:3]
                        
                        seasonal_patterns[col] = {
                            "potential_periods": [{"lag": lag, "correlation": float(corr)} for lag, corr in top_periods],
                            "strongest_period": top_periods[0][0] if top_periods else None,
                            "seasonality_strength": float(top_periods[0][1]) if top_periods else 0
                        }
                        
                except Exception as e:
                    logger.warning(f"Ошибка анализа сезонности для {col}: {e}")
        
        return seasonal_patterns
    
    def _generate_trend_recommendations(self, analysis_result: Dict[str, Any]) -> List[str]:
        """Генерация рекомендаций на основе анализа трендов."""
        recommendations = []
        
        # Анализ временных трендов
        temporal_trends = analysis_result.get("temporal_trends", {})
        strong_trends = [(col, info) for col, info in temporal_trends.items() 
                        if info.get("r_squared", 0) > 0.5 and info.get("significant", False)]
        
        if strong_trends:
            col, info = strong_trends[0]
            recommendations.append(
                f"Обнаружен {info['strength']} {info['direction']} тренд в '{col}' "
                f"(R²={info['r_squared']:.3f}). Рассмотрите прогнозирование."
            )
        
        # Анализ монотонности
        monotonic_trends = analysis_result.get("monotonic_trends", {})
        highly_monotonic = [(col, info) for col, info in monotonic_trends.items()
                           if info.get("increasing_percentage", 0) > 80 or info.get("decreasing_percentage", 0) > 80]
        
        if highly_monotonic:
            col, info = highly_monotonic[0]
            direction = "возрастания" if info.get("increasing_percentage", 0) > 80 else "убывания"
            recommendations.append(f"Переменная '{col}' показывает устойчивый тренд {direction}.")
        
        # Анализ сезонности
        seasonal_patterns = analysis_result.get("seasonal_patterns", {})
        strong_seasonal = [(col, info) for col, info in seasonal_patterns.items()
                          if info.get("seasonality_strength", 0) > 0.6]
        
        if strong_seasonal:
            col, info = strong_seasonal[0]
            period = info.get("strongest_period")
            recommendations.append(f"В '{col}' обнаружен сезонный паттерн с периодом {period}.")
        
        return recommendations[:5]


class AnomalyTool(BaseTool):
    """Инструмент для поиска аномалий в данных."""
    
    name: str = "anomaly_detection"
    description: str = "Обнаруживает аномалии и выбросы в данных"
    
    def _run(self, data: pd.DataFrame, question: str = "") -> Dict[str, Any]:
        """Выполнение поиска аномалий."""
        try:
            result = {
                "analysis_type": "anomaly",
                "statistical_outliers": self._detect_statistical_outliers(data),
                "isolation_anomalies": self._detect_isolation_anomalies(data),
                "pattern_anomalies": self._detect_pattern_anomalies(data),
                "anomaly_summary": {},
                "recommendations": []
            }
            
            result["anomaly_summary"] = self._summarize_anomalies(result)
            result["recommendations"] = self._generate_anomaly_recommendations(result)
            
            logger.info(f"Поиск аномалий завершен для {data.shape[0]} записей")
            return result
            
        except Exception as e:
            logger.error(f"Ошибка поиска аномалий: {e}")
            return {"error": str(e), "analysis_type": "anomaly"}
    
    def _detect_statistical_outliers(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Обнаружение статистических выбросов."""
        outliers = {}
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            col_data = data[col].dropna()
            if len(col_data) < 4:
                continue
            
            # IQR метод
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            iqr_outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
            
            # Z-score метод
            z_scores = np.abs(stats.zscore(col_data))
            z_outlier_mask = z_scores > 3
            z_outliers = data.iloc[col_data.index[z_outlier_mask]]
            
            # Modified Z-score (более устойчивый к выбросам)
            median = col_data.median()
            mad = np.median(np.abs(col_data - median))
            modified_z_scores = 0.6745 * (col_data - median) / mad if mad != 0 else np.zeros_like(col_data)
            modified_z_outliers = data.iloc[col_data.index[np.abs(modified_z_scores) > 3.5]]
            
            outliers[col] = {
                "iqr_outliers": {
                    "count": len(iqr_outliers),
                    "percentage": float(len(iqr_outliers) / len(data) * 100),
                    "indices": iqr_outliers.index.tolist()[:10],  # Первые 10
                    "values": iqr_outliers[col].tolist()[:10],
                    "bounds": {"lower": float(lower_bound), "upper": float(upper_bound)}
                },
                "zscore_outliers": {
                    "count": len(z_outliers),
                    "percentage": float(len(z_outliers) / len(data) * 100),
                    "indices": z_outliers.index.tolist()[:10]
                },
                "modified_zscore_outliers": {
                    "count": len(modified_z_outliers),
                    "percentage": float(len(modified_z_outliers) / len(data) * 100),
                    "indices": modified_z_outliers.index.tolist()[:10]
                }
            }
        
        return outliers
    
    def _detect_isolation_anomalies(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Обнаружение аномалий методом изоляции (упрощенная версия)."""
        try:
            from sklearn.ensemble import IsolationForest
            
            numeric_data = data.select_dtypes(include=[np.number]).dropna()
            if numeric_data.empty or numeric_data.shape[1] < 2:
                return {"message": "Недостаточно данных для метода изоляции"}
            
            # Стандартизация данных
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(numeric_data)
            
            # Isolation Forest
            isolation_forest = IsolationForest(contamination=0.1, random_state=42)
            anomaly_labels = isolation_forest.fit_predict(scaled_data)
            
            # Индексы аномалий
            anomaly_indices = numeric_data.index[anomaly_labels == -1].tolist()
            anomaly_scores = isolation_forest.decision_function(scaled_data)
            
            return {
                "anomaly_count": len(anomaly_indices),
                "anomaly_percentage": float(len(anomaly_indices) / len(numeric_data) * 100),
                "anomaly_indices": anomaly_indices[:20],  # Первые 20
                "average_anomaly_score": float(np.mean(anomaly_scores[anomaly_labels == -1])),
                "features_used": list(numeric_data.columns)
            }
            
        except ImportError:
            return {"message": "sklearn не установлен для метода изоляции"}
        except Exception as e:
            logger.warning(f"Ошибка метода изоляции: {e}")
            return {"error": str(e)}
    
    def _detect_pattern_anomalies(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Обнаружение аномалий паттернов."""
        pattern_anomalies = {}
        
        # Аномалии в категориальных данных
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            value_counts = data[col].value_counts()
            total_count = len(data[col].dropna())
            
            # Находим очень редкие значения (< 1% от общего количества)
            rare_values = value_counts[value_counts / total_count < 0.01]
            
            if len(rare_values) > 0:
                pattern_anomalies[col] = {
                    "rare_values": rare_values.to_dict(),
                    "rare_count": len(rare_values),
                    "rare_percentage": float(len(rare_values) / len(value_counts) * 100)
                }
        
        # Аномалии в распределениях
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            col_data = data[col].dropna()
            if len(col_data) >= 10:
                # Проверяем на нормальность распределения
                _, p_value = stats.shapiro(col_data[:5000])  # Ограничиваем выборку для Shapiro
                
                # Проверяем на многомодальность (упрощенно)
                hist, _ = np.histogram(col_data, bins=min(50, len(col_data)//10))
                peaks = np.where((hist[1:-1] > hist[:-2]) & (hist[1:-1] > hist[2:]))[0]
                
                pattern_anomalies[f"{col}_distribution"] = {
                    "normality_p_value": float(p_value),
                    "is_likely_normal": p_value > 0.05,
                    "potential_peaks": len(peaks),
                    "multimodal_suspicion": len(peaks) > 1
                }
        
        return pattern_anomalies
    
    def _summarize_anomalies(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Суммирование найденных аномалий."""
        summary = {
            "total_statistical_outliers": 0,
            "columns_with_outliers": [],
            "highest_outlier_percentage": 0,
            "most_problematic_column": None,
            "isolation_anomalies": 0,
            "pattern_anomalies": 0
        }
        
        # Суммируем статистические выбросы
        statistical_outliers = analysis_result.get("statistical_outliers", {})
        for col, outlier_info in statistical_outliers.items():
            iqr_count = outlier_info.get("iqr_outliers", {}).get("count", 0)
            iqr_percentage = outlier_info.get("iqr_outliers", {}).get("percentage", 0)
            
            if iqr_count > 0:
                summary["total_statistical_outliers"] += iqr_count
                summary["columns_with_outliers"].append(col)
                
                if iqr_percentage > summary["highest_outlier_percentage"]:
                    summary["highest_outlier_percentage"] = iqr_percentage
                    summary["most_problematic_column"] = col
        
        # Изоляционные аномалии
        isolation_anomalies = analysis_result.get("isolation_anomalies", {})
        summary["isolation_anomalies"] = isolation_anomalies.get("anomaly_count", 0)
        
        # Паттерн аномалии
        pattern_anomalies = analysis_result.get("pattern_anomalies", {})
        summary["pattern_anomalies"] = len(pattern_anomalies)
        
        return summary
    
    def _generate_anomaly_recommendations(self, analysis_result: Dict[str, Any]) -> List[str]:
        """Генерация рекомендаций по аномалиям."""
        recommendations = []
        
        summary = analysis_result.get("anomaly_summary", {})
        
        # Рекомендации по статистическим выбросам
        if summary.get("highest_outlier_percentage", 0) > 5:
            problematic_col = summary.get("most_problematic_column")
            percentage = summary.get("highest_outlier_percentage")
            recommendations.append(
                f"В колонке '{problematic_col}' обнаружено {percentage:.1f}% выбросов. "
                "Проверьте данные на ошибки ввода или рассмотрите специальную обработку."
            )
        
        # Рекомендации по изоляционным аномалиям
        isolation_count = summary.get("isolation_anomalies", 0)
        if isolation_count > 0:
            recommendations.append(
                f"Метод изоляции обнаружил {isolation_count} многомерных аномалий. "
                "Исследуйте эти записи для понимания причин отклонений."
            )
        
        # Рекомендации по паттернам
        pattern_count = summary.get("pattern_anomalies", 0)
        if pattern_count > 0:
            recommendations.append(
                f"Обнаружены аномалии в {pattern_count} паттернах данных. "
                "Проверьте качество и консистентность данных."
            )
        
        # Общие рекомендации
        total_outliers = summary.get("total_statistical_outliers", 0)
        if total_outliers > 0:
            recommendations.append(
                f"Общее количество статистических выбросов: {total_outliers}. "
                "Рассмотрите стратегию их обработки (удаление, трансформация, отдельный анализ)."
            )
        
        return recommendations[:5]


class ComparisonTool(BaseTool):
    """Инструмент для сравнительного анализа данных."""
    
    name: str = "comparison_analysis"
    description: str = "Проводит сравнительный анализ между группами или периодами"
    
    def _run(self, data: pd.DataFrame, question: str = "") -> Dict[str, Any]:
        """Выполнение сравнительного анализа."""
        try:
            result = {
                "analysis_type": "comparison",
                "group_comparisons": self._compare_groups(data),
                "statistical_tests": self._perform_statistical_tests(data),
                "effect_sizes": self._calculate_effect_sizes(data),
                "ranking_analysis": self._perform_ranking_analysis(data),
                "recommendations": []
            }
            
            result["recommendations"] = self._generate_comparison_recommendations(result)
            
            logger.info(f"Сравнительный анализ завершен")
            return result
            
        except Exception as e:
            logger.error(f"Ошибка сравнительного анализа: {e}")
            return {"error": str(e), "analysis_type": "comparison"}
    
    def _compare_groups(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Сравнение групп в данных."""
        group_comparisons = {}
        
        # Ищем потенциальные группирующие переменные
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for cat_col in categorical_cols:
            unique_values = data[cat_col].dropna().unique()
            if 2 <= len(unique_values) <= 10:  # Разумное количество групп
                
                group_stats = {}
                for group in unique_values:
                    group_data = data[data[cat_col] == group]
                    
                    group_stats[str(group)] = {
                        "count": len(group_data),
                        "percentage": float(len(group_data) / len(data) * 100)
                    }
                    
                    # Статистики по числовым колонкам
                    for num_col in numeric_cols:
                        group_values = group_data[num_col].dropna()
                        if len(group_values) > 0:
                            if num_col not in group_stats[str(group)]:
                                group_stats[str(group)][num_col] = {}
                            
                            group_stats[str(group)][num_col] = {
                                "mean": float(group_values.mean()),
                                "median": float(group_values.median()),
                                "std": float(group_values.std()),
                                "min": float(group_values.min()),
                                "max": float(group_values.max()),
                                "count": len(group_values)
                            }
                
                group_comparisons[cat_col] = group_stats
        
        return group_comparisons
    
    def _perform_statistical_tests(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Выполнение статистических тестов."""
        test_results = {}
        
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for cat_col in categorical_cols:
            unique_values = data[cat_col].dropna().unique()
            if 2 <= len(unique_values) <= 5:
                
                for num_col in numeric_cols:
                    groups = []
                    group_names = []
                    
                    for group in unique_values:
                        group_data = data[data[cat_col] == group][num_col].dropna()
                        if len(group_data) >= 3:  # Минимум для тестов
                            groups.append(group_data)
                            group_names.append(str(group))
                    
                    if len(groups) >= 2:
                        test_key = f"{cat_col}_vs_{num_col}"
                        test_results[test_key] = {
                            "grouping_variable": cat_col,
                            "numeric_variable": num_col,
                            "groups": group_names,
                            "group_sizes": [len(g) for g in groups]
                        }
                        
                        # Тест на равенство дисперсий (Levene)
                        try:
                            levene_stat, levene_p = stats.levene(*groups)
                            test_results[test_key]["levene_test"] = {
                                "statistic": float(levene_stat),
                                "p_value": float(levene_p),
                                "equal_variances": levene_p > 0.05
                            }
                        except:
                            test_results[test_key]["levene_test"] = {"error": "Не удалось выполнить"}
                        
                        # T-test для двух групп
                        if len(groups) == 2:
                            try:
                                t_stat, t_p = stats.ttest_ind(groups[0], groups[1])
                                test_results[test_key]["t_test"] = {
                                    "statistic": float(t_stat),
                                    "p_value": float(t_p),
                                    "significant": t_p < 0.05
                                }
                            except:
                                test_results[test_key]["t_test"] = {"error": "Не удалось выполнить"}
                        
                        # ANOVA для множественных групп
                        if len(groups) > 2:
                            try:
                                f_stat, f_p = stats.f_oneway(*groups)
                                test_results[test_key]["anova"] = {
                                    "f_statistic": float(f_stat),
                                    "p_value": float(f_p),
                                    "significant": f_p < 0.05
                                }
                            except:
                                test_results[test_key]["anova"] = {"error": "Не удалось выполнить"}
                        
                        # Kruskal-Wallis (непараметрический аналог ANOVA)
                        try:
                            kw_stat, kw_p = stats.kruskal(*groups)
                            test_results[test_key]["kruskal_wallis"] = {
                                "statistic": float(kw_stat),
                                "p_value": float(kw_p),
                                "significant": kw_p < 0.05
                            }
                        except:
                            test_results[test_key]["kruskal_wallis"] = {"error": "Не удалось выполнить"}
        
        return test_results
    
    def _calculate_effect_sizes(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Расчет размеров эффекта."""
        effect_sizes = {}
        
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for cat_col in categorical_cols:
            unique_values = data[cat_col].dropna().unique()
            if len(unique_values) == 2:  # Только для двух групп
                
                for num_col in numeric_cols:
                    groups = []
                    group_names = []
                    
                    for group in unique_values:
                        group_data = data[data[cat_col] == group][num_col].dropna()
                        if len(group_data) >= 3:
                            groups.append(group_data)
                            group_names.append(str(group))
                    
                    if len(groups) == 2:
                        try:
                            # Cohen's d
                            mean1, mean2 = groups[0].mean(), groups[1].mean()
                            std1, std2 = groups[0].std(), groups[1].std()
                            n1, n2 = len(groups[0]), len(groups[1])
                            
                            # Pooled standard deviation
                            pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
                            cohens_d = (mean1 - mean2) / pooled_std if pooled_std != 0 else 0
                            
                            # Интерпретация размера эффекта
                            if abs(cohens_d) < 0.2:
                                effect_interpretation = "малый"
                            elif abs(cohens_d) < 0.5:
                                effect_interpretation = "средний"
                            elif abs(cohens_d) < 0.8:
                                effect_interpretation = "большой"
                            else:
                                effect_interpretation = "очень большой"
                            
                            effect_sizes[f"{cat_col}_vs_{num_col}"] = {
                                "cohens_d": float(cohens_d),
                                "effect_size": effect_interpretation,
                                "groups": group_names,
                                "means": [float(mean1), float(mean2)],
                                "difference": float(mean1 - mean2)
                            }
                            
                        except Exception as e:
                            logger.warning(f"Ошибка расчета размера эффекта для {cat_col} vs {num_col}: {e}")
        
        return effect_sizes
    
    def _perform_ranking_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Ранжирование и анализ позиций."""
        ranking_analysis = {}
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        
        # Ранжирование по числовым переменным
        for num_col in numeric_cols:
            col_data = data[num_col].dropna()
            if len(col_data) > 0:
                # Создаем ранги
                ranks = col_data.rank(method='average', ascending=False)
                
                ranking_analysis[num_col] = {
                    "top_performers": col_data.nlargest(5).to_dict(),
                    "bottom_performers": col_data.nsmallest(5).to_dict(),
                    "median_rank": float(ranks.median()),
                    "rank_distribution": {
                        "top_quartile": float(ranks.quantile(0.25)),
                        "median": float(ranks.quantile(0.5)),
                        "bottom_quartile": float(ranks.quantile(0.75))
                    }
                }
        
        # Ранжирование групп по средним значениям
        for cat_col in categorical_cols:
            unique_values = data[cat_col].dropna().unique()
            if 2 <= len(unique_values) <= 10:
                
                group_rankings = {}
                for num_col in numeric_cols:
                    group_means = []
                    for group in unique_values:
                        group_data = data[data[cat_col] == group][num_col].dropna()
                        if len(group_data) > 0:
                            group_means.append((str(group), float(group_data.mean())))
                    
                    if group_means:
                        # Сортируем по убыванию средних значений
                        group_means.sort(key=lambda x: x[1], reverse=True)
                        group_rankings[num_col] = {
                            "ranking": [{"group": group, "mean": mean, "rank": idx + 1} 
                                       for idx, (group, mean) in enumerate(group_means)],
                            "leader": group_means[0][0] if group_means else None,
                            "outsider": group_means[-1][0] if group_means else None
                        }
                
                if group_rankings:
                    ranking_analysis[f"groups_by_{cat_col}"] = group_rankings
        
        return ranking_analysis
    
    def _generate_comparison_recommendations(self, analysis_result: Dict[str, Any]) -> List[str]:
        """Генерация рекомендаций по сравнительному анализу."""
        recommendations = []
        
        # Анализ статистических тестов
        test_results = analysis_result.get("statistical_tests", {})
        significant_tests = []
        
        for test_name, test_info in test_results.items():
            if test_info.get("t_test", {}).get("significant", False):
                significant_tests.append((test_name, "t-test", test_info["t_test"]["p_value"]))
            elif test_info.get("anova", {}).get("significant", False):
                significant_tests.append((test_name, "ANOVA", test_info["anova"]["p_value"]))
            elif test_info.get("kruskal_wallis", {}).get("significant", False):
                significant_tests.append((test_name, "Kruskal-Wallis", test_info["kruskal_wallis"]["p_value"]))
        
        if significant_tests:
            test_name, test_type, p_value = significant_tests[0]
            grouping_var = test_results[test_name]["grouping_variable"]
            numeric_var = test_results[test_name]["numeric_variable"]
            recommendations.append(
                f"Обнаружены значимые различия в '{numeric_var}' между группами '{grouping_var}' "
                f"({test_type}, p={p_value:.4f}). Исследуйте причины различий."
            )
        
        # Анализ размеров эффекта
        effect_sizes = analysis_result.get("effect_sizes", {})
        large_effects = [(name, info) for name, info in effect_sizes.items() 
                        if info.get("effect_size") in ["большой", "очень большой"]]
        
        if large_effects:
            name, info = large_effects[0]
            recommendations.append(
                f"Обнаружен {info['effect_size']} размер эффекта (Cohen's d={info['cohens_d']:.3f}) "
                f"между группами. Это практически значимое различие."
            )
        
        # Анализ ранжирования
        ranking_analysis = analysis_result.get("ranking_analysis", {})
        for ranking_name, ranking_info in ranking_analysis.items():
            if isinstance(ranking_info, dict) and "ranking" in str(ranking_info):
                # Это групповое ранжирование
                for var, var_ranking in ranking_info.items():
                    if isinstance(var_ranking, dict) and "leader" in var_ranking:
                        leader = var_ranking["leader"]
                        outsider = var_ranking["outsider"]
                        if leader != outsider:
                            recommendations.append(
                                f"По показателю '{var}' лидирует группа '{leader}', "
                                f"аутсайдер - '{outsider}'. Анализируйте факторы успеха."
                            )
                            break
        
        return recommendations[:5]


# === ФАБРИЧНЫЕ ФУНКЦИИ ===

def get_statistical_tool() -> StatisticalTool:
    """Получение экземпляра StatisticalTool."""
    return StatisticalTool()


def get_correlation_tool() -> CorrelationTool:
    """Получение экземпляра CorrelationTool."""
    return CorrelationTool()


def get_trend_tool() -> TrendTool:
    """Получение экземпляра TrendTool."""
    return TrendTool()


def get_anomaly_tool() -> AnomalyTool:
    """Получение экземпляра AnomalyTool."""
    return AnomalyTool()


def get_comparison_tool() -> ComparisonTool:
    """Получение экземпляра ComparisonTool."""
    return ComparisonTool()


def get_all_specialized_tools() -> List[BaseTool]:
    """Получение всех специализированных инструментов."""
    return [
        get_statistical_tool(),
        get_correlation_tool(),
        get_trend_tool(),
        get_anomaly_tool(),
        get_comparison_tool()
    ] 