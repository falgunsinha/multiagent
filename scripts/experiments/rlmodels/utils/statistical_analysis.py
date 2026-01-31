"""
Statistical Analysis utilities for experiment results
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple, List, Dict, Optional


class StatisticalAnalysis:
    """Provides statistical analysis functions for experiment results"""
    
    @staticmethod
    def calculate_confidence_interval(data: np.ndarray, 
                                     confidence: float = 0.95) -> Tuple[float, float, float]:
        """
        Calculate mean and confidence interval
        
        Args:
            data: Array of values
            confidence: Confidence level (default 0.95 for 95% CI)
            
        Returns:
            Tuple of (mean, lower_bound, upper_bound)
        """
        mean = np.mean(data)
        std_err = stats.sem(data)
        margin = std_err * stats.t.ppf((1 + confidence) / 2, len(data) - 1)
        
        return mean, mean - margin, mean + margin
    
    @staticmethod
    def calculate_percentiles(data: np.ndarray, 
                            percentiles: List[float] = [10, 90]) -> Dict[str, float]:
        """
        Calculate percentiles for data
        
        Args:
            data: Array of values
            percentiles: List of percentiles to calculate
            
        Returns:
            Dictionary mapping percentile to value
        """
        result = {}
        for p in percentiles:
            result[f'p{int(p)}'] = np.percentile(data, p)
        return result
    
    @staticmethod
    def compare_models(model1_data: np.ndarray, 
                      model2_data: np.ndarray,
                      test: str = 't-test') -> Dict[str, float]:
        """
        Statistical comparison between two models
        
        Args:
            model1_data: Results from model 1
            model2_data: Results from model 2
            test: Statistical test to use ('t-test' or 'mann-whitney')
            
        Returns:
            Dictionary with test statistic and p-value
        """
        if test == 't-test':
            statistic, p_value = stats.ttest_ind(model1_data, model2_data)
        elif test == 'mann-whitney':
            statistic, p_value = stats.mannwhitneyu(model1_data, model2_data)
        else:
            raise ValueError(f"Unknown test: {test}")
            
        return {
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    @staticmethod
    def calculate_effect_size(model1_data: np.ndarray, 
                            model2_data: np.ndarray) -> float:
        """
        Calculate Cohen's d effect size
        
        Args:
            model1_data: Results from model 1
            model2_data: Results from model 2
            
        Returns:
            Cohen's d effect size
        """
        mean1, mean2 = np.mean(model1_data), np.mean(model2_data)
        std1, std2 = np.std(model1_data, ddof=1), np.std(model2_data, ddof=1)
        n1, n2 = len(model1_data), len(model2_data)
        
        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
        
        return (mean1 - mean2) / pooled_std
    
    @staticmethod
    def calculate_summary_stats(data: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive summary statistics
        
        Args:
            data: Array of values
            
        Returns:
            Dictionary of statistics
        """
        return {
            'mean': np.mean(data),
            'median': np.median(data),
            'std': np.std(data, ddof=1),
            'min': np.min(data),
            'max': np.max(data),
            'q25': np.percentile(data, 25),
            'q75': np.percentile(data, 75),
            'count': len(data)
        }
    
    @staticmethod
    def rank_models(df: pd.DataFrame, metric: str, ascending: bool = False) -> pd.DataFrame:
        """
        Rank models by a metric
        
        Args:
            df: DataFrame with model results
            metric: Metric to rank by
            ascending: Whether lower is better
            
        Returns:
            DataFrame with rankings
        """
        summary = df.groupby('model_name')[metric].mean().reset_index()
        summary = summary.sort_values(metric, ascending=ascending)
        summary['rank'] = range(1, len(summary) + 1)
        
        return summary
    
    @staticmethod
    def moving_average(data: np.ndarray, window: int = 10) -> np.ndarray:
        """
        Calculate moving average
        
        Args:
            data: Array of values
            window: Window size
            
        Returns:
            Moving average array
        """
        return np.convolve(data, np.ones(window)/window, mode='valid')
    
    @staticmethod
    def normalize_metrics(df: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
        """
        Normalize metrics to 0-100 scale
        
        Args:
            df: DataFrame with metrics
            metrics: List of metric columns to normalize
            
        Returns:
            DataFrame with normalized metrics
        """
        df_norm = df.copy()
        
        for metric in metrics:
            min_val = df[metric].min()
            max_val = df[metric].max()
            
            if max_val > min_val:
                df_norm[f'{metric}_normalized'] = 100 * (df[metric] - min_val) / (max_val - min_val)
            else:
                df_norm[f'{metric}_normalized'] = 50.0
                
        return df_norm

