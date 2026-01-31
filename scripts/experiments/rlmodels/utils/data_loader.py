"""
Data Loader for loading experiment results and preparing data for visualization
"""

import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np


class DataLoader:
    """Loads and prepares experiment data for analysis and visualization"""
    
    def __init__(self, results_dir: str = "cobotproject/scripts/experiments/rlmodels/results"):
        """
        Initialize DataLoader
        
        Args:
            results_dir: Directory containing experiment results
        """
        self.results_dir = Path(results_dir)
        
    def load_experiment_results(self, experiment_id: str, format: str = 'csv') -> pd.DataFrame:
        """
        Load results for a specific experiment
        
        Args:
            experiment_id: Experiment ID (e.g., 'exp1')
            format: File format ('csv' or 'json')
            
        Returns:
            DataFrame containing experiment results
        """
        exp_dir = self.results_dir / experiment_id
        
        if format == 'csv':
            results_file = exp_dir / "results.csv"
            if results_file.exists():
                return pd.read_csv(results_file)
        elif format == 'json':
            results_file = exp_dir / "results.json"
            if results_file.exists():
                with open(results_file, 'r') as f:
                    data = json.load(f)
                return pd.DataFrame(data)
        
        raise FileNotFoundError(f"Results file not found for {experiment_id}")
    
    def load_model_results(self, experiment_id: str, model_name: str) -> pd.DataFrame:
        """
        Load results for a specific model within an experiment
        
        Args:
            experiment_id: Experiment ID
            model_name: Name of the model
            
        Returns:
            DataFrame containing model results
        """
        df = self.load_experiment_results(experiment_id)
        return df[df['model_name'] == model_name]
    
    def load_all_experiments(self) -> Dict[str, pd.DataFrame]:
        """
        Load results from all experiments
        
        Returns:
            Dictionary mapping experiment IDs to DataFrames
        """
        results = {}
        
        if not self.results_dir.exists():
            return results
            
        for exp_dir in self.results_dir.iterdir():
            if exp_dir.is_dir():
                try:
                    df = self.load_experiment_results(exp_dir.name)
                    results[exp_dir.name] = df
                except FileNotFoundError:
                    continue
                    
        return results
    
    def get_metric_summary(self, experiment_id: str, metric: str) -> pd.DataFrame:
        """
        Get summary statistics for a specific metric
        
        Args:
            experiment_id: Experiment ID
            metric: Metric name (e.g., 'success_rate', 'avg_reward')
            
        Returns:
            DataFrame with summary statistics per model
        """
        df = self.load_experiment_results(experiment_id)
        
        summary = df.groupby('model_name')[metric].agg([
            'mean', 'std', 'min', 'max', 'count'
        ]).reset_index()
        
        # Calculate confidence interval (95%)
        summary['ci_lower'] = summary['mean'] - 1.96 * summary['std'] / np.sqrt(summary['count'])
        summary['ci_upper'] = summary['mean'] + 1.96 * summary['std'] / np.sqrt(summary['count'])
        
        return summary
    
    def get_episode_data(self, experiment_id: str, model_name: str) -> pd.DataFrame:
        """
        Get per-episode data for a model
        
        Args:
            experiment_id: Experiment ID
            model_name: Model name
            
        Returns:
            DataFrame with per-episode metrics
        """
        return self.load_model_results(experiment_id, model_name)
    
    def prepare_comparison_data(self, experiment_id: str, metric: str) -> pd.DataFrame:
        """
        Prepare data for cross-model comparison
        
        Args:
            experiment_id: Experiment ID
            metric: Metric to compare
            
        Returns:
            DataFrame formatted for comparison plots
        """
        df = self.load_experiment_results(experiment_id)
        
        # Reshape for plotting
        comparison_df = df[['model_name', 'episode', metric]].copy()
        
        return comparison_df
    
    def save_results(self, data: Union[pd.DataFrame, Dict], 
                    experiment_id: str, 
                    filename: str,
                    format: str = 'csv'):
        """
        Save results to file
        
        Args:
            data: Data to save (DataFrame or dict)
            experiment_id: Experiment ID
            filename: Output filename
            format: File format ('csv' or 'json')
        """
        exp_dir = self.results_dir / experiment_id
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = exp_dir / filename
        
        if format == 'csv':
            if isinstance(data, pd.DataFrame):
                data.to_csv(output_path, index=False)
            else:
                pd.DataFrame(data).to_csv(output_path, index=False)
        elif format == 'json':
            with open(output_path, 'w') as f:
                if isinstance(data, pd.DataFrame):
                    json.dump(data.to_dict(orient='records'), f, indent=2)
                else:
                    json.dump(data, f, indent=2)

