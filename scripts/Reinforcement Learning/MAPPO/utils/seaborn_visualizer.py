"""
Seaborn Visualization Utilities for WandB Logging

Generates publication-quality plots with confidence intervals using Seaborn.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional
import wandb


class SeabornVisualizer:
    """
    Creates Seaborn visualizations for training metrics.
    """
    
    def __init__(self, style: str = "whitegrid"):
        """
        Initialize visualizer.
        
        Args:
            style: Seaborn style ('whitegrid', 'darkgrid', 'white', 'dark', 'ticks')
        """
        sns.set_style(style)
        sns.set_palette("husl")
        
    def plot_learning_curve(
        self,
        data: Dict[str, List[float]],
        title: str,
        xlabel: str = "Iterations",
        ylabel: str = "Reward",
        window_size: int = 10,
        confidence_interval: float = 0.95,
    ) -> plt.Figure:
        """
        Create a learning curve plot with confidence intervals.
        
        Args:
            data: Dictionary mapping label -> list of values
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            window_size: Rolling window size for smoothing
            confidence_interval: Confidence interval (0.0-1.0)
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for label, values in data.items():
            if len(values) == 0:
                continue
                
            x = np.arange(len(values))
            y = np.array(values)
            
            # Calculate rolling mean and std
            if len(values) >= window_size:
                rolling_mean = np.convolve(y, np.ones(window_size)/window_size, mode='valid')
                rolling_std = np.array([
                    np.std(y[max(0, i-window_size):i+1]) 
                    for i in range(window_size-1, len(y))
                ])
                x_smooth = x[window_size-1:]
            else:
                rolling_mean = y
                rolling_std = np.zeros_like(y)
                x_smooth = x
            
            # Calculate confidence interval
            ci_multiplier = 1.96 if confidence_interval == 0.95 else 2.576  # 95% or 99%
            ci_lower = rolling_mean - ci_multiplier * rolling_std
            ci_upper = rolling_mean + ci_multiplier * rolling_std
            
            # Plot
            ax.plot(x_smooth, rolling_mean, label=label, linewidth=2)
            ax.fill_between(x_smooth, ci_lower, ci_upper, alpha=0.3)
        
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_distribution(
        self,
        data: Dict[str, List[float]],
        title: str,
        xlabel: str = "Value",
        ylabel: str = "Probability",
    ) -> plt.Figure:
        """
        Create a distribution plot (KDE) for multiple datasets.
        
        Args:
            data: Dictionary mapping label -> list of values
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for label, values in data.items():
            if len(values) == 0:
                continue
            sns.kdeplot(data=values, label=label, ax=ax, fill=True, alpha=0.5)
        
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def log_to_wandb(self, fig: plt.Figure, name: str, step: int):
        """
        Log figure to WandB.
        
        Args:
            fig: Matplotlib figure
            name: Name for the plot in WandB
            step: Current training step
        """
        wandb.log({name: wandb.Image(fig)}, step=step)
        plt.close(fig)
    
    def create_and_log_learning_curve(
        self,
        rewards: List[float],
        name: str,
        step: int,
        title: str = "Learning Curve",
        window_size: int = 10,
    ):
        """
        Convenience method to create and log a learning curve in one call.
        
        Args:
            rewards: List of reward values
            name: Name for the plot in WandB
            step: Current training step
            title: Plot title
            window_size: Rolling window size
        """
        fig = self.plot_learning_curve(
            data={"Reward": rewards},
            title=title,
            ylabel="Reward",
            window_size=window_size,
        )
        self.log_to_wandb(fig, name, step)

