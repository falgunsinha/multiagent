"""
Visualization modules for RL experiments
"""

# Individual model visualizations
from .individual_models.seaborn import (
    plot_success_rate_seaborn,
    plot_reward_distribution_seaborn,
    plot_steps_distribution_seaborn,
    plot_multi_metric_dashboard_seaborn,
    plot_performance_radar_seaborn,
    plot_pairplot_matrix_seaborn,
    plot_corner_pairplot_seaborn,
    plot_custom_pairplot_grid,
    plot_regression_pairplot,
    plot_timeseries_with_ci_seaborn,
    plot_learning_curves_with_ci,
    plot_prediction_intervals
)

from .individual_models.plotly import (
    plot_success_rate_plotly,
    plot_reward_distribution_plotly,
    plot_steps_distribution_plotly,
    plot_multi_metric_dashboard_plotly,
    plot_performance_radar_plotly,
    plot_box_with_points_plotly,
    plot_multi_metric_boxes_plotly,
    plot_histogram_with_kde_plotly,
    plot_multi_histogram_kde_grid,
    plot_timeseries_with_ci_plotly,
    plot_prediction_intervals_plotly
)

# Cross-model visualizations
from .cross_model.seaborn import (
    plot_grouped_bars_with_ci_seaborn,
    plot_multi_metric_grouped_bars,
    plot_dataset_comparison_bars
)

from .cross_model.plotly import (
    plot_grouped_bars_with_ci_plotly,
    plot_multi_metric_grouped_bars_plotly
)

__all__ = [
    # Seaborn individual
    'plot_success_rate_seaborn',
    'plot_reward_distribution_seaborn',
    'plot_steps_distribution_seaborn',
    'plot_multi_metric_dashboard_seaborn',
    'plot_performance_radar_seaborn',
    'plot_pairplot_matrix_seaborn',
    'plot_corner_pairplot_seaborn',
    'plot_custom_pairplot_grid',
    'plot_regression_pairplot',
    'plot_timeseries_with_ci_seaborn',
    'plot_learning_curves_with_ci',
    'plot_prediction_intervals',
    # Plotly individual
    'plot_success_rate_plotly',
    'plot_reward_distribution_plotly',
    'plot_steps_distribution_plotly',
    'plot_multi_metric_dashboard_plotly',
    'plot_performance_radar_plotly',
    'plot_box_with_points_plotly',
    'plot_multi_metric_boxes_plotly',
    'plot_histogram_with_kde_plotly',
    'plot_multi_histogram_kde_grid',
    'plot_timeseries_with_ci_plotly',
    'plot_prediction_intervals_plotly',
    # Seaborn cross-model
    'plot_grouped_bars_with_ci_seaborn',
    'plot_multi_metric_grouped_bars',
    'plot_dataset_comparison_bars',
    # Plotly cross-model
    'plot_grouped_bars_with_ci_plotly',
    'plot_multi_metric_grouped_bars_plotly',
]

