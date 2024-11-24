"""
Cross-cluster Weighted Forests

A package implementing the Cross-cluster Weighted Forests method
for ensemble learning across multiple data clusters.
"""

from .models.forest import CrossClusterForest, evaluate_model
from .models.wrapper import SingleDatasetForest  # Add this line
from .data_generation.simulation import sim_data
from .visualization.plots import plot_results, interpret_results

__version__ = "0.1.0"

__all__ = [
    "CrossClusterForest",
    "SingleDatasetForest",  # Add this line
    "evaluate_model",
    "sim_data",
    "plot_results",
    "interpret_results"
]