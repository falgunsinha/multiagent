"""
GAT + CVD (Graph Attention Network + Counterfactual Value Decomposition)
Multi-Agent Reinforcement Learning for Spatial Object Rearrangement

This package implements a novel approach combining:
- Graph Attention Networks (GAT) for spatial reasoning
- Counterfactual Value Decomposition (CVD) for credit assignment
- Heterogeneous agents (DDQN + MASAC)
"""

from .gat_encoder import GATLayer, SharedGATEncoder
from .graph_utils import build_graph, compute_edge_features
from .cvd_module import CVDModule
from .gat_policy import GATPolicy

__all__ = [
    'GATLayer',
    'SharedGATEncoder',
    'build_graph',
    'compute_edge_features',
    'CVDModule',
    'GATPolicy',
]

