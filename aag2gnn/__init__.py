"""
AAG-to-GNN Graph Conversion and Feature Extraction Library

This library transforms AAG (And-Inverter Graph) netlists into fully enriched
PyTorch Geometric graph representations with node, edge, and graph-level features.
"""

from .parser import parse_aag
from .graph_builder import build_graph
from .feature_extractor import add_node_and_edge_features

__version__ = "1.0.0"
__all__ = [
    "parse_aag",
    "build_graph", 
    "add_node_and_edge_features",
    "load_aag_as_gnn_graph"
]


def load_aag_as_gnn_graph(file_path: str, include_inverter: bool = True):
    """
    One-step pipeline to load an AAG file and convert it to a PyG graph.
    
    Args:
        file_path (str): Path to the .aag file
        include_inverter (bool): Whether to include inverter edge features
        
    Returns:
        torch_geometric.data.Data: Fully enriched graph with all features
    """
    parsed = parse_aag(file_path)
    data = build_graph(parsed, include_inverter=include_inverter)
    data = add_node_and_edge_features(data, parsed)
    return data 