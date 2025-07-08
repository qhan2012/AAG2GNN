"""
Feature Extractor for AAG-to-GNN conversion.

Adds comprehensive node, edge, and graph-level features to PyTorch Geometric
graphs.
"""

import torch
from torch_geometric.data import Data
from typing import Dict
from .utils import (
    literal_to_node, 
    topological_sort, 
    get_node_types, 
    compute_fanin_fanout,
    is_inverted
)


def add_node_and_edge_features(data: Data, parsed: Dict) -> Data:
    """
    Add comprehensive node, edge, and graph-level features to the graph.
    
    Args:
        data (torch_geometric.data.Data): Graph from build_graph()
        parsed (dict): Parsed AAG data
        
    Returns:
        torch_geometric.data.Data: Graph with all features added
    """
    # Get node mappings
    node_to_idx = data.node_to_idx
    idx_to_node = data.idx_to_node
    
    # Compute node features
    node_features = _compute_node_features(data, parsed, node_to_idx, idx_to_node)
    
    # Compute edge features
    edge_features = _compute_edge_features(data, parsed, node_to_idx, idx_to_node)
    
    # Compute graph-level features
    global_features = _compute_global_features(data, parsed, node_to_idx, idx_to_node)
    
    # Update the data object
    data.x = node_features
    data.edge_attr = edge_features
    data.global_x = global_features
    
    return data


def _compute_node_features(data: Data, parsed: Dict, node_to_idx: Dict, idx_to_node: Dict) -> torch.Tensor:
    """
    Compute node features for each node.
    
    Features: [is_input, is_output, is_and, fanin, fanout, level]
    """
    num_nodes = data.num_nodes
    node_features = torch.zeros((num_nodes, 6), dtype=torch.float)
    
    # Get node types and topological levels
    node_types = get_node_types(parsed)
    levels = topological_sort(parsed)
    fanin_dict, fanout_dict = compute_fanin_fanout(parsed)
    
    for idx in range(num_nodes):
        node = idx_to_node[idx]
        
        # Node type features
        node_type = node_types.get(node, "unknown")
        node_features[idx, 0] = 1.0 if node_type == "input" else 0.0  # is_input
        node_features[idx, 1] = 1.0 if node_type == "output" else 0.0  # is_output
        node_features[idx, 2] = 1.0 if node_type == "and_gate" else 0.0  # is_and
        
        # Connectivity features
        node_features[idx, 3] = float(fanin_dict.get(node, 0))  # fanin
        node_features[idx, 4] = float(fanout_dict.get(node, 0))  # fanout
        
        # Topological level
        node_features[idx, 5] = float(levels.get(node, 0))  # level
    
    return node_features


def _compute_edge_features(data: Data, parsed: Dict, node_to_idx: Dict, idx_to_node: Dict) -> torch.Tensor:
    """
    Compute edge features for each edge.
    
    Features: [is_inverted, level_diff]
    """
    edge_index = data.edge_index
    num_edges = edge_index.size(1)
    edge_features = torch.zeros((num_edges, 2), dtype=torch.float)
    
    # Get topological levels
    levels = topological_sort(parsed)
    
    # Get original edge attributes if they exist
    original_edge_attr = data.edge_attr
    
    for edge_idx in range(num_edges):
        src_idx = edge_index[0, edge_idx].item()
        dst_idx = edge_index[1, edge_idx].item()
        
        src_node = idx_to_node[src_idx]
        dst_node = idx_to_node[dst_idx]
        
        # Find the corresponding literal for this edge
        is_inverted_edge = 0.0
        for lhs, rhs0, rhs1 in parsed["and_gates"]:
            lhs_node = literal_to_node(lhs)
            rhs0_node = literal_to_node(rhs0)
            rhs1_node = literal_to_node(rhs1)
            
            if lhs_node == dst_node:
                if rhs0_node == src_node:
                    is_inverted_edge = 1.0 if is_inverted(rhs0) else 0.0
                    break
                elif rhs1_node == src_node:
                    is_inverted_edge = 1.0 if is_inverted(rhs1) else 0.0
                    break
        
        # Level difference
        src_level = levels.get(src_node, 0)
        dst_level = levels.get(dst_node, 0)
        level_diff = dst_level - src_level
        
        edge_features[edge_idx, 0] = is_inverted_edge
        edge_features[edge_idx, 1] = float(level_diff)
    
    return edge_features


def _compute_global_features(data: Data, parsed: Dict, node_to_idx: Dict, idx_to_node: Dict) -> torch.Tensor:
    """
    Compute graph-level features.
    
    Features: [num_nodes, num_edges, num_inputs, num_outputs, avg_fanin, max_level]
    """
    num_nodes = data.num_nodes
    num_edges = data.edge_index.size(1)
    num_inputs = len(parsed["inputs"])
    num_outputs = len(parsed["outputs"])
    
    # Compute average fanin
    fanin_dict, _ = compute_fanin_fanout(parsed)
    and_gate_nodes = [literal_to_node(lhs) for lhs, _, _ in parsed["and_gates"]]
    
    if and_gate_nodes:
        total_fanin = sum(fanin_dict.get(node, 0) for node in and_gate_nodes)
        avg_fanin = total_fanin / len(and_gate_nodes)
    else:
        avg_fanin = 0.0

    # Compute max topological level
    levels = topological_sort(parsed)
    max_level = max(levels.values()) if levels else 0.0

    global_features = torch.tensor([
        float(num_nodes),
        float(num_edges),
        float(num_inputs),
        float(num_outputs),
        avg_fanin,
        float(max_level)
    ], dtype=torch.float)
    
    return global_features 