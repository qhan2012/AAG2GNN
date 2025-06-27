"""
Graph Builder for AAG-to-GNN conversion.

Converts parsed AAG data into PyTorch Geometric Data objects with basic
edge_index and placeholder node features.
"""

import torch
from torch_geometric.data import Data
from typing import Dict, List, Tuple
from .utils import literal_to_node, is_inverted


def build_graph(parsed: Dict, include_inverter: bool = True) -> Data:
    """
    Build a PyTorch Geometric graph from parsed AAG data.
    
    Args:
        parsed (dict): Parsed AAG data from parse_aag()
        include_inverter (bool): Whether to include inverter edge features
        
    Returns:
        torch_geometric.data.Data: Graph with edge_index and basic node features
    """
    # Collect all nodes
    nodes = set()
    
    # Add input nodes
    for literal in parsed["inputs"]:
        nodes.add(literal_to_node(literal))
    
    # Add output nodes
    for literal in parsed["outputs"]:
        nodes.add(literal_to_node(literal))
    
    # Add AND gate nodes
    for lhs, rhs0, rhs1 in parsed["and_gates"]:
        nodes.add(literal_to_node(lhs))
        nodes.add(literal_to_node(rhs0))
        nodes.add(literal_to_node(rhs1))
    
    # Create node index mapping
    node_to_idx = {node: idx for idx, node in enumerate(sorted(nodes))}
    idx_to_node = {idx: node for node, idx in node_to_idx.items()}
    
    # Build edge list
    edges = []
    edge_attrs = []
    
    # Add edges from AND gates (rhs -> lhs)
    for lhs, rhs0, rhs1 in parsed["and_gates"]:
        lhs_node = literal_to_node(lhs)
        rhs0_node = literal_to_node(rhs0)
        rhs1_node = literal_to_node(rhs1)
        
        # Edge from rhs0 to lhs
        edges.append([node_to_idx[rhs0_node], node_to_idx[lhs_node]])
        if include_inverter:
            edge_attrs.append([1 if is_inverted(rhs0) else 0])
        
        # Edge from rhs1 to lhs
        edges.append([node_to_idx[rhs1_node], node_to_idx[lhs_node]])
        if include_inverter:
            edge_attrs.append([1 if is_inverted(rhs1) else 0])
    
    # Convert to tensors
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    # Create placeholder node features (will be replaced by feature_extractor)
    num_nodes = len(nodes)
    x = torch.zeros((num_nodes, 1), dtype=torch.float)  # Placeholder
    
    # Create edge attributes if requested
    edge_attr = None
    if include_inverter and edge_attrs:
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    
    # Store node mapping for later use
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=num_nodes
    )
    
    # Store metadata for feature extraction
    data.node_to_idx = node_to_idx
    data.idx_to_node = idx_to_node
    
    return data 