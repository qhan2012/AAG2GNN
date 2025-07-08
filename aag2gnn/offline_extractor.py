"""
Offline Feature Extractor for AAG-to-GNN conversion.

This module provides functionality to extract and save node, edge, and graph-level
features to separate files for offline analysis and processing.
"""

import os
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, Optional, Union
from torch_geometric.data import Data

# Use absolute imports to avoid relative import issues
try:
    from .parser import parse_aag
    from .graph_builder import build_graph
    from .feature_extractor import add_node_and_edge_features
except ImportError:
    # Fallback for direct execution
    from parser import parse_aag
    from graph_builder import build_graph
    from feature_extractor import add_node_and_edge_features


def extract_features_to_files(
    aag_file_path: str,
    output_dir: str,
    include_inverter: bool = True,
    formats: Optional[list] = None,
    node_mapping: bool = True
) -> Dict[str, str]:
    """
    Extract features from an AAG file and save them to separate files.
    
    Args:
        aag_file_path (str): Path to the .aag file
        output_dir (str): Directory to save the extracted features
        include_inverter (bool): Whether to include inverter edge features
        formats (list, optional): List of output formats. Defaults to ['csv', 'json', 'npy']
        node_mapping (bool): Whether to save node mapping information
        
    Returns:
        Dict[str, str]: Dictionary mapping feature types to their file paths
    """
    if formats is None:
        formats = ['csv', 'json', 'npy']
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load and process the AAG file
    parsed = parse_aag(aag_file_path)
    data = build_graph(parsed, include_inverter=include_inverter)
    data = add_node_and_edge_features(data, parsed)
    
    # Extract base filename for output files
    base_name = Path(aag_file_path).stem
    
    # Dictionary to store file paths
    saved_files = {}
    
    # Extract and save node features
    if 'node' in formats or any(fmt in formats for fmt in ['csv', 'json', 'npy']):
        node_files = _save_node_features(data, parsed, output_path, base_name, formats)
        saved_files['node'] = node_files
    
    # Extract and save edge features
    if 'edge' in formats or any(fmt in formats for fmt in ['csv', 'json', 'npy']):
        edge_files = _save_edge_features(data, parsed, output_path, base_name, formats)
        saved_files['edge'] = edge_files
    
    # Extract and save graph features
    if 'graph' in formats or any(fmt in formats for fmt in ['csv', 'json', 'npy']):
        graph_files = _save_graph_features(data, parsed, output_path, base_name, formats)
        saved_files['graph'] = graph_files
    
    # Save node mapping if requested
    if node_mapping:
        mapping_files = _save_node_mapping(data, output_path, base_name, formats)
        saved_files['mapping'] = mapping_files
    
    return saved_files


def _save_node_features(
    data: Data, 
    parsed: Dict, 
    output_path: Path, 
    base_name: str, 
    formats: list
) -> Dict[str, str]:
    """Save node features to files."""
    node_features = data.x.numpy()
    node_to_idx = data.node_to_idx
    idx_to_node = data.idx_to_node
    
    # Create feature names
    feature_names = ['is_input', 'is_output', 'is_and', 'fanin', 'fanout', 'level']
    
    # Create DataFrame for better CSV output
    df = pd.DataFrame(node_features, columns=feature_names)
    df['node_id'] = [idx_to_node[i] for i in range(len(node_features))]
    
    saved_files = {}
    
    if 'csv' in formats:
        csv_path = output_path / f"{base_name}_node_features.csv"
        df.to_csv(csv_path, index=False)
        saved_files['csv'] = str(csv_path)
    
    if 'json' in formats:
        json_path = output_path / f"{base_name}_node_features.json"
        node_data = {
            'features': node_features.tolist(),
            'feature_names': feature_names,
            'node_ids': [idx_to_node[i] for i in range(len(node_features))],
            'num_nodes': len(node_features)
        }
        with open(json_path, 'w') as f:
            json.dump(node_data, f, indent=2)
        saved_files['json'] = str(json_path)
    
    if 'npy' in formats:
        npy_path = output_path / f"{base_name}_node_features.npy"
        np.save(npy_path, node_features)
        saved_files['npy'] = str(npy_path)
    
    return saved_files


def _save_edge_features(
    data: Data, 
    parsed: Dict, 
    output_path: Path, 
    base_name: str, 
    formats: list
) -> Dict[str, str]:
    """Save edge features to files."""
    edge_features = data.edge_attr.numpy()
    edge_index = data.edge_index.numpy()
    node_to_idx = data.node_to_idx
    idx_to_node = data.idx_to_node
    
    # Create feature names
    feature_names = ['is_inverted', 'level_diff']
    
    # Create DataFrame for better CSV output
    df = pd.DataFrame(edge_features, columns=feature_names)
    df['src_node'] = [idx_to_node[edge_index[0, i]] for i in range(edge_index.shape[1])]
    df['dst_node'] = [idx_to_node[edge_index[1, i]] for i in range(edge_index.shape[1])]
    
    saved_files = {}
    
    if 'csv' in formats:
        csv_path = output_path / f"{base_name}_edge_features.csv"
        df.to_csv(csv_path, index=False)
        saved_files['csv'] = str(csv_path)
    
    if 'json' in formats:
        json_path = output_path / f"{base_name}_edge_features.json"
        edge_data = {
            'features': edge_features.tolist(),
            'feature_names': feature_names,
            'edge_index': edge_index.tolist(),
            'src_nodes': [idx_to_node[edge_index[0, i]] for i in range(edge_index.shape[1])],
            'dst_nodes': [idx_to_node[edge_index[1, i]] for i in range(edge_index.shape[1])],
            'num_edges': edge_index.shape[1]
        }
        with open(json_path, 'w') as f:
            json.dump(edge_data, f, indent=2)
        saved_files['json'] = str(json_path)
    
    if 'npy' in formats:
        npy_path = output_path / f"{base_name}_edge_features.npy"
        np.save(npy_path, edge_features)
        saved_files['npy'] = str(npy_path)
    
    return saved_files


def _save_graph_features(
    data: Data, 
    parsed: Dict, 
    output_path: Path, 
    base_name: str, 
    formats: list
) -> Dict[str, str]:
    """Save graph features to files."""
    graph_features = data.global_x.numpy()
    
    # Create feature names
    feature_names = ['num_nodes', 'num_edges', 'num_inputs', 'num_outputs', 'avg_fanin', 'max_level']
    
    # Create DataFrame for better CSV output
    df = pd.DataFrame([graph_features], columns=feature_names)
    
    saved_files = {}
    
    if 'csv' in formats:
        csv_path = output_path / f"{base_name}_graph_features.csv"
        df.to_csv(csv_path, index=False)
        saved_files['csv'] = str(csv_path)
    
    if 'json' in formats:
        json_path = output_path / f"{base_name}_graph_features.json"
        graph_data = {
            'features': graph_features.tolist(),
            'feature_names': feature_names,
            'feature_values': dict(zip(feature_names, graph_features.tolist()))
        }
        with open(json_path, 'w') as f:
            json.dump(graph_data, f, indent=2)
        saved_files['json'] = str(json_path)
    
    if 'npy' in formats:
        npy_path = output_path / f"{base_name}_graph_features.npy"
        np.save(npy_path, graph_features)
        saved_files['npy'] = str(npy_path)
    
    return saved_files


def _save_node_mapping(
    data: Data, 
    output_path: Path, 
    base_name: str, 
    formats: list
) -> Dict[str, str]:
    """Save node mapping information to files."""
    node_to_idx = data.node_to_idx
    idx_to_node = data.idx_to_node
    
    # Create DataFrame for better CSV output
    mapping_data = [(node, idx) for node, idx in node_to_idx.items()]
    df = pd.DataFrame(mapping_data, columns=['node_id', 'index'])
    
    saved_files = {}
    
    if 'csv' in formats:
        csv_path = output_path / f"{base_name}_node_mapping.csv"
        df.to_csv(csv_path, index=False)
        saved_files['csv'] = str(csv_path)
    
    if 'json' in formats:
        json_path = output_path / f"{base_name}_node_mapping.json"
        mapping_data = {
            'node_to_idx': node_to_idx,
            'idx_to_node': idx_to_node,
            'num_nodes': len(node_to_idx)
        }
        with open(json_path, 'w') as f:
            json.dump(mapping_data, f, indent=2)
        saved_files['json'] = str(json_path)
    
    return saved_files


def load_features_from_files(
    base_path: str,
    formats: Optional[list] = None
) -> Dict[str, Union[np.ndarray, pd.DataFrame, Dict]]:
    """
    Load features from saved files.
    
    Args:
        base_path (str): Base path to the saved feature files
        formats (list, optional): List of formats to load. Defaults to ['npy']
        
    Returns:
        Dict: Dictionary containing loaded features
    """
    if formats is None:
        formats = ['npy']
    
    base_path = Path(base_path)
    loaded_features = {}
    
    # Load node features
    if 'npy' in formats and (base_path / f"{base_path.stem}_node_features.npy").exists():
        node_path = base_path / f"{base_path.stem}_node_features.npy"
        loaded_features['node'] = np.load(node_path)
    
    # Load edge features
    if 'npy' in formats and (base_path / f"{base_path.stem}_edge_features.npy").exists():
        edge_path = base_path / f"{base_path.stem}_edge_features.npy"
        loaded_features['edge'] = np.load(edge_path)
    
    # Load graph features
    if 'npy' in formats and (base_path / f"{base_path.stem}_graph_features.npy").exists():
        graph_path = base_path / f"{base_path.stem}_graph_features.npy"
        loaded_features['graph'] = np.load(graph_path)
    
    return loaded_features 