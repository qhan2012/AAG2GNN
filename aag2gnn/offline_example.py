"""
Example script demonstrating offline feature extraction functionality.
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add the parent directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from aag2gnn import extract_features_to_files, load_features_from_files
except ImportError:
    # Fallback: try importing directly from the module
    from aag2gnn.offline_extractor import extract_features_to_files, load_features_from_files


def create_sample_aag():
    """Create a sample AAG file for demonstration."""
    sample_aag = """aag 7 2 0 2 3
2
3
6
7
6 2 4
7 3 5
8 6 7"""
    
    with open("sample_circuit.aag", "w") as f:
        f.write(sample_aag)
    
    return "sample_circuit.aag"


def main():
    """Demonstrate offline feature extraction."""
    print("=== AAG2GNN Offline Feature Extraction Demo ===\n")
    
    # Create sample AAG file
    aag_file = create_sample_aag()
    print(f"✓ Created sample AAG file: {aag_file}")
    
    # Extract features to files
    output_dir = "extracted_features"
    print(f"\n📁 Extracting features to: {output_dir}")
    
    saved_files = extract_features_to_files(
        aag_file_path=aag_file,
        output_dir=output_dir,
        formats=['csv', 'json', 'npy'],
        node_mapping=True
    )
    
    # Display saved files
    print("\n📄 Saved files:")
    for feature_type, files in saved_files.items():
        print(f"  {feature_type.upper()} features:")
        for fmt, path in files.items():
            print(f"    {fmt.upper()}: {path}")
    
    # Load features back
    print(f"\n🔄 Loading features from files...")
    base_path = Path(output_dir) / "sample_circuit"
    loaded_features = load_features_from_files(str(base_path), formats=['npy'])
    
    # Display loaded features
    print("\n📊 Loaded feature shapes:")
    for feature_type, data in loaded_features.items():
        print(f"  {feature_type}: {data.shape}")
    
    # Show some statistics
    if 'node' in loaded_features:
        node_features = loaded_features['node']
        print(f"\n📈 Node feature statistics:")
        print(f"  Number of nodes: {node_features.shape[0]}")
        print(f"  Features per node: {node_features.shape[1]}")
        print(f"  Input nodes: {np.sum(node_features[:, 0])}")
        print(f"  Output nodes: {np.sum(node_features[:, 1])}")
        print(f"  AND gates: {np.sum(node_features[:, 2])}")
    
    if 'graph' in loaded_features:
        graph_features = loaded_features['graph']
        feature_names = ['num_nodes', 'num_edges', 'num_inputs', 'num_outputs', 'avg_fanin', 'max_level']
        print(f"\n🌐 Graph features:")
        for name, value in zip(feature_names, graph_features[0]):
            print(f"  {name}: {value}")
    
    print(f"\n✅ Demo completed! Check the '{output_dir}' directory for extracted files.")


if __name__ == "__main__":
    main() 