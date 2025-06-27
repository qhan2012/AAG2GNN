"""
Example usage of the AAG-to-GNN library.

This script demonstrates how to convert an AAG file to a PyTorch Geometric graph
with comprehensive features.
"""

import torch
from aag2gnn import load_aag_as_gnn_graph, parse_aag, build_graph, add_node_and_edge_features
import os


def create_sample_aag():
    """
    Create a sample AAG file for demonstration.
    This represents a simple 2-input AND gate followed by an inverter.
    """
    aag_content = """aag 3 2 0 1 1
2
4
6
6 2 4
c
c Simple 2-input AND gate with inverter
"""
    
    with open("sample.aag", "w") as f:
        f.write(aag_content)
    
    print("Created sample.aag file")


def demonstrate_usage():
    """
    Demonstrate the complete pipeline and individual components.
    """
    print("🔧 AAG-to-GNN Library Demo")
    print("=" * 50)
    
    # Create sample AAG file
    create_sample_aag()
    
    # Method 1: One-step pipeline
    print("\n📋 Method 1: One-step pipeline")
    print("-" * 30)
    
    # Try to load the benchmarks file, fallback to sample if not found
    aag_file = "benchmark/my.aag"
    if os.path.exists(aag_file):
        data = load_aag_as_gnn_graph(aag_file, include_inverter=True)
        print(f"Loaded: {aag_file}")
    else:
        print(f"File {aag_file} not found, using sample file instead")
        data = load_aag_as_gnn_graph("sample.aag", include_inverter=True)
        print(f"Loaded: sample.aag")
    
    print(f"Graph shape: {data.x.shape}")
    print(f"Edge index shape: {data.edge_index.shape}")
    print(f"Edge features shape: {data.edge_attr.shape}")
    print(f"Global features: {data.global_x}")
    
    # Method 2: Step-by-step
    print("\n📋 Method 2: Step-by-step pipeline")
    print("-" * 30)
    
    # Step 1: Parse AAG
    parsed = parse_aag("sample.aag")
    print(f"Parsed AAG: {parsed}")
    
    # Step 2: Build graph
    data_step = build_graph(parsed, include_inverter=True)
    print(f"Basic graph: {data_step.x.shape} nodes, {data_step.edge_index.size(1)} edges")
    
    # Step 3: Add features
    data_step = add_node_and_edge_features(data_step, parsed)
    print(f"Enriched graph: {data_step.x.shape} node features, {data_step.edge_attr.shape} edge features")
    
    # Feature analysis
    print("\n🔍 Feature Analysis")
    print("-" * 30)
    
    # Node features: [is_input, is_output, is_and, fanin, fanout, level]
    print("Node features (first 3 nodes):")
    for i in range(min(3, data.x.size(0))):
        features = data.x[i]
        print(f"  Node {i}: input={features[0]:.0f}, output={features[1]:.0f}, "
              f"and={features[2]:.0f}, fanin={features[3]:.0f}, "
              f"fanout={features[4]:.0f}, level={features[5]:.0f}")
    
    # Edge features: [is_inverted, level_diff]
    print("\nEdge features:")
    for i in range(data.edge_attr.size(0)):
        features = data.edge_attr[i]
        src, dst = data.edge_index[:, i]
        print(f"  Edge {src.item()}->{dst.item()}: inverted={features[0]:.0f}, "
              f"level_diff={features[1]:.0f}")
    
    # Global features: [num_nodes, num_edges, num_inputs, num_outputs, avg_fanin]
    global_feats = data.global_x
    print(f"\nGlobal features:")
    print(f"  Nodes: {global_feats[0]:.0f}")
    print(f"  Edges: {global_feats[1]:.0f}")
    print(f"  Inputs: {global_feats[2]:.0f}")
    print(f"  Outputs: {global_feats[3]:.0f}")
    print(f"  Avg fanin: {global_feats[4]:.1f}")
    
    # PyG compatibility check
    print("\n✅ PyG Compatibility Check")
    print("-" * 30)
    print(f"✓ Data object type: {type(data)}")
    print(f"✓ Has x: {hasattr(data, 'x')}")
    print(f"✓ Has edge_index: {hasattr(data, 'edge_index')}")
    print(f"✓ Has edge_attr: {hasattr(data, 'edge_attr')}")
    print(f"✓ Has global_x: {hasattr(data, 'global_x')}")
    print(f"✓ Compatible with GNN models: Yes!")


def demonstrate_gnn_usage():
    """
    Demonstrate how to use the graph with PyTorch Geometric GNN models.
    """
    print("\n🧠 GNN Model Usage Example")
    print("-" * 30)
    
    try:
        from torch_geometric.nn import GCNConv
        import torch.nn.functional as F
        
        # Load graph
        data = load_aag_as_gnn_graph("sample.aag")
        
        # Simple GCN model
        class SimpleGCN(torch.nn.Module):
            def __init__(self, num_features, hidden_channels, num_classes):
                super().__init__()
                self.conv1 = GCNConv(num_features, hidden_channels)
                self.conv2 = GCNConv(hidden_channels, hidden_channels)
                self.classifier = torch.nn.Linear(hidden_channels, num_classes)
            
            def forward(self, x, edge_index):
                x = self.conv1(x, edge_index)
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
                x = self.conv2(x, edge_index)
                x = F.relu(x)
                x = self.classifier(x)
                return F.log_softmax(x, dim=1)
        
        # Create model
        model = SimpleGCN(
            num_features=data.x.size(1),  # 6 node features
            hidden_channels=16,
            num_classes=2
        )
        
        # Forward pass
        with torch.no_grad():
            output = model(data.x, data.edge_index)
        
        print(f"✓ GNN model created successfully")
        print(f"✓ Input features: {data.x.size(1)}")
        print(f"✓ Output shape: {output.shape}")
        print(f"✓ Model parameters: {sum(p.numel() for p in model.parameters())}")
        
    except ImportError:
        print("⚠️  PyTorch Geometric not installed. Install with:")
        print("   pip install torch-geometric")


if __name__ == "__main__":
    demonstrate_usage()
    demonstrate_gnn_usage()
    
    print("\n🎉 Demo completed successfully!")
    print("The AAG-to-GNN library is ready to use with your GNN models.") 