# AAG-to-GNN: Graph Conversion and Feature Extraction Library

A Python library that transforms AAG (And-Inverter Graph) netlists into fully enriched PyTorch Geometric graph representations with comprehensive node, edge, and graph-level features.

## 🚀 Features

- **Complete AAG Parser**: Robust parsing of `.aag` files with error handling
- **PyTorch Geometric Integration**: Native compatibility with all PyG-based GNN models
- **Rich Feature Extraction**: 
  - **Node Features**: Type indicators, connectivity metrics, topological levels
  - **Edge Features**: Inversion flags, level differences
  - **Graph Features**: Global statistics and metrics
- **Offline Feature Extraction**: Save features to separate files (CSV, JSON, NumPy) for analysis
- **One-Step Pipeline**: Simple API for quick conversion
- **Modular Design**: Use individual components or the complete pipeline

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/qhan2012/AAG2GNN.git
cd aag2gnn

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

- `torch >= 1.9.0`
- `torch-geometric >= 2.0.0`
- `numpy >= 1.21.0`
- `pandas >= 1.3.0`

## 🎯 Quick Start

```python
from aag2gnn import load_aag_as_gnn_graph

# One-step conversion
data = load_aag_as_gnn_graph("circuit.aag")

# Use with any PyG GNN model
print(f"Nodes: {data.x.shape}")
print(f"Edges: {data.edge_index.shape}")
print(f"Edge features: {data.edge_attr.shape}")
print(f"Global features: {data.global_x}")
```

## 📚 API Reference

### Main Functions

#### `load_aag_as_gnn_graph(file_path, include_inverter=True)`
One-step pipeline to convert AAG to PyG graph.

**Parameters:**
- `file_path` (str): Path to the `.aag` file
- `include_inverter` (bool): Include inverter edge features

**Returns:**
- `torch_geometric.data.Data`: Fully enriched graph with 6 global features (see above)

#### `parse_aag(file_path)`
Parse an AAG file and extract structure.

**Returns:**
```python
{
    "max_var": int,
    "inputs": List[int],
    "outputs": List[int], 
    "and_gates": List[Tuple[int, int, int]],  # (lhs, rhs0, rhs1)
    "comments": Optional[List[str]]
}
```

#### `build_graph(parsed, include_inverter=True)`
Build basic PyG graph from parsed data.

#### `add_node_and_edge_features(data, parsed)`
Add comprehensive features to the graph.

#### `extract_features_to_files(aag_file_path, output_dir, include_inverter=True, formats=None, node_mapping=True)`
Extract features from an AAG file and save them to separate files.

**Parameters:**
- `aag_file_path` (str): Path to the `.aag` file
- `output_dir` (str): Directory to save the extracted features
- `include_inverter` (bool): Whether to include inverter edge features
- `formats` (list, optional): List of output formats. Defaults to `['csv', 'json', 'npy']`
- `node_mapping` (bool): Whether to save node mapping information

**Returns:**
- `Dict[str, str]`: Dictionary mapping feature types to their file paths

#### `load_features_from_files(base_path, formats=None)`
Load features from saved files.

**Parameters:**
- `base_path` (str): Base path to the saved feature files
- `formats` (list, optional): List of formats to load. Defaults to `['npy']`

**Returns:**
- `Dict`: Dictionary containing loaded features

## 🧠 Feature Details

### Node Features (`data.x`)
Each node has 6 features: `[is_input, is_output, is_and, fanin, fanout, level]`

| Feature | Type | Description |
|---------|------|-------------|
| `is_input` | int | 1 if node is primary input |
| `is_output` | int | 1 if node is primary output |
| `is_and` | int | 1 if node is AND gate |
| `fanin` | int | Number of incoming edges |
| `fanout` | int | Number of outgoing edges |
| `level` | int | Topological level |

### Edge Features (`data.edge_attr`)
Each edge has 2 features: `[is_inverted, level_diff]`

| Feature | Type | Description |
|---------|------|-------------|
| `is_inverted` | int | 1 if source literal is inverted |
| `level_diff` | int | `level(dst) - level(src)` |

### Graph Features (`data.global_x`)

- Global features: `[num_nodes, num_edges, num_inputs, num_outputs, avg_fanin, max_level]`

| Feature      | Type  | Description                                 |
|------------- |------ |---------------------------------------------|
| num_nodes    | int   | Number of nodes in the graph                |
| num_edges    | int   | Number of edges in the graph                |
| num_inputs   | int   | Number of primary input nodes               |
| num_outputs  | int   | Number of primary output nodes              |
| avg_fanin    | float | Average number of incoming edges per node   |
| max_level    | int   | Maximum topological level in the graph      |

## 🔧 Usage Examples

### Basic Usage

```python
from aag2gnn import load_aag_as_gnn_graph

# Convert AAG to PyG graph
data = load_aag_as_gnn_graph("c17.aag")

# Access features
print(f"Node features: {data.x.shape}")
print(f"Edge features: {data.edge_attr.shape}")
print(f"Global features: {data.global_x}")  # Now has 6 features
```

### Step-by-Step Pipeline

```python
from aag2gnn import parse_aag, build_graph, add_node_and_edge_features

# Step 1: Parse AAG
parsed = parse_aag("c17.aag")

# Step 2: Build basic graph
data = build_graph(parsed, include_inverter=True)

# Step 3: Add features
data = add_node_and_edge_features(data, parsed)
```

### GNN Model Integration

```python
import torch
from torch_geometric.nn import GCNConv
from aag2gnn import load_aag_as_gnn_graph

# Load graph
data = load_aag_as_gnn_graph("circuit.aag")

# Define GNN model
class CircuitGNN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.classifier = torch.nn.Linear(hidden_channels, num_classes)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        return self.classifier(x)

# Create and use model
model = CircuitGNN(
    num_features=data.x.size(1),  # 6 node features
    hidden_channels=64,
    num_classes=2
)

output = model(data.x, data.edge_index)
```

### Offline Feature Extraction

```python
from aag2gnn import extract_features_to_files, load_features_from_files

# Extract features to files
saved_files = extract_features_to_files(
    aag_file_path="circuit.aag",
    output_dir="extracted_features",
    formats=['csv', 'json', 'npy'],
    node_mapping=True
)

# Files saved:
# - circuit_node_features.csv/json/npy
# - circuit_edge_features.csv/json/npy  
# - circuit_graph_features.csv/json/npy
# - circuit_node_mapping.csv/json

# Load features back
loaded_features = load_features_from_files(
    base_path="extracted_features/circuit",
    formats=['npy']
)

print(f"Node features shape: {loaded_features['node'].shape}")
print(f"Edge features shape: {loaded_features['edge'].shape}")
print(f"Graph features: {loaded_features['graph']}")
```

## 📁 Project Structure

```
aag2gnn/
├── __init__.py              # Main package with one-step pipeline
├── parser.py                # AAG file parser
├── graph_builder.py         # PyG graph construction
├── feature_extractor.py     # Feature computation
├── offline_extractor.py     # Offline feature extraction
├── utils.py                 # Helper utilities
├── example.py               # Usage examples
├── offline_example.py       # Offline extraction examples
├── requirements.txt         # Dependencies
└── README.md               # This file
```

## 🧪 Testing

Run the example script to test the library:

```bash
python aag2gnn/example.py
```

This will:
1. Create a sample AAG file
2. Demonstrate the conversion pipeline
3. Show feature analysis
4. Test PyG compatibility
5. Demonstrate GNN model integration

## 📋 AAG File Format

The library supports the standard AAG format:

```
aag max_var num_inputs num_latches num_outputs num_and_gates
input1
input2
...
latch1
latch2
...
output1
output2
...
and_gate1_lhs and_gate1_rhs0 and_gate1_rhs1
and_gate2_lhs and_gate2_rhs0 and_gate2_rhs1
...
c
comments...
```

## 🔄 Compatibility

- **Python**: ≥ 3.8
- **PyTorch**: ≥ 1.9.0
- **PyTorch Geometric**: ≥ 2.0.0
- **GNN Models**: Compatible with GCN, GAT, GIN, GraphSAGE, etc.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- PyTorch Geometric team for the excellent graph neural network framework
- The AAG format specification and community

---

**Ready to convert your AAG circuits to GNN-ready graphs! 🚀** 