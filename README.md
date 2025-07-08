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

The library provides comprehensive offline feature extraction capabilities, allowing you to save and load features in multiple formats for analysis, visualization, and machine learning workflows.

#### Basic Feature Extraction

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

#### Advanced Feature Extraction Options

```python
# Extract only specific formats
saved_files = extract_features_to_files(
    aag_file_path="large_circuit.aag",
    output_dir="features",
    formats=['csv', 'npy'],  # Skip JSON for large files
    node_mapping=False,      # Don't save node mapping
    include_inverter=False   # Skip inverter edge features
)

# Extract to custom directory structure
saved_files = extract_features_to_files(
    aag_file_path="benchmark/c17.aag",
    output_dir="results/features/c17",
    formats=['json']  # Only JSON format
)
```

#### Working with Extracted Features

**CSV Format** - Human-readable with headers:
```python
import pandas as pd

# Load node features as DataFrame
node_df = pd.read_csv("extracted_features/circuit_node_features.csv")
print(node_df.head())

# Analyze node types
input_nodes = node_df[node_df['is_input'] == 1.0]
and_gates = node_df[node_df['is_and'] == 1.0]
print(f"Input nodes: {len(input_nodes)}")
print(f"AND gates: {len(and_gates)}")
```

**JSON Format** - Structured data with metadata:
```python
import json

# Load graph features
with open("extracted_features/circuit_graph_features.json", 'r') as f:
    graph_data = json.load(f)

print("Graph statistics:")
for name, value in graph_data['feature_values'].items():
    print(f"  {name}: {value}")

# Load node mapping
with open("extracted_features/circuit_node_mapping.json", 'r') as f:
    mapping_data = json.load(f)

print(f"Node to index mapping: {mapping_data['node_to_idx']}")
```

**NumPy Format** - Efficient binary storage:
```python
import numpy as np

# Load features for machine learning
node_features = np.load("extracted_features/circuit_node_features.npy")
edge_features = np.load("extracted_features/circuit_edge_features.npy")
graph_features = np.load("extracted_features/circuit_graph_features.npy")

# Use in ML pipeline
print(f"Node features: {node_features.shape}")
print(f"Edge features: {edge_features.shape}")
print(f"Graph features: {graph_features.shape}")
```

#### Batch Processing Multiple Circuits

```python
import os
from pathlib import Path

# Process all AAG files in a directory
aag_files = list(Path("benchmark/").glob("*.aag"))

for aag_file in aag_files:
    print(f"Processing {aag_file.name}...")
    
    saved_files = extract_features_to_files(
        aag_file_path=str(aag_file),
        output_dir=f"extracted_features/{aag_file.stem}",
        formats=['csv', 'npy']
    )
    
    print(f"✓ Saved {len(saved_files)} feature types")

# Load all extracted features
all_features = {}
for aag_file in aag_files:
    base_path = f"extracted_features/{aag_file.stem}/{aag_file.stem}"
    features = load_features_from_files(base_path, formats=['npy'])
    all_features[aag_file.stem] = features

print(f"Loaded features for {len(all_features)} circuits")
```

#### Feature Analysis and Visualization

```python
import matplotlib.pyplot as plt
import numpy as np

# Load features for analysis
features = load_features_from_files("extracted_features/circuit", formats=['npy'])
node_features = features['node']

# Analyze node distribution by level
levels = node_features[:, 5]  # level feature
plt.figure(figsize=(10, 6))
plt.hist(levels, bins=range(int(max(levels)) + 2), alpha=0.7)
plt.xlabel('Topological Level')
plt.ylabel('Number of Nodes')
plt.title('Node Distribution by Topological Level')
plt.show()

# Analyze fanin/fanout distribution
fanin = node_features[:, 3]   # fanin feature
fanout = node_features[:, 4]  # fanout feature

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(fanin, alpha=0.7)
plt.xlabel('Fan-in')
plt.ylabel('Count')
plt.title('Fan-in Distribution')

plt.subplot(1, 2, 2)
plt.hist(fanout, alpha=0.7)
plt.xlabel('Fan-out')
plt.ylabel('Count')
plt.title('Fan-out Distribution')
plt.tight_layout()
plt.show()
```

#### Integration with Machine Learning Pipelines

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load features for multiple circuits
circuit_features = {}
for circuit_name in ['c17', 'c432', 'c499']:
    features = load_features_from_files(
        f"extracted_features/{circuit_name}/{circuit_name}", 
        formats=['npy']
    )
    circuit_features[circuit_name] = features

# Prepare dataset for circuit classification
X = []  # Graph features
y = []  # Circuit labels

for circuit_name, features in circuit_features.items():
    graph_feat = features['graph'].flatten()  # 6 global features
    X.append(graph_feat)
    y.append(circuit_name)

X = np.array(X)
y = np.array(y)

# Train a classifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

print(f"Accuracy: {clf.score(X_test, y_test):.3f}")
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

### Basic Library Testing

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

### Offline Feature Extraction Testing

Test the offline feature extraction functionality:

```bash
python aag2gnn/offline_example.py
```

This will:
1. Create a sample AAG file
2. Extract features to multiple formats (CSV, JSON, NumPy)
3. Save node, edge, graph, and mapping features
4. Load features back and display statistics
5. Demonstrate the complete offline workflow

### Expected Output

The offline example should create the following directory structure:

```
extracted_features/
├── sample_circuit_node_features.csv
├── sample_circuit_node_features.json
├── sample_circuit_node_features.npy
├── sample_circuit_edge_features.csv
├── sample_circuit_edge_features.json
├── sample_circuit_edge_features.npy
├── sample_circuit_graph_features.csv
├── sample_circuit_graph_features.json
├── sample_circuit_graph_features.npy
├── sample_circuit_node_mapping.csv
└── sample_circuit_node_mapping.json
```

### Feature Validation

You can validate the extracted features by checking:

```python
import numpy as np
import pandas as pd

# Check node features
node_df = pd.read_csv("extracted_features/sample_circuit_node_features.csv")
print("Node features shape:", node_df.shape)
print("Feature columns:", list(node_df.columns))

# Check graph features
graph_features = np.load("extracted_features/sample_circuit_graph_features.npy")
print("Graph features:", graph_features)
print("Max level feature:", graph_features[0, 5])  # 6th feature
```

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

## 📁 Output File Formats

The offline feature extraction supports multiple output formats, each optimized for different use cases:

### CSV Format (`.csv`)
- **Use case**: Human-readable analysis, Excel/Google Sheets, data exploration
- **Features**: Headers, tabular format, easy to filter and sort
- **Example**:
  ```csv
  is_input,is_output,is_and,fanin,fanout,level,node_id
  1.0,0.0,0.0,0.0,2.0,0.0,1
  0.0,0.0,1.0,2.0,1.0,1.0,3
  ```

### JSON Format (`.json`)
- **Use case**: Web applications, structured data exchange, metadata storage
- **Features**: Hierarchical structure, includes feature names and metadata
- **Example**:
  ```json
  {
    "features": [4.0, 6.0, 2.0, 2.0, 3.33, 2.0],
    "feature_names": ["num_nodes", "num_edges", "num_inputs", "num_outputs", "avg_fanin", "max_level"],
    "feature_values": {
      "num_nodes": 4.0,
      "max_level": 2.0
    }
  }
  ```

### NumPy Format (`.npy`)
- **Use case**: Machine learning pipelines, high-performance computing
- **Features**: Binary format, fast I/O, memory efficient
- **Example**:
  ```python
  # Load directly as numpy array
  features = np.load("circuit_features.npy")
  print(features.shape)  # (1, 6) for graph features
  ```

### Node Mapping Files
- **Purpose**: Maintains correspondence between AAG node IDs and feature indices
- **Formats**: CSV and JSON
- **Use case**: Debugging, result interpretation, custom analysis

## 🔄 Compatibility

- **Python**: ≥ 3.8
- **PyTorch**: ≥ 1.9.0
- **PyTorch Geometric**: ≥ 2.0.0
- **Pandas**: ≥ 1.3.0 (for CSV output)
- **NumPy**: ≥ 1.21.0
- **GNN Models**: Compatible with GCN, GAT, GIN, GraphSAGE, etc.

## 💡 Best Practices & Tips

### Feature Extraction Workflow

1. **For Large Datasets**: Use NumPy format for storage efficiency
   ```python
   # Efficient for large circuits
   extract_features_to_files("large_circuit.aag", "features", formats=['npy'])
   ```

2. **For Analysis**: Use CSV format for easy exploration
   ```python
   # Good for data analysis
   extract_features_to_files("circuit.aag", "analysis", formats=['csv'])
   ```

3. **For Web Applications**: Use JSON format for structured data
   ```python
   # Good for web APIs
   extract_features_to_files("circuit.aag", "api", formats=['json'])
   ```

### Performance Optimization

- **Batch Processing**: Process multiple circuits in a loop
- **Memory Management**: Use NumPy format for large feature matrices
- **Directory Organization**: Use descriptive directory structures
- **Error Handling**: Always check if files exist before loading

### Common Use Cases

- **Circuit Analysis**: Extract features for statistical analysis
- **Machine Learning**: Prepare datasets for GNN training
- **Visualization**: Create plots and charts from extracted features
- **Benchmarking**: Compare different circuits using global features
- **Debugging**: Use node mapping to trace feature indices back to circuit nodes

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