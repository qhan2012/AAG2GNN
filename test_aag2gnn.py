"""
Basic test suite for the AAG-to-GNN library.
"""

import unittest
import torch
import tempfile
import os
from aag2gnn import load_aag_as_gnn_graph, parse_aag


class TestAAG2GNN(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_aag_content = """aag 3 2 0 1 1
2
4
6
6 2 4
c
Test circuit
"""
        
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.aag', delete=False)
        self.temp_file.write(self.test_aag_content)
        self.temp_file.close()
        self.aag_file_path = self.temp_file.name
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.aag_file_path):
            os.unlink(self.aag_file_path)
    
    def test_parse_aag(self):
        """Test AAG parsing."""
        parsed = parse_aag(self.aag_file_path)
        
        self.assertEqual(parsed['max_var'], 3)
        self.assertEqual(parsed['inputs'], [2, 4])
        self.assertEqual(parsed['outputs'], [6])
        self.assertEqual(parsed['and_gates'], [(6, 2, 4)])
    
    def test_load_aag_as_gnn_graph(self):
        """Test complete pipeline."""
        data = load_aag_as_gnn_graph(self.aag_file_path)
        
        # Check basic structure
        self.assertIsInstance(data.x, torch.Tensor)
        self.assertIsInstance(data.edge_index, torch.Tensor)
        self.assertIsInstance(data.edge_attr, torch.Tensor)
        self.assertIsInstance(data.global_x, torch.Tensor)
        
        # Check shapes
        self.assertEqual(data.x.size(0), 3)  # 3 nodes
        self.assertEqual(data.x.size(1), 6)  # 6 node features
        self.assertEqual(data.edge_index.size(1), 2)  # 2 edges
        self.assertEqual(data.edge_attr.size(1), 2)  # 2 edge features
        self.assertEqual(data.global_x.size(0), 5)  # 5 global features


if __name__ == '__main__':
    unittest.main(verbosity=2) 