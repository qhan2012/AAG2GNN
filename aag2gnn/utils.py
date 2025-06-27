"""
Utility functions for AAG-to-GNN conversion.

Includes topological sorting and other helper functions.
"""

from typing import Dict, List, Set, Tuple
from collections import defaultdict, deque


def literal_to_node(literal: int) -> int:
    """
    Convert a literal to a node index.
    
    Args:
        literal (int): AAG literal (even = non-inverted, odd = inverted)
        
    Returns:
        int: Node index (literal // 2)
    """
    return literal // 2


def node_to_literal(node: int, inverted: bool = False) -> int:
    """
    Convert a node index to a literal.
    
    Args:
        node (int): Node index
        inverted (bool): Whether the literal should be inverted
        
    Returns:
        int: AAG literal
    """
    return 2 * node + (1 if inverted else 0)


def is_inverted(literal: int) -> bool:
    """
    Check if a literal is inverted.
    
    Args:
        literal (int): AAG literal
        
    Returns:
        bool: True if literal is odd (inverted)
    """
    return literal % 2 == 1


def topological_sort(parsed: Dict) -> Dict[int, int]:
    """
    Compute topological levels for all nodes in the AAG.
    
    Args:
        parsed (dict): Parsed AAG data
        
    Returns:
        dict: Mapping from node index to topological level
    """
    # Build adjacency lists
    in_edges = defaultdict(list)  # node -> list of incoming nodes
    out_edges = defaultdict(list)  # node -> list of outgoing nodes
    
    # Add edges from AND gates
    for lhs, rhs0, rhs1 in parsed["and_gates"]:
        lhs_node = literal_to_node(lhs)
        rhs0_node = literal_to_node(rhs0)
        rhs1_node = literal_to_node(rhs1)
        
        in_edges[lhs_node].extend([rhs0_node, rhs1_node])
        out_edges[rhs0_node].append(lhs_node)
        out_edges[rhs1_node].append(lhs_node)
    
    # Initialize levels
    levels = {}
    in_degree = defaultdict(int)
    
    # Calculate in-degrees
    for node in in_edges:
        in_degree[node] = len(in_edges[node])
    
    # Add input nodes (level 0)
    queue = deque()
    for literal in parsed["inputs"]:
        node = literal_to_node(literal)
        levels[node] = 0
        queue.append(node)
    
    # Process nodes in topological order
    while queue:
        node = queue.popleft()
        current_level = levels[node]
        
        # Process outgoing edges
        for neighbor in out_edges[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                levels[neighbor] = current_level + 1
                queue.append(neighbor)
    
    return levels


def get_node_types(parsed: Dict) -> Dict[int, str]:
    """
    Get the type of each node (input, output, and_gate).
    
    Args:
        parsed (dict): Parsed AAG data
        
    Returns:
        dict: Mapping from node index to node type
    """
    node_types = {}
    
    # Mark inputs
    for literal in parsed["inputs"]:
        node = literal_to_node(literal)
        node_types[node] = "input"
    
    # Mark outputs
    for literal in parsed["outputs"]:
        node = literal_to_node(literal)
        node_types[node] = "output"
    
    # Mark AND gates
    for lhs, _, _ in parsed["and_gates"]:
        node = literal_to_node(lhs)
        node_types[node] = "and_gate"
    
    return node_types


def compute_fanin_fanout(parsed: Dict) -> Tuple[Dict[int, int], Dict[int, int]]:
    """
    Compute fanin and fanout for each node.
    
    Args:
        parsed (dict): Parsed AAG data
        
    Returns:
        tuple: (fanin_dict, fanout_dict) mapping node indices to counts
    """
    fanin = defaultdict(int)
    fanout = defaultdict(int)
    
    # Count edges from AND gates
    for lhs, rhs0, rhs1 in parsed["and_gates"]:
        lhs_node = literal_to_node(lhs)
        rhs0_node = literal_to_node(rhs0)
        rhs1_node = literal_to_node(rhs1)
        
        fanin[lhs_node] += 2  # Each AND gate has 2 inputs
        fanout[rhs0_node] += 1
        fanout[rhs1_node] += 1
    
    return dict(fanin), dict(fanout) 