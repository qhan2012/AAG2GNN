"""
AAG (And-Inverter Graph) Parser

Parses .aag files and extracts the graph structure including inputs, outputs,
and AND gates.
"""

from typing import Dict, List, Tuple, Optional
import re


def parse_aag(file_path: str) -> Dict:
    """
    Parse an AAG file and extract the graph structure.
    
    Args:
        file_path (str): Path to the .aag file
        
    Returns:
        dict: Parsed AAG data with keys:
            - max_var: Maximum variable index
            - inputs: List of input variable indices
            - outputs: List of output variable indices  
            - and_gates: List of (lhs, rhs0, rhs1) tuples
            - comments: Optional list of comment lines
            
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file format is invalid
    """
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        raise FileNotFoundError(f"AAG file not found: {file_path}")
    
    if not lines:
        raise ValueError("Empty AAG file")
    
    # Parse header line: aag max_var num_inputs num_latches num_outputs num_and_gates
    header = lines[0].strip()
    if not header.startswith('aag'):
        raise ValueError("Invalid AAG file: must start with 'aag'")
    
    header_parts = header.split()
    if len(header_parts) != 6:
        raise ValueError("Invalid AAG header format")
    
    try:
        max_var = int(header_parts[1])
        num_inputs = int(header_parts[2])
        num_latches = int(header_parts[3])
        num_outputs = int(header_parts[4])
        num_and_gates = int(header_parts[5])
    except ValueError:
        raise ValueError("Invalid numeric values in AAG header")
    
    # Parse inputs
    inputs = []
    line_idx = 1
    for i in range(num_inputs):
        if line_idx >= len(lines):
            raise ValueError("Unexpected end of file while parsing inputs")
        try:
            inputs.append(int(lines[line_idx].strip()))
            line_idx += 1
        except ValueError:
            raise ValueError(f"Invalid input value at line {line_idx + 1}")
    
    # Parse latches (skip for now, but maintain line count)
    for i in range(num_latches):
        if line_idx >= len(lines):
            raise ValueError("Unexpected end of file while parsing latches")
        line_idx += 1
    
    # Parse outputs
    outputs = []
    for i in range(num_outputs):
        if line_idx >= len(lines):
            raise ValueError("Unexpected end of file while parsing outputs")
        try:
            outputs.append(int(lines[line_idx].strip()))
            line_idx += 1
        except ValueError:
            raise ValueError(f"Invalid output value at line {line_idx + 1}")
    
    # Parse AND gates
    and_gates = []
    for i in range(num_and_gates):
        if line_idx >= len(lines):
            raise ValueError("Unexpected end of file while parsing AND gates")
        try:
            gate_parts = lines[line_idx].strip().split()
            if len(gate_parts) != 3:
                raise ValueError(f"Invalid AND gate format at line {line_idx + 1}")
            lhs = int(gate_parts[0])
            rhs0 = int(gate_parts[1])
            rhs1 = int(gate_parts[2])
            and_gates.append((lhs, rhs0, rhs1))
            line_idx += 1
        except ValueError:
            raise ValueError(f"Invalid AND gate values at line {line_idx + 1}")
    
    # Parse comments (optional)
    comments = []
    while line_idx < len(lines):
        line = lines[line_idx].strip()
        if line.startswith('c'):
            comments.append(line)
        elif line:  # Non-empty non-comment line
            raise ValueError(f"Unexpected content at line {line_idx + 1}: {line}")
        line_idx += 1
    
    return {
        "max_var": max_var,
        "inputs": inputs,
        "outputs": outputs,
        "and_gates": and_gates,
        "comments": comments if comments else None
    } 