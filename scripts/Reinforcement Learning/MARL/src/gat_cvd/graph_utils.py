"""
Graph Construction Utilities

Converts environment state into graph representation for GAT encoder.
Nodes: robots, objects, obstacles, targets
Edges: proximity, reachability, blocking relationships
"""

import torch
import numpy as np
from torch_geometric.data import Data


def build_graph(obs, robot_positions, object_positions, obstacles=None, targets=None, 
                edge_threshold=2.0, device='cpu'):
    """
    Build graph from environment state.
    
    Args:
        obs: Observation array (can be flattened or structured)
        robot_positions: List of robot positions [(x1, y1, z1), (x2, y2, z2)]
        object_positions: List of object positions [(x, y, z), ...]
        obstacles: List of obstacle positions (optional)
        targets: List of target positions (optional)
        edge_threshold: Distance threshold for creating edges
        device: torch device
    
    Returns:
        graph: PyTorch Geometric Data object with nodes, edges, edge_attr
    """
    nodes = []
    node_types = []  # 0: robot, 1: object, 2: obstacle, 3: target
    
    # Add robot nodes
    for i, pos in enumerate(robot_positions):
        # Node features: [x, y, z, is_robot, is_obstacle, is_target, object_id]
        node_feat = [pos[0], pos[1], pos[2], 1, 0, 0, -1]
        nodes.append(node_feat)
        node_types.append(0)
    
    # Add object nodes
    for i, pos in enumerate(object_positions):
        node_feat = [pos[0], pos[1], pos[2], 0, 0, 0, i]
        nodes.append(node_feat)
        node_types.append(1)
    
    # Add obstacle nodes (if provided)
    if obstacles is not None:
        for pos in obstacles:
            node_feat = [pos[0], pos[1], pos[2], 0, 1, 0, -1]
            nodes.append(node_feat)
            node_types.append(2)
    
    # Add target nodes (if provided)
    if targets is not None:
        for i, pos in enumerate(targets):
            node_feat = [pos[0], pos[1], pos[2], 0, 0, 1, i]
            nodes.append(node_feat)
            node_types.append(3)
    
    # Convert to tensor
    x = torch.tensor(nodes, dtype=torch.float32, device=device)
    num_nodes = x.size(0)
    
    # Build edges based on proximity
    edge_index = []
    edge_attr = []
    
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:
                continue
            
            # Compute distance
            pos_i = x[i, :3]
            pos_j = x[j, :3]
            distance = torch.norm(pos_i - pos_j).item()
            
            # Add edge if within threshold
            if distance <= edge_threshold:
                edge_index.append([i, j])
                
                # Edge features: [distance, reachability, blocking_score]
                # Reachability and blocking will be computed separately
                edge_feat = [distance, 1.0, 0.0]  # Default: reachable, not blocking
                edge_attr.append(edge_feat)
    
    # Convert to tensors
    if len(edge_index) > 0:
        edge_index = torch.tensor(edge_index, dtype=torch.long, device=device).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float32, device=device)
    else:
        # No edges (isolated nodes)
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        edge_attr = torch.empty((0, 3), dtype=torch.float32, device=device)
    
    # Create PyG Data object
    graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    return graph


def compute_edge_features(graph, rrt_estimator=None):
    """
    Compute advanced edge features (reachability, blocking).
    
    Args:
        graph: PyG Data object
        rrt_estimator: RRT-based reachability estimator (optional)
    
    Returns:
        graph: Updated graph with edge features
    """
    if graph.edge_attr is None or graph.edge_attr.size(0) == 0:
        return graph
    
    num_edges = graph.edge_attr.size(0)
    edge_index = graph.edge_index
    x = graph.x
    
    # Update reachability using RRT (if available)
    if rrt_estimator is not None:
        for e in range(num_edges):
            src, dst = edge_index[0, e].item(), edge_index[1, e].item()
            pos_src = x[src, :3].cpu().numpy()
            pos_dst = x[dst, :3].cpu().numpy()
            
            # Check reachability using RRT
            try:
                reachable = rrt_estimator.is_reachable(pos_src, pos_dst)
                graph.edge_attr[e, 1] = 1.0 if reachable else 0.0
            except:
                # If RRT fails, assume reachable
                graph.edge_attr[e, 1] = 1.0
    
    # Compute blocking scores
    # An object blocks another if it's between the robot and the target object
    for e in range(num_edges):
        src, dst = edge_index[0, e].item(), edge_index[1, e].item()
        
        # Check if src is robot and dst is object
        is_robot_src = x[src, 3].item() == 1
        is_object_dst = x[dst, 6].item() >= 0  # object_id >= 0
        
        if is_robot_src and is_object_dst:
            # Check if any other object is blocking
            blocking_score = compute_blocking_score(x, src, dst, edge_index)
            graph.edge_attr[e, 2] = blocking_score
    
    return graph


def compute_blocking_score(x, robot_idx, object_idx, edge_index):
    """
    Compute how much other objects block the path from robot to object.
    
    Args:
        x: Node features
        robot_idx: Robot node index
        object_idx: Object node index
        edge_index: Edge indices
    
    Returns:
        blocking_score: Score in [0, 1] indicating blocking severity
    """
    robot_pos = x[robot_idx, :3]
    object_pos = x[object_idx, :3]
    
    blocking_count = 0
    total_objects = 0
    
    # Check all other objects
    for i in range(x.size(0)):
        if i == robot_idx or i == object_idx:
            continue
        
        # Check if it's an object
        if x[i, 6].item() >= 0:  # object_id >= 0
            total_objects += 1
            other_pos = x[i, :3]
            
            # Check if other object is between robot and target object
            if is_between(robot_pos, object_pos, other_pos, threshold=0.5):
                blocking_count += 1
    
    if total_objects == 0:
        return 0.0
    
    return blocking_count / total_objects


def is_between(p1, p2, p_test, threshold=0.5):
    """
    Check if p_test is between p1 and p2.
    
    Args:
        p1: Start point
        p2: End point
        p_test: Test point
        threshold: Distance threshold
    
    Returns:
        True if p_test is between p1 and p2
    """
    # Compute distances
    d12 = torch.norm(p2 - p1)
    d1t = torch.norm(p_test - p1)
    d2t = torch.norm(p_test - p2)
    
    # Check if p_test is on the line segment
    return (d1t + d2t - d12).abs() < threshold

