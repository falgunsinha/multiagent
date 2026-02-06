import torch
import numpy as np
from torch_geometric.data import Data


def build_graph(obs, robot_positions, object_positions, obstacles=None, targets=None, 
                edge_threshold=2.0, device='cpu'):
    """
    Build graph from environment state.
    """
    nodes = []
    node_types = []  # 0: robot, 1: object, 2: obstacle, 3: target
    for i, pos in enumerate(robot_positions):
        # Node features: [x, y, z, is_robot, is_obstacle, is_target, object_id]
        node_feat = [pos[0], pos[1], pos[2], 1, 0, 0, -1]
        nodes.append(node_feat)
        node_types.append(0)
    for i, pos in enumerate(object_positions):
        node_feat = [pos[0], pos[1], pos[2], 0, 0, 0, i]
        nodes.append(node_feat)
        node_types.append(1)
  
    if obstacles is not None:
        for pos in obstacles:
            node_feat = [pos[0], pos[1], pos[2], 0, 1, 0, -1]
            nodes.append(node_feat)
            node_types.append(2)
    if targets is not None:
        for i, pos in enumerate(targets):
            node_feat = [pos[0], pos[1], pos[2], 0, 0, 1, i]
            nodes.append(node_feat)
            node_types.append(3)
  
    x = torch.tensor(nodes, dtype=torch.float32, device=device)
    num_nodes = x.size(0)
    edge_index = []
    edge_attr = []
    
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:
                continue
            pos_i = x[i, :3]
            pos_j = x[j, :3]
            distance = torch.norm(pos_i - pos_j).item()
            if distance <= edge_threshold:
                edge_index.append([i, j])
                edge_feat = [distance, 1.0, 0.0]  # Default: reachable, not blocking
                edge_attr.append(edge_feat)
    
    if len(edge_index) > 0:
        edge_index = torch.tensor(edge_index, dtype=torch.long, device=device).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float32, device=device)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        edge_attr = torch.empty((0, 3), dtype=torch.float32, device=device)
    
    graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    return graph


def compute_edge_features(graph, rrt_estimator=None):
    """
    Compute advanced edge features (reachability, blocking).
    
    """
    if graph.edge_attr is None or graph.edge_attr.size(0) == 0:
        return graph
    
    num_edges = graph.edge_attr.size(0)
    edge_index = graph.edge_index
    x = graph.x
    if rrt_estimator is not None:
        for e in range(num_edges):
            src, dst = edge_index[0, e].item(), edge_index[1, e].item()
            pos_src = x[src, :3].cpu().numpy()
            pos_dst = x[dst, :3].cpu().numpy()
            try:
                reachable = rrt_estimator.is_reachable(pos_src, pos_dst)
                graph.edge_attr[e, 1] = 1.0 if reachable else 0.0
            except:
                # If RRT fails, assume reachable
                graph.edge_attr[e, 1] = 1.0
    
    for e in range(num_edges):
        src, dst = edge_index[0, e].item(), edge_index[1, e].item()
        is_robot_src = x[src, 3].item() == 1
        is_object_dst = x[dst, 6].item() >= 0  # object_id >= 0
        
        if is_robot_src and is_object_dst:
            blocking_score = compute_blocking_score(x, src, dst, edge_index)
            graph.edge_attr[e, 2] = blocking_score
    
    return graph


def compute_blocking_score(x, robot_idx, object_idx, edge_index):
    """
    Compute how much other objects block the path from robot to object.
    """
    robot_pos = x[robot_idx, :3]
    object_pos = x[object_idx, :3]
    
    blocking_count = 0
    total_objects = 0
    for i in range(x.size(0)):
        if i == robot_idx or i == object_idx:
            continue
        if x[i, 6].item() >= 0:  # object_id >= 0
            total_objects += 1
            other_pos = x[i, :3]
            if is_between(robot_pos, object_pos, other_pos, threshold=0.5):
                blocking_count += 1
    
    if total_objects == 0:
        return 0.0
    
    return blocking_count / total_objects


def is_between(p1, p2, p_test, threshold=0.5):
    """
    Check if p_test is between p1 and p2.
    """
    d12 = torch.norm(p2 - p1)
    d1t = torch.norm(p_test - p1)
    d2t = torch.norm(p_test - p2)
    
    return (d1t + d2t - d12).abs() < threshold

