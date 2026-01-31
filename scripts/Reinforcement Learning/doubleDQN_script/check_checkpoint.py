"""Check checkpoint information"""
import torch
from pathlib import Path

checkpoint_path = Path(r"C:\isaacsim\cobotproject\scripts\Reinforcement Learning\doubleDQN_script\models\ddqn_rrt_isaacsim_grid4_cubes9_20260112_035337_step_20000.pt")

print(f"Loading checkpoint: {checkpoint_path.name}")
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

print("\nCheckpoint keys:", list(checkpoint.keys()))
print("\n" + "="*60)
print("CHECKPOINT INFORMATION")
print("="*60)
print(f"Steps: {checkpoint.get('steps', 'N/A')}")
print(f"Episodes: {checkpoint.get('episodes', 'N/A')}")
print(f"Epsilon: {checkpoint.get('epsilon', 'N/A')}")
print(f"State dim: {checkpoint.get('state_dim', 'N/A')}")
print(f"Action dim: {checkpoint.get('action_dim', 'N/A')}")
print(f"Gamma: {checkpoint.get('gamma', 'N/A')}")
print(f"Batch size: {checkpoint.get('batch_size', 'N/A')}")
print(f"Learning rate: {checkpoint.get('learning_rate', 'N/A')}")
print(f"Epsilon start: {checkpoint.get('epsilon_start', 'N/A')}")
print(f"Epsilon end: {checkpoint.get('epsilon_end', 'N/A')}")
print(f"Epsilon decay: {checkpoint.get('epsilon_decay', 'N/A')}")
print(f"Epsilon decay type: {checkpoint.get('epsilon_decay_type', 'N/A')}")
print(f"Epsilon decay rate: {checkpoint.get('epsilon_decay_rate', 'N/A')}")
print(f"Target update freq: {checkpoint.get('target_update_freq', 'N/A')}")
print(f"Target update tau: {checkpoint.get('target_update_tau', 'N/A')}")
print(f"Warmup steps: {checkpoint.get('warmup_steps', 'N/A')}")
print("="*60)

