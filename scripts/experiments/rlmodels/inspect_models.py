"""
Inspect pretrained model structures to understand their architectures.
"""

import torch
from pathlib import Path

def inspect_model(model_path):
    """Inspect a PyTorch model file."""
    print(f"\n{'='*80}")
    print(f"Model: {model_path.name}")
    print(f"{'='*80}")
    
    try:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        print(f"Type: {type(checkpoint)}")
        
        if isinstance(checkpoint, dict):
            print(f"\nKeys in checkpoint:")
            for key in checkpoint.keys():
                print(f"  - {key}")
            
            # Check if it's a state_dict
            if all(isinstance(k, str) and '.' in k for k in list(checkpoint.keys())[:5]):
                print(f"\nThis appears to be a state_dict with {len(checkpoint)} parameters:")
                for key, value in list(checkpoint.items())[:10]:
                    print(f"  {key}: {value.shape}")
                if len(checkpoint) > 10:
                    print(f"  ... and {len(checkpoint) - 10} more parameters")
                
                # Try to infer architecture
                first_key = list(checkpoint.keys())[0]
                if 'weight' in first_key:
                    first_weight = checkpoint[first_key]
                    print(f"\nFirst layer input dim: {first_weight.shape[1]}")
                    print(f"First layer output dim: {first_weight.shape[0]}")
            
            # Check for specific keys
            if 'policy_net_state_dict' in checkpoint:
                print("\n✅ This is a DoubleDQN checkpoint")
                policy_net = checkpoint['policy_net_state_dict']
                print(f"   Policy network has {len(policy_net)} parameters")
            
            if 'state_dim' in checkpoint:
                print(f"\n✅ State dim: {checkpoint['state_dim']}")
            if 'action_dim' in checkpoint:
                print(f"✅ Action dim: {checkpoint['action_dim']}")
        
        else:
            print(f"\nNot a dictionary - this is a {type(checkpoint)}")
    
    except Exception as e:
        print(f"❌ ERROR: {e}")

def main():
    base_dir = Path(__file__).parent / "models" / "pretrained"
    
    print("="*80)
    print("PRETRAINED MODEL INSPECTION")
    print("="*80)
    
    models = [
        "duel_ddqn_lunarlander.pth",
        "per_ddqn_light_lunarlander.pth",
        "per_ddqn_full_lunarlander.pth",
        "c51_ddqn_lunarlander.pth",
        "sac_discrete_lunarlander_actor.pth",
        "ppo_discrete_lunarlander_actor.pth",
        "custom_ddqn_isaacsim.pt",
    ]
    
    for model_name in models:
        model_path = base_dir / model_name
        if model_path.exists():
            inspect_model(model_path)
        else:
            print(f"\n❌ Model not found: {model_path}")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("""
The pretrained models (Duel-DDQN, PER-DDQN, C51-DDQN, SAC, PPO) were trained on
LunarLander-v2 with:
  - State dimension: 8
  - Action dimension: 4

Your Isaac Sim RRT task requires:
  - State dimension: 54 (9 objects × 6 features)
  - Action dimension: 9 (9 objects to pick)

❌ These models CANNOT be directly used without retraining!

✅ Only DDQN was trained specifically for Isaac Sim and will work.

To use the other models, you need to:
1. Train them from scratch on Isaac Sim (recommended)
2. Use transfer learning with adapter layers (complex)
3. Only test DDQN (current best option)
    """)

if __name__ == "__main__":
    main()

