"""
Quick test script for testing a single model
Usage: python test_single_model.py --model "PPO-Discrete" --seed 42 --episodes 1
"""
import argparse
import sys

# Parse arguments BEFORE importing anything else
parser = argparse.ArgumentParser(description="Test Single Model")
parser.add_argument("--model", type=str, required=True, help="Model name to test (e.g., 'PPO-Discrete', 'DDQN', 'SAC-Discrete')")
parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
parser.add_argument("--episodes", type=int, default=1, help="Number of episodes (default: 1)")
parser.add_argument("--grid_size", type=int, default=4, help="Grid size (default: 4)")
parser.add_argument("--num_cubes", type=int, default=9, help="Number of cubes (default: 9)")
args = parser.parse_args()

# Now run the main test script with filtered models
import os
os.environ["MODEL_FILTER"] = args.model
os.environ["SINGLE_MODEL_TEST"] = "1"

# Import and run main test
sys.argv = [
    "test_two_agent_system.py",
    "--action_space", "discrete" if "Discrete" in args.model or args.model == "DDQN" or args.model == "Heuristic" else "continuous",
    "--seeds", str(args.seed),
    "--episodes", str(args.episodes),
    "--grid_size", str(args.grid_size),
    "--num_cubes", str(args.num_cubes)
]

# Import main script
import test_two_agent_system
test_two_agent_system.main()

