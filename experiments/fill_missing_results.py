"""
Find Missing Evaluation Results and Generate Them

This script scans data directories for experiments with checkpoints but missing
evaluation_results.json files, then evaluates the trained models to fill in the gaps.

Usage:
    python experiments/fill_missing_results.py --data_dir ./data/all
    python experiments/fill_missing_results.py --data_dir ./data/correlated_demand --dry_run
"""

import argparse
import json
import numpy as np
from pathlib import Path
from glob import glob
import sys
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from Src.parser import Parser
from Src.config import Config
from time import time


def find_missing_results(data_dir):
    """
    Find all experiment directories with checkpoints but missing evaluation_results.json.
    
    Returns:
        List of dicts with: {
            'path': Path to seed directory,
            'has_checkpoint': bool,
            'has_results': bool,
            'config_info': parsed info from directory name
        }
    """
    missing = []
    
    # Find all seed directories
    seed_dirs = glob(f"{data_dir}/*/seed*/")
    
    for seed_dir in seed_dirs:
        seed_path = Path(seed_dir)
        results_file = seed_path / "Results" / "evaluation_results.json"
        
        # Check for checkpoints (actor.pt or critic.pt in Checkpoints or direct files)
        checkpoint_patterns = [
            seed_path / "Checkpoints" / "actor.pt",
            seed_path / "Checkpoints" / "critic.pt",
            seed_path / "Checkpointsactor.pt",  # Sometimes concatenated
            seed_path / "Checkpointscritic.pt",
        ]
        
        has_checkpoint = any(cp.exists() for cp in checkpoint_patterns)
        has_results = results_file.exists()
        
        if has_checkpoint and not has_results:
            # Parse config from directory name
            config_info = parse_directory_name(seed_path)
            
            missing.append({
                'path': str(seed_path),
                'has_checkpoint': has_checkpoint,
                'has_results': has_results,
                'config_info': config_info,
                'seed': int(seed_path.name.replace('seed', ''))
            })
    
    return missing


def parse_directory_name(seed_path):
    """
    Parse experiment configuration from directory name.
    Example: dnc_O75_N20_ScaleTrue_Diststandard_Corr0.5_NPSA_Cool0.1/seed42
    """
    parent = seed_path.parent.name
    parts = parent.split('_')
    
    config = {}
    
    # Parse algorithm
    config['algorithm'] = parts[0] if parts else 'unknown'
    
    # Parse other parameters
    for part in parts:
        if part.startswith('O'):
            config['O'] = int(part[1:])
        elif part.startswith('N') and part[1:].isdigit():
            config['N'] = int(part[1:])
        elif part.startswith('Scale'):
            config['scale'] = part.replace('Scale', '') == 'True'
        elif part.startswith('Dist'):
            config['dist'] = part.replace('Dist', '')
        elif part.startswith('Corr'):
            config['corr'] = float(part.replace('Corr', ''))
        elif part.startswith('NP'):
            config['neighbor_picking'] = part.replace('NP', '')
        elif part.startswith('Cool'):
            config['cooling'] = float(part.replace('Cool', ''))
    
    return config


def evaluate_from_checkpoint(seed_path, config_info, n_eval_episodes=100):
    """
    Load checkpoint and evaluate the model to generate evaluation_results.json
    
    Args:
        seed_path: Path to seed directory
        config_info: Parsed configuration dict
        n_eval_episodes: Number of evaluation episodes
    
    Returns:
        dict: Evaluation results
    """
    print(f"\nEvaluating: {seed_path}")
    print(f"Config: {config_info}")
    
    # Create parser and override arguments
    parser = Parser()
    args = parser.get_parser().parse_args(args=[])
    
    # Override with config from directory name
    args.env_name = 'JointReplenishment_py'
    args.n_actions = config_info.get('N', 2)
    args.commonOrderCosts = config_info.get('O', 75)
    args.seed = config_info.get('seed', 42)
    args.scale_reward = config_info.get('scale', False)
    args.demand_dist = config_info.get('dist', 'standard')
    args.demand_correlation = config_info.get('corr', 0.0)
    args.mapping = 'dnc_mapping'
    args.max_steps = 100
    
    # Algorithm-specific settings
    algo = config_info.get('algorithm', 'dnc')
    if algo == 'minmax':
        args.SA_search_steps = 0
        args.cooling = 0
        args.neighbor_picking = 'SA'
    elif algo == 'cma_es':
        args.neighbor_picking = 'cma_es'
    elif algo == 'dnc':
        args.neighbor_picking = config_info.get('neighbor_picking', 'SA')
        args.cooling = config_info.get('cooling', 0.1)
    
    # Create config
    config = Config(args)
    
    # Create model
    model = config.algo(config=config)
    env = config.env
    
    # Load checkpoint
    seed_path_obj = Path(seed_path)
    
    # Try different checkpoint locations
    checkpoint_dirs = [
        seed_path_obj / "Checkpoints",
        seed_path_obj / "Checkpointsactor.pt"  # Check if it's a concatenated path
    ]
    
    actor_checkpoint = None
    for checkpoint_dir in checkpoint_dirs:
        if (checkpoint_dir / "actor.pt").exists():
            actor_checkpoint = checkpoint_dir / "actor.pt"
            break
        elif str(checkpoint_dir).endswith("actor.pt") and checkpoint_dir.exists():
            # Handle concatenated path
            actor_checkpoint = checkpoint_dir
            break
    
    if actor_checkpoint is None:
        # Try alternative: Checkpointsactor.pt directly
        alt_path = seed_path_obj / "Checkpointsactor.pt"
        if alt_path.exists():
            actor_checkpoint = alt_path
    
    if actor_checkpoint is None:
        print(f"  Error: No actor checkpoint found in {seed_path}")
        return None
    
    try:
        # Load the actor model
        model.actor.load_state_dict(torch.load(actor_checkpoint))
        model.actor.eval()
        print(f"  Loaded checkpoint from: {actor_checkpoint}")
    except Exception as e:
        print(f"  Error loading checkpoint: {e}")
        return None
    
    # Evaluate model
    print(f"  Running {n_eval_episodes} evaluation episodes...")
    rewards = []
    infos_collection = {}
    
    for episode in range(n_eval_episodes):
        state = np.float32(env.reset(training=False))
        total_r = 0
        step = 0
        done = False
        
        while not done:
            if episode == 0 and step == 0:
                model.weights_changed = True
            else:
                model.weights_changed = False
            
            action, _ = model.get_action(state, training=False)
            new_state, reward, done, info = env.step(action, training=False)
            
            # Collect info
            if isinstance(info, dict):
                for k, v in info.items():
                    if k not in infos_collection:
                        infos_collection[k] = []
                    infos_collection[k].append(v)
            
            state = new_state
            total_r += reward
            step += 1
            
            if step > 100:
                break
        
        rewards.append(total_r)
    
    # Calculate statistics
    mean_cost = float(np.mean(rewards))
    std_cost = float(np.std(rewards))
    
    # Process stockouts
    stockouts_raw = infos_collection.get('stockouts', [])
    stockouts = []
    for s in stockouts_raw:
        arr = np.asarray(s, dtype=float)
        stockouts.append(float(arr) if arr.size == 1 else arr.tolist())
    
    # Create results dict
    results = {
        'algorithm': algo,
        'O': config_info.get('O', 75),
        'seed': config_info.get('seed', 42),
        'n_items': config_info.get('N', 2),
        'mean_cost': mean_cost,
        'std_cost': std_cost,
        'costs': [float(r) for r in rewards],
        'stockouts': stockouts,
        'mean_stockout_rate': float(np.mean(infos_collection.get('mean_stockout_rate', [0]))),
        'generated_from_checkpoint': True
    }
    
    print(f"  Mean Cost: {mean_cost:.2f} ± {std_cost:.2f}")
    
    return results


def save_results(seed_path, results):
    """Save evaluation results to JSON file"""
    results_dir = Path(seed_path) / "Results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = results_dir / "evaluation_results.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"  Saved results to: {results_file}")


def main():
    parser = argparse.ArgumentParser(description='Find and fill missing evaluation results')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Data directory to scan (e.g., ./data/all)')
    parser.add_argument('--n_eval_episodes', type=int, default=100,
                       help='Number of evaluation episodes')
    parser.add_argument('--dry_run', action='store_true',
                       help='Only show what would be evaluated, don\'t actually run')
    
    args = parser.parse_args()
    
    # Find missing results
    print(f"Scanning {args.data_dir} for missing results...")
    missing = find_missing_results(args.data_dir)
    
    print(f"\nFound {len(missing)} experiments with checkpoints but missing evaluation results:")
    print("=" * 80)
    
    for item in missing:
        config = item['config_info']
        print(f"\n{item['path']}")
        print(f"  Algorithm: {config.get('algorithm', 'unknown')}")
        print(f"  O={config.get('O', '?')}, N={config.get('N', '?')}, Seed={item['seed']}")
        if 'neighbor_picking' in config:
            print(f"  Neighbor Picking: {config['neighbor_picking']}")
        if 'cooling' in config:
            print(f"  Cooling: {config['cooling']}")
    
    if args.dry_run:
        print(f"\n{'='*80}")
        print("DRY RUN: No evaluation performed")
        return
    
    # Ask for confirmation
    if missing:
        print(f"\n{'='*80}")
        response = input(f"Evaluate {len(missing)} experiments? (y/n): ")
        if response.lower() != 'y':
            print("Cancelled")
            return
    
    # Evaluate and save
    print(f"\n{'='*80}")
    print("Starting evaluation...")
    
    success_count = 0
    fail_count = 0
    
    for i, item in enumerate(missing, 1):
        print(f"\n[{i}/{len(missing)}]")
        
        # Add seed to config_info
        item['config_info']['seed'] = item['seed']
        
        results = evaluate_from_checkpoint(
            item['path'],
            item['config_info'],
            args.n_eval_episodes
        )
        
        if results:
            save_results(item['path'], results)
            success_count += 1
        else:
            fail_count += 1
    
    print(f"\n{'='*80}")
    print(f"Summary: {success_count} successful, {fail_count} failed")


if __name__ == "__main__":
    main()
