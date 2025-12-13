"""
Single Configuration Training Script
Can be called from command line with algorithm and O (common order cost) parameters
"""
import sys
import os
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from run import Solver
from Src.parser import Parser
from Src.config import Config
from experiments.baseline_solver import solve_independent_sS, evaluate_baseline_policy
import numpy as np
import json


def train_dnc_or_minmax(O, algorithm='dnc', seed=42, n_actions=2, max_episodes=5000, 
                        output_dir='$SCRATCH/ift6162-project', neighbor_picking='SA', demand_dist='standard'):
    """
    Train DNC or MinMax agent
    
    Args:
        O: Common order cost (K)
        algorithm: 'dnc' or 'minmax'
        seed: Random seed
        n_actions: Number of items
        max_episodes: Training episodes
        output_dir: Directory to save results
        neighbor_picking: 'SA' or 'greedy'
        demand_dist: 'standard' or 'heterogeneous'
    """
    print(f"\n{'='*60}")
    print(f"Training {algorithm.upper()}: O={O}, seed={seed}, items={n_actions}")
    print(f"{'='*60}\n")
    
    # Create parser and override arguments
    parser = Parser()
    args = parser.get_parser().parse_args(args=[])
    
    # Override with experiment settings
    args.env_name = 'JointReplenishment_py'
    args.n_actions = n_actions
    args.commonOrderCosts = O
    args.seed = seed
    args.max_episodes = max_episodes
    args.mapping = 'dnc_mapping'  # Both DNC and MinMax use dnc_mapping
    args.experiment = f'{algorithm}_O{O}'
    args.folder_suffix = f'seed{seed}'
    args.save_count = max(1, max_episodes // 10)
    args.neighbor_picking = neighbor_picking
    args.demand_dist = demand_dist
    
    # Set MinMax parameters if needed
    if algorithm == 'minmax':
        args.SA_search_steps = 0  # This makes it MinMax
        args.cooling = 0
    
    # Create config
    config = Config(args)
    
    # Override paths to use scratch directory
    output_path = Path(output_dir) / f'{algorithm}_O{O}' / f'seed{seed}'
    config.paths['experiment'] = str(output_path)
    config.paths['logs'] = str(output_path / 'Logs')
    config.paths['checkpoint'] = str(output_path / 'Checkpoints')
    config.paths['results'] = str(output_path / 'Results')
    
    # Create directories
    for path_key in ['logs', 'checkpoint', 'results']:
        Path(config.paths[path_key]).mkdir(parents=True, exist_ok=True)
    
    solver = Solver(config)
    
    # Train
    solver.train()
    
    # Evaluate
    print(f"\nEvaluating {algorithm.upper()}: O={O}, seed={seed}")
    eval_rewards, _, _, infos = solver.eval(episodes=100)
    eval_costs = [-r for r in eval_rewards]  # Convert rewards to costs
    
    mean_cost = float(np.mean(eval_costs))
    std_cost = float(np.std(eval_costs))

    stockouts_raw = infos.get('stockouts', [])
    stockouts = []
    for s in stockouts_raw:
        arr = np.asarray(s, dtype=float)
        stockouts.append(float(arr) if arr.size == 1 else arr.tolist())

    
    # Save evaluation results
    results = {
        'algorithm': algorithm,
        'O': O,
        'seed': seed,
        'n_actions': n_actions,
        'mean_cost': mean_cost,
        'std_cost': std_cost,
        'costs': [float(c) for c in eval_costs],
        'stockouts': stockouts,
        'mean_stockout_rate': float(np.mean(infos.get('mean_stockout_rate', [0])))
    }
    
    results_file = Path(config.paths['results']) / 'evaluation_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    print(f"Mean Cost: {mean_cost:.2f} ± {std_cost:.2f}")
    
    return mean_cost, std_cost


def run_baseline(O, seed=42, n_items=2, output_dir='$SCRATCH/ift6162-project'):
    """
    Run baseline (s,S) policy
    
    Args:
        O: Common order cost (K)
        seed: Random seed
        n_items: Number of items
        output_dir: Directory to save results
    """
    print(f"\n{'='*60}")
    print(f"Solving Baseline (s,S): O={O}, seed={seed}, items={n_items}")
    print(f"{'='*60}\n")
    
    np.random.seed(seed)
    
    # Solve for optimal (s,S)
    s_list, S_list, _ = solve_independent_sS(
        n_items=n_items,
        K=O,
        h_even=-1, b_even=-19, h_uneven=-1, b_uneven=-19,
        lambda_even=20, lambda_uneven=10,
        smin=0, smax=66,
        max_steps=100,
        replications=50
    )
    
    print(f"Optimal s: {s_list}")
    print(f"Optimal S: {S_list}")
    
    # Evaluate policy
    mean_cost, std_cost, cost_list, infos = evaluate_baseline_policy(
        n_items=n_items,
        K=O,
        s_list=s_list,
        S_list=S_list,
        max_steps=100,
        n_eval_episodes=100
    )
    
    # Save results
    output_path = Path(output_dir) / f'baseline_O{O}' / f'seed{seed}'
    output_path.mkdir(parents=True, exist_ok=True)
    
    stockouts_raw = infos.get('stockouts', [])
    stockouts = []
    for s in stockouts_raw:
        arr = np.asarray(s, dtype=float)
        stockouts.append(float(arr) if arr.size == 1 else arr.tolist())
    
    results = {
        'algorithm': 'baseline',
        'O': O,
        'seed': seed,
        'n_items': n_items,
        'mean_cost': float(mean_cost),
        'std_cost': float(std_cost),
        'costs': [float(c) for c in cost_list],
        'optimal_s': s_list,
        'optimal_S': S_list,
        'stockouts': stockouts,
        'mean_stockout_rate': float(np.mean(infos.get('mean_stockout_rate', [0])))
    }
    
    results_file = output_path / 'evaluation_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    print(f"Mean Cost: {mean_cost:.2f} ± {std_cost:.2f}")
    
    return mean_cost, std_cost


def main():
    parser = argparse.ArgumentParser(description='Train single configuration')
    parser.add_argument('--algorithm', type=str, required=True, 
                       choices=['dnc', 'minmax', 'baseline'],
                       help='Algorithm to run')
    parser.add_argument('--O', type=int, required=True,
                       help='Common order cost (K)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (for DNC/MinMax)')
    parser.add_argument('--n_actions', type=int, default=2,
                       help='Number of items')
    parser.add_argument('--max_episodes', type=int, default=30000,
                       help='Maximum training episodes')
    parser.add_argument('--output_dir', type=str, 
                       default='$SCRATCH/ift6162-project',
                       help='Output directory for results')
    parser.add_argument('--neighbor_picking', type=str, default='SA',
                       choices=['greedy', 'SA'],
                       help='Neighbor picking strategy for DNC')
    parser.add_argument('--demand_dist', type=str, default='standard',
                       choices=['standard', 'heterogeneous'],
                       help='Demand distribution: standard (10/20) or heterogeneous (0.5/20)')
    
    args = parser.parse_args()
    
    if args.algorithm == 'baseline':
        run_baseline(O=args.O, seed=args.seed, n_items=args.n_actions, output_dir=args.output_dir)
    else:
        train_dnc_or_minmax(
            O=args.O,
            algorithm=args.algorithm,
            seed=args.seed,
            n_actions=args.n_actions,
            max_episodes=args.max_episodes,
            neighbor_picking=args.neighbor_picking,
            demand_dist=args.demand_dist,
            output_dir=args.output_dir
        )


if __name__ == "__main__":
    main()

