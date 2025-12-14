"""
Timing Benchmark for Algorithm Comparison

This script runs 100 episodes for each algorithm (DNC, DNC Greedy, MinMax, CMA-ES)
and measures the average time per episode for comparison.

Usage:
    python experiments/timing_benchmark.py --O 75 --n_actions 2 --output_dir ./timing_results
"""

import sys
import os
import argparse
import numpy as np
import json
from pathlib import Path
from time import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from Src.parser import Parser
from Src.config import Config


def run_episode(env, model, training=True):
    """Run a single episode and return the time taken"""
    state = np.float32(env.reset(training=training))
    done = False
    step = 0
    total_reward = 0
    
    start_time = time()
    while not done:
        if step == 0:
            model.weights_changed = True
        else:
            model.weights_changed = False
            
        action, a_hat = model.get_action(state, training=training)
        new_state, reward, done, info = env.step(action=action, training=training)
        
        if training:
            model.update(state, action, a_hat, reward, new_state, done)
        
        state = new_state
        total_reward += reward
        step += 1
        
        if step > 100:  # Max steps per episode
            break
    
    episode_time = time() - start_time
    return episode_time, total_reward


def benchmark_algorithm(algorithm, O, n_actions, n_episodes=100, neighbor_picking='SA', cooling=0.1):
    """
    Benchmark a specific algorithm for n_episodes
    
    Args:
        algorithm: 'dnc', 'minmax', or 'cma_es'
        O: Common order cost
        n_actions: Number of items
        n_episodes: Number of episodes to run (default 100)
        neighbor_picking: 'SA', 'greedy', or 'cma_es'
        cooling: Cooling parameter for DNC
    
    Returns:
        dict with timing statistics
    """
    print(f"\n{'='*60}")
    print(f"Benchmarking {algorithm.upper()}: O={O}, n_actions={n_actions}, neighbor_picking={neighbor_picking}")
    print(f"{'='*60}\n")
    
    # Create parser and override arguments
    parser = Parser()
    args = parser.get_parser().parse_args(args=[])
    
    # Override with experiment settings
    args.env_name = 'JointReplenishment_py'
    args.n_actions = n_actions
    args.commonOrderCosts = O
    args.seed = 42
    args.max_episodes = 1  # We'll manually control episodes
    args.mapping = 'dnc_mapping'
    args.demand_dist = 'standard'
    args.demand_correlation = 0.0
    args.neighbor_picking = neighbor_picking
    args.cooling = cooling
    args.scale_reward = False
    args.max_steps = 100
    
    # Set algorithm-specific parameters
    if algorithm == 'minmax':
        args.SA_search_steps = 0
        args.cooling = 0
        algo_name = 'MinMax'
    elif algorithm == 'cma_es':
        args.neighbor_picking = 'cma_es'
        algo_name = 'CMA-ES'
    elif algorithm == 'dnc' and neighbor_picking == 'greedy':
        args.neighbor_picking = 'greedy'
        algo_name = 'DNC Greedy'
    else:
        args.neighbor_picking = 'SA'
        algo_name = f'DNC (SA, cooling={cooling})'
    
    # Create config
    config = Config(args)
    
    # Create model
    model = config.algo(config=config)
    env = config.env
    
    # Run episodes and collect timing
    episode_times = []
    rewards = []
    
    print(f"Running {n_episodes} episodes...")
    for episode in range(n_episodes):
        episode_time, reward = run_episode(env, model, training=True)
        episode_times.append(episode_time)
        rewards.append(reward)
        
        if (episode + 1) % 10 == 0:
            avg_time = np.mean(episode_times[-10:])
            print(f"  Episode {episode + 1}/{n_episodes} - Avg time (last 10): {avg_time:.4f}s")
    
    # Calculate statistics
    results = {
        'algorithm': algo_name,
        'neighbor_picking': neighbor_picking,
        'cooling': cooling if algorithm == 'dnc' and neighbor_picking == 'SA' else None,
        'O': O,
        'n_actions': n_actions,
        'n_episodes': n_episodes,
        'mean_episode_time': float(np.mean(episode_times)),
        'median_episode_time': float(np.median(episode_times)),
        'std_episode_time': float(np.std(episode_times)),
        'min_episode_time': float(np.min(episode_times)),
        'max_episode_time': float(np.max(episode_times)),
        'total_time': float(np.sum(episode_times)),
        'mean_reward': float(np.mean(rewards)),
        'episode_times': [float(t) for t in episode_times]
    }
    
    print(f"\nResults for {algo_name}:")
    print(f"  Mean episode time: {results['mean_episode_time']:.4f}s")
    print(f"  Median episode time: {results['median_episode_time']:.4f}s")
    print(f"  Std episode time: {results['std_episode_time']:.4f}s")
    print(f"  Total time: {results['total_time']:.2f}s")
    print(f"  Mean reward: {results['mean_reward']:.2f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Benchmark training time for different algorithms')
    parser.add_argument('--O', type=int, default=75, help='Common order cost (K)')
    parser.add_argument('--n_actions', type=int, default=2, help='Number of items')
    parser.add_argument('--n_episodes', type=int, default=100, help='Number of episodes per algorithm')
    parser.add_argument('--output_dir', type=str, default='./timing_results', help='Output directory')
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Benchmark each algorithm
    all_results = []
    
    # 1. MinMax
    results_minmax = benchmark_algorithm('minmax', args.O, args.n_actions, args.n_episodes)
    all_results.append(results_minmax)
    
    # 2. DNC Greedy
    results_dnc_greedy = benchmark_algorithm('dnc', args.O, args.n_actions, args.n_episodes, 
                                             neighbor_picking='greedy')
    all_results.append(results_dnc_greedy)
    
    # 3. DNC (SA) with cooling=0.1
    results_dnc_sa_01 = benchmark_algorithm('dnc', args.O, args.n_actions, args.n_episodes, 
                                            neighbor_picking='SA', cooling=0.1)
    all_results.append(results_dnc_sa_01)
    
    # 4. DNC (SA) with cooling=0.5
    results_dnc_sa_05 = benchmark_algorithm('dnc', args.O, args.n_actions, args.n_episodes, 
                                            neighbor_picking='SA', cooling=0.5)
    all_results.append(results_dnc_sa_05)
    
    # 5. CMA-ES
    results_cma = benchmark_algorithm('cma_es', args.O, args.n_actions, args.n_episodes,
                                     neighbor_picking='cma_es')
    all_results.append(results_cma)
    
    # Save results
    output_file = output_path / f'timing_benchmark_O{args.O}_N{args.n_actions}.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}\n")
    
    # Create comparison table
    print(f"{'Algorithm':<20} {'Mean Time (s)':<15} {'Median Time (s)':<15} {'Total Time (s)':<15}")
    print("-" * 65)
    for result in all_results:
        print(f"{result['algorithm']:<20} {result['mean_episode_time']:<15.4f} "
              f"{result['median_episode_time']:<15.4f} {result['total_time']:<15.2f}")
    
    print(f"\nResults saved to: {output_file}")
    
    # Calculate speedup
    baseline_time = results_minmax['mean_episode_time']
    print(f"\nSpeedup compared to MinMax (baseline):")
    for result in all_results[1:]:  # Skip MinMax itself
        speedup = baseline_time / result['mean_episode_time']
        print(f"  {result['algorithm']:<20}: {speedup:.2f}x {'(slower)' if speedup < 1 else '(faster)'}")


if __name__ == "__main__":
    main()
