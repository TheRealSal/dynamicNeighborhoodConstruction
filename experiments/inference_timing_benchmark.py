"""
Inference Timing Benchmark for Algorithm Comparison

This script runs evaluation episodes for each algorithm and measures the average time 
per step for inference (action selection only, no training updates).

Usage:
    python experiments/inference_timing_benchmark.py --O 75 --n_actions 2 --output_dir ./inference_timing_results
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


def run_inference_episode(env, model):
    """Run a single inference episode and return step times"""
    state = np.float32(env.reset(training=False))
    done = False
    step = 0
    total_reward = 0
    step_times = []
    
    while not done:
        if step == 0:
            model.weights_changed = True
        else:
            model.weights_changed = False
        
        # Time only the action selection (inference)
        start_time = time()
        action, _ = model.get_action(state, training=False)
        step_time = time() - start_time
        step_times.append(step_time)
        
        new_state, reward, done, info = env.step(action=action, training=False)
        
        state = new_state
        total_reward += reward
        step += 1
        
        if step > 100:  # Max steps per episode
            break
    
    return step_times, total_reward, step


def benchmark_inference(algorithm, O, n_actions, n_episodes=100, neighbor_picking='SA', cooling=0.1):
    """
    Benchmark inference time for a specific algorithm
    
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
    print(f"Benchmarking Inference - {algorithm.upper()}: O={O}, n_actions={n_actions}, neighbor_picking={neighbor_picking}")
    print(f"{'='*60}\n")
    
    # Create parser and override arguments
    parser = Parser()
    args = parser.get_parser().parse_args(args=[])
    
    # Override with experiment settings
    args.env_name = 'JointReplenishment_py'
    args.n_actions = n_actions
    args.commonOrderCosts = O
    args.seed = 42
    args.max_episodes = 1
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
    
    # Run episodes and collect step timing
    all_step_times = []
    rewards = []
    total_steps = 0
    
    print(f"Running {n_episodes} evaluation episodes...")
    for episode in range(n_episodes):
        env.seed(42 + episode)
        step_times, reward, num_steps = run_inference_episode(env, model)
        all_step_times.extend(step_times)
        rewards.append(reward)
        total_steps += num_steps
        
        if (episode + 1) % 10 == 0:
            avg_step_time = np.mean([t for times in [step_times] for t in times])
            print(f"  Episode {episode + 1}/{n_episodes} - Avg step time: {avg_step_time*1000:.2f}ms")
    
    # Calculate statistics
    results = {
        'algorithm': algo_name,
        'neighbor_picking': neighbor_picking,
        'cooling': cooling if algorithm == 'dnc' and neighbor_picking == 'SA' else None,
        'O': O,
        'n_actions': n_actions,
        'n_episodes': n_episodes,
        'total_steps': total_steps,
        'mean_step_time': float(np.mean(all_step_times)),
        'median_step_time': float(np.median(all_step_times)),
        'std_step_time': float(np.std(all_step_times)),
        'min_step_time': float(np.min(all_step_times)),
        'max_step_time': float(np.max(all_step_times)),
        'p95_step_time': float(np.percentile(all_step_times, 95)),
        'p99_step_time': float(np.percentile(all_step_times, 99)),
        'mean_reward': float(np.mean(rewards)),
        'mean_steps_per_episode': total_steps / n_episodes,
    }
    
    print(f"\nInference Results for {algo_name}:")
    print(f"  Mean step time: {results['mean_step_time']*1000:.2f}ms")
    print(f"  Median step time: {results['median_step_time']*1000:.2f}ms")
    print(f"  95th percentile: {results['p95_step_time']*1000:.2f}ms")
    print(f"  99th percentile: {results['p99_step_time']*1000:.2f}ms")
    print(f"  Total steps: {total_steps}")
    print(f"  Mean reward: {results['mean_reward']:.2f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Benchmark inference time for different algorithms')
    parser.add_argument('--O', type=int, default=75, help='Common order cost (K)')
    parser.add_argument('--n_actions', type=int, default=2, help='Number of items')
    parser.add_argument('--n_episodes', type=int, default=100, help='Number of episodes per algorithm')
    parser.add_argument('--output_dir', type=str, default='./inference_timing_results', help='Output directory')
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Benchmark each algorithm
    all_results = []
    
    # 1. MinMax
    results_minmax = benchmark_inference('minmax', args.O, args.n_actions, args.n_episodes)
    all_results.append(results_minmax)
    
    # 2. DNC Greedy
    results_dnc_greedy = benchmark_inference('dnc', args.O, args.n_actions, args.n_episodes, 
                                             neighbor_picking='greedy')
    all_results.append(results_dnc_greedy)
    
    # 3. DNC (SA) with cooling=0.1
    results_dnc_sa_01 = benchmark_inference('dnc', args.O, args.n_actions, args.n_episodes, 
                                            neighbor_picking='SA', cooling=0.1)
    all_results.append(results_dnc_sa_01)
    
    # 4. DNC (SA) with cooling=0.5
    results_dnc_sa_05 = benchmark_inference('dnc', args.O, args.n_actions, args.n_episodes, 
                                            neighbor_picking='SA', cooling=0.5)
    all_results.append(results_dnc_sa_05)
    
    # 5. CMA-ES
    results_cma = benchmark_inference('cma_es', args.O, args.n_actions, args.n_episodes,
                                     neighbor_picking='cma_es')
    all_results.append(results_cma)
    
    # Save results
    output_file = output_path / f'inference_timing_benchmark_O{args.O}_N{args.n_actions}.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("INFERENCE TIMING SUMMARY")
    print(f"{'='*60}\n")
    
    # Create comparison table
    print(f"{'Algorithm':<25} {'Mean (ms)':<12} {'Median (ms)':<12} {'P95 (ms)':<12} {'P99 (ms)':<12}")
    print("-" * 73)
    for result in all_results:
        print(f"{result['algorithm']:<25} "
              f"{result['mean_step_time']*1000:<12.2f} "
              f"{result['median_step_time']*1000:<12.2f} "
              f"{result['p95_step_time']*1000:<12.2f} "
              f"{result['p99_step_time']*1000:<12.2f}")
    
    print(f"\nResults saved to: {output_file}")
    
    # Calculate speedup
    baseline_time = results_minmax['mean_step_time']
    print(f"\nInference speedup compared to MinMax (baseline):")
    for result in all_results[1:]:  # Skip MinMax itself
        speedup = baseline_time / result['mean_step_time']
        slowdown = result['mean_step_time'] / baseline_time
        if speedup < 1:
            print(f"  {result['algorithm']:<25}: {slowdown:.2f}x slower ({result['mean_step_time']*1000:.2f}ms vs {baseline_time*1000:.2f}ms)")
        else:
            print(f"  {result['algorithm']:<25}: {speedup:.2f}x faster ({result['mean_step_time']*1000:.2f}ms vs {baseline_time*1000:.2f}ms)")


if __name__ == "__main__":
    main()
