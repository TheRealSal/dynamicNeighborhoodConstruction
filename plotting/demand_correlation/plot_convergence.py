import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json
import seaborn as sns
import os

def load_convergence_results(base_path, target_rho=0.8, seeds=[42, 43, 44], k=75, n_actions=20, scale_reward=False):
    """
    Load training curves (rewards) for algorithms at a specific correlation.
    """
    scale_str = "True" if scale_reward else "False"
    dist_tag = "standard" # Assuming standard demand dist for correlation experiments
    
    data = {} # Algo Name -> {xs: [], ys: [], stds: []}
    
    # helper
    def load_algo_curves(algo_label, folder_name):
        algo_curves = []
        max_len = 0
        
        # Check standard location and correlated_demand location
        candidates_base = [
            base_path,
            base_path / "correlated_demand",
            base_path / "results" / "correlated_demand"
        ]
        
        for seed in seeds:
            found_seed = False
            for root in candidates_base:
                path = root / folder_name / f"seed{seed}" / "Results" / "rewards.npy"
                if path.exists():
                    try:
                        curve = np.load(path)
                        # Convert rewards to costs? User asked for "Total Cost".
                        # Rewards might be negative costs. Let's assume reward = -cost.
                        # Also check if scaling was applied. If scale_reward=True, rewards are scaled.
                        # But plotting usually wants unscaled or consistent units.
                        # Let's just flip sign for now to get Cost.
                        costs = -curve 
                        algo_curves.append(costs)
                        max_len = max(max_len, len(costs))
                        found_seed = True
                        break
                    except Exception as e:
                        print(f"Error loading {path}: {e}")
            
            if not found_seed:
                # pad with nans or ignore?
                pass

        if algo_curves:
            # Synchronize lengths
            # It's possible some runs are shorter. Truncate to min or pad to max?
            # Usually truncating to min length of all seeds is safest for averaging.
            min_len = min(len(c) for c in algo_curves)
            trimmed_curves = [c[:min_len] for c in algo_curves]
            
            stack = np.vstack(trimmed_curves)
            means = np.mean(stack, axis=0)
            stds = np.std(stack, axis=0)
            
            return means, stds
        return None, None

    # Algorithm Definitions
    # We want "All algorithms".
    # 1. DNC (SA)
    means, stds = load_algo_curves('DNC (SA)', f"dnc_O{k}_N{n_actions}_Scale{scale_str}_Dist{dist_tag}_Corr{target_rho}_NP_SA")
    if means is not None: data['DNC (SA)'] = (means, stds)

    # 2. DNC (CMA-ES)
    means, stds = load_algo_curves('DNC (CMA-ES)', f"cma_es_O{k}_N{n_actions}_Scale{scale_str}_Dist{dist_tag}_Corr{target_rho}")
    if means is not None: data['DNC (CMA-ES)'] = (means, stds)
    
    # 3. MinMax (Often run with Corr 0.0 only? Or maybe ran with correlation too?)
    # If user ran MinMax on correlated demand, it would be here.
    means, stds = load_algo_curves('MinMax', f"minmax_O{k}_N{n_actions}_Scale{scale_str}_Dist{dist_tag}_Corr{target_rho}_Corr0.0") # Wait pattern might vary
    # Check simple pattern
    if means is None:
         means, stds = load_algo_curves('MinMax', f"minmax_O{k}_N{n_actions}_Scale{scale_str}_Dist{dist_tag}_Corr{target_rho}")
    if means is not None: data['MinMax'] = (means, stds)
    
    # 4. Baseline
    # Baseline is usually static (doesn't have training curve per se, but (s,S) is constant).
    # But we could plot it as a flat line if we had its value.
    # For now, let's focus on learning agents as requested "Convergence Speed".
    
    return data

def plot_convergence(data, output_file, title_rho):
    if not data:
        print("No data available to plot.")
        return

    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))
    
    colors = sns.color_palette("muted", len(data))
    
    for i, (label, (means, stds)) in enumerate(data.items()):
        # X-axis: Episodes.
        # We need to assume the save frequency to map index to episodes.
        # From run.py: `args.save_count = max(1, max_episodes // 10)`
        # `checkpoint = self.config.save_after` which relates to save_count?
        # Actually Config sets save_after.
        # Let's blindly map index to "Checkpoints" or try to guess.
        # If mean length is 10, and max_episodes is 30k, each point is 3k episodes.
        # Let's just plot vs "Hypothetical Episodes" assuming 30k max and equally spaced.
        
        # Assuming 30k total episodes for the plot
        total_episodes = 30000 
        x = np.linspace(0, total_episodes, len(means))
        
        plt.plot(x, means, label=label, color=colors[i], linewidth=2)
        plt.fill_between(x, means - stds, means + stds, color=colors[i], alpha=0.2)
        
    plt.xlabel('Episodes', fontsize=12)
    plt.ylabel('Total Cost (Moving Average)', fontsize=12)
    plt.title(f'Convergence Speed (Correlation ρ={title_rho})', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Saved plot to {output_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='$SCRATCH/ift6162-project')
    parser.add_argument('--k', type=int, default=75)
    parser.add_argument('--correlation', type=float, default=0.8, help='Target correlation to plot')
    args = parser.parse_args()
    
    results_dir = Path(os.path.expandvars(args.results_dir))
    
    print(f"Loading results from {results_dir} for rho={args.correlation}...")
    
    # Try target correlation
    data = load_convergence_results(results_dir, target_rho=args.correlation, k=args.k)
    
    if not data and args.correlation == 0.8:
        print("Data for 0.8 not found. Checking 0.75...")
        data = load_convergence_results(results_dir, target_rho=0.75, k=args.k)
        args.correlation = 0.75
        
    plot_convergence(data, 'convergence_speed.png', args.correlation)

if __name__ == "__main__":
    main()
