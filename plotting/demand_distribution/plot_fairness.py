import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json
import seaborn as sns
import os

def load_stockouts(base_dir, O=75, seeds=[42, 43, 44], scale_reward=False, n_actions=20, demand_dist='heterogeneous'):
    """
    Load stockout data for all algorithms for a specific configuration.
    Returns a dict: {AlgoName: np.array([mean_stockout_item_0, ..., mean_stockout_item_19])}
    """
    
    # Algorithms to plot
    # We want Baseline, DNC (SA), DNC (Greedy), and maybe MinMax
    algos_config = [
        {'name': 'baseline', 'np': None, 'label': 'Baseline (s,S)'},
        {'name': 'dnc', 'np': 'SA', 'label': 'DNC (SA)'},
        {'name': 'dnc', 'np': 'greedy', 'label': 'DNC (Greedy)'},
        {'name': 'minmax', 'np': 'SA', 'label': 'MinMax'}
    ]
    
    results_per_algo = {}
    
    for config in algos_config:
        algo_name = config['name']
        np_type = config['np']
        label = config['label']
        
        all_seeds_stockouts = []
        
        for seed in seeds:
            # Construct folder name based on logic in train_single_config.py
            scale_str = "True" if scale_reward else "False"
            
            # Baseline special case? 
            # In plot_common_cost.py we saw baseline path was:
            # baseline_O{k}_N{n_actions}_Scale{scale_str}_Dist{dist}
            # All others: {algo}_O{k}_N{n_actions}_Scale{scale_str}_Dist{dist}_Corr0.0[_NP{np}]
            
            if algo_name == 'baseline':
                # Baseline folder pattern (No Corr)
                folder_name = f"baseline_O{O}_N{n_actions}_Scale{scale_str}_Dist{demand_dist}"
            elif algo_name == 'dnc':
                # DNC has NP and Corr
                folder_name = f"{algo_name}_O{O}_N{n_actions}_Scale{scale_str}_Dist{demand_dist}_Corr0.0_NP{np_type}"
            else:
                # MinMax has Corr but NO NP
                folder_name = f"{algo_name}_O{O}_N{n_actions}_Scale{scale_str}_Dist{demand_dist}_Corr0.0"

            path = Path(base_dir) / folder_name / f"seed{seed}" / "Results" / "evaluation_results.json"
            
            if path.exists():
                with open(path, 'r') as f:
                    res = json.load(f)
                    # res['stockouts'] is List[List[float]] (episodes x items)
                    stockouts = np.array(res.get('stockouts', []))
                    if stockouts.size > 0:
                        # Mean over episodes for this seed
                        mean_per_item = np.mean(stockouts, axis=0)
                        all_seeds_stockouts.append(mean_per_item)
            else:
                print(f"Missing: {path}")

        if all_seeds_stockouts:
            # Average across seeds
            # Stack: (n_seeds, n_items) -> mean -> (n_items,)
            avg_stockouts = np.mean(np.stack(all_seeds_stockouts), axis=0)
            results_per_algo[label] = avg_stockouts
        else:
            print(f"No data for {label}")
            
    return results_per_algo

def plot_combined_fairness(data_standard, data_hetero, output_file, O=75):
    """
    Plot Standard and Heterogeneous side-by-side or stacked.
    """
    if not data_standard and not data_hetero:
        print("No data to plot.")
        return

    # Use first available data to define items
    if data_standard:
        first_key = next(iter(data_standard))
        n_items = len(data_standard[first_key])
    elif data_hetero:
        first_key = next(iter(data_hetero))
        n_items = len(data_hetero[first_key])
    else:
        return
        
    indices = np.arange(n_items)
    
    # Setup plot: 2 Rows, 1 Column
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
    
    # Shared settings
    # Consolidate all algos to assign consistent colors
    all_algos = set()
    if data_standard: all_algos.update(data_standard.keys())
    if data_hetero: all_algos.update(data_hetero.keys())
    all_algos = sorted(list(all_algos))
    
    # Map algo -> color
    palette = sns.color_palette("muted", len(all_algos))
    color_map = {algo: palette[i] for i, algo in enumerate(all_algos)}
    
    width = 0.8 / len(all_algos)

    # --- Plot Standard ---
    ax0 = axes[0]
    if data_standard:
        for i, algo in enumerate(all_algos):
            if algo in data_standard:
                values = data_standard[algo]
                # Calculate offset based on algo index in all_algos to match colors/pos
                ax0.bar(indices + i * width, values, width, label=algo, alpha=0.9, color=color_map[algo])
    
    ax0.set_ylabel('Avg Stockout Quantity', fontsize=12)
    ax0.set_title(f'Standard Demand (O={O})', fontsize=14)
    ax0.grid(True, axis='y', linestyle='--', alpha=0.7)
    ax0.legend(title='Algorithm', loc='upper right')

    # --- Plot Heterogeneous ---
    ax1 = axes[1]
    if data_hetero:
        for i, algo in enumerate(all_algos):
            if algo in data_hetero:
                values = data_hetero[algo]
                ax1.bar(indices + i * width, values, width, label=algo, alpha=0.9, color=color_map[algo])

    ax1.set_xlabel('Item Index', fontsize=12)
    ax1.set_ylabel('Avg Stockout Quantity', fontsize=12)
    ax1.set_title(f'Heterogeneous Demand (O={O})', fontsize=14)
    ax1.set_xticks(indices + width * (len(all_algos) - 1) / 2)
    ax1.set_xticklabels(indices)
    ax1.grid(True, axis='y', linestyle='--', alpha=0.7)
    # ax1.legend() # Legend already in top plot

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Saved plot to {output_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='$SCRATCH/ift6162-project')
    parser.add_argument('--O', type=int, default=75, help="Common Order Cost (K)")
    # parser.add_argument('--demand_dist', type=str, default='heterogeneous', choices=['standard', 'heterogeneous']) # Removed, we do both
    parser.add_argument('--scale_reward', action='store_true', help="Use scaled reward results")
    args = parser.parse_args()
    
    results_dir = os.path.expandvars(args.results_dir)
    
    print(f"Loading results for O={args.O}, Scale={args.scale_reward}...")
    
    # Load Standard
    print("Loading Standard...")
    data_std = load_stockouts(results_dir, O=args.O, scale_reward=args.scale_reward, demand_dist='standard')
    
    # Load Heterogeneous
    print("Loading Heterogeneous...")
    data_het = load_stockouts(results_dir, O=args.O, scale_reward=args.scale_reward, demand_dist='heterogeneous')
    
    filename = f'fairness_combined_O{args.O}_scale{args.scale_reward}.png'
    plot_combined_fairness(data_std, data_het, filename, O=args.O)

if __name__ == "__main__":
    main()
