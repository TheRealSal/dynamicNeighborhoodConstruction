import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json
import seaborn as sns
import os

def load_correlation_results(base_path, correlations=[0.0, 0.25, 0.5, 0.75], seeds=[42, 43, 44], k=75, n_actions=20, scale_reward=False):
    """
    Load results for DNC (SA) and DNC (CMA-ES) across correlation levels.
    """
    scale_str = "True" if scale_reward else "False"
    
    data = []
    
    # Helper to load costs
    def get_costs_from_path(path_pattern):
        costs = []
        for seed in seeds:
            # Construct full path
            # Pattern examples:
            # 0.0 (Standard): {algo}_O{K}_N{N}_Scale{S}_Diststandard_Corr0.0[_NP{NP}]
            # > 0.0: {algo}_O{K}_N{N}_Scale{S}_Diststandard_Corr{C}[_NP{NP}]
            # Wait, 0.0 might be in standard results dir, while others in correlated_demand dir?
            # Or assume user points to base dir and we look in appropriate places.
            
            # Let's try constructing path relative to base_path.
            # If correlation > 0.0, it's likely in 'correlated_demand' subdirectory if base_path is root.
            # But the user might point directly to correlated_demand?
            # Let's search in both base_path and base_path/correlated_demand and base_path/../correlated_demand
            
            # Actually, let's assume base_path is the root experiment dir.
            # Standard (0.0) -> base_path / ...
            # Correlated (>0.0) -> base_path / 'correlated_demand' / ...
            
            # Only apply this logic if we can't find it directly.
            pass

        return costs

    algorithms = [
        ('dnc', 'SA', 'DNC (SA)'),
        ('cma_es', None, 'DNC (CMA-ES)')
    ]
    
    for algo_name, np_type, label in algorithms:
        for rho in correlations:
            costs = []
            for seed in seeds:
                # Determine folder name
                # MinMax/CMA don't have NP suffix usually, DNC does.
                np_suffix = f"_NP{np_type}" if np_type else ""
                
                folder_name = f"{algo_name}_O{k}_N{n_actions}_Scale{scale_str}_Diststandard_Corr{rho}{np_suffix}"
                
                # Determine parent directory
                # Logic: if rho == 0.0, it's often in root. If rho > 0.0, it's in 'correlated_demand'.
                # We check both to be safe.
                
                candidates = [
                    base_path / folder_name / f"seed{seed}",
                    base_path / "correlated_demand" / folder_name / f"seed{seed}",
                    base_path / "results" / "correlated_demand" / folder_name / f"seed{seed}" # Just in case
                ]
                
                found = False
                for p in candidates:
                    json_path = p / "Results" / "evaluation_results.json"
                    if json_path.exists():
                        try:
                            with open(json_path, 'r') as f:
                                res = json.load(f)
                                costs.append(res['mean_cost'])
                                found = True
                                break
                        except:
                            pass
                
                if not found:
                    costs.append(np.nan)
            
            if any(not np.isnan(c) for c in costs):
                 data.append({
                    'Algorithm': label,
                    'Correlation': rho,
                    'Cost': np.nanmean(costs),
                    'Std': np.nanstd(costs)
                })

    return data

def plot_sensitivity(data, output_file):
    if not data:
        print("No data available to plot.")
        return

    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))
    
    # Line plot
    # Hue: Algorithm
    # X: Correlation
    # Y: Cost
    
    # Convert to easier format or iterate
    algos = sorted(list(set(d['Algorithm'] for d in data)))
    correlations = sorted(list(set(d['Correlation'] for d in data)))
    
    colors = {'DNC (SA)': 'red', 'DNC (CMA-ES)': 'blue'}
    markers = {'DNC (SA)': 'o', 'DNC (CMA-ES)': 's'}
    
    for algo in algos:
        # Sort by correlation
        algo_data = sorted([d for d in data if d['Algorithm'] == algo], key=lambda x: x['Correlation'])
        xs = [d['Correlation'] for d in algo_data]
        ys = [d['Cost'] for d in algo_data]
        errs = [d['Std'] for d in algo_data]
        
        plt.errorbar(xs, ys, yerr=errs, label=algo, 
                     color=colors.get(algo, 'black'), 
                     marker=markers.get(algo, 'x'), 
                     capsize=5, linewidth=2, markersize=8)

    plt.xlabel('Correlation Strength (ρ)', fontsize=12)
    plt.ylabel('Total Cost', fontsize=12)
    plt.title('Sensitivity to Demand Correlation', fontsize=14)
    plt.xticks(correlations)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Saved plot to {output_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='$SCRATCH/ift6162-project')
    parser.add_argument('--k', type=int, default=75)
    args = parser.parse_args()
    
    results_dir = Path(os.path.expandvars(args.results_dir))
    
    print(f"Loading results from {results_dir}...")
    
    # User requested: 0.0, 0.25, 0.5, 0.75
    rhos = [0.0, 0.25, 0.5, 0.75]
    
    data = load_correlation_results(results_dir, correlations=rhos, k=args.k)
    
    plot_sensitivity(data, 'correlation_sensitivity.png')

if __name__ == "__main__":
    main()
