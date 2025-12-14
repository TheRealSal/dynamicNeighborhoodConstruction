import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json
import seaborn as sns

def load_results(base_dir, k_values=[0, 75, 200], seeds=[42, 43, 44], scale_reward=False, n_actions=20):
    """
    Load results for all algorithms across K values and seeds.
    Algorithms: Baseline, MinMax, DNC (SA), DNC (Greedy)
    """
    
    # Define experimental configurations to look for
    # Label -> (Algorithm, NeighborPicking, DemandCorrelation, DemandDist)
    # Note: Baseline doesn't use NeighborPicking, but the path might not include it or might be default?
    # Actually, run_baseline uses a different path structure:
    # baseline_O{O}_N{n_items}_Scale{scale_reward}_Dist{demand_dist}/seed{seed}
    # while train_dnc_or_minmax uses:
    # {algorithm}_O{O}_N{n_actions}_Scale{scale_reward}_Dist{demand_dist}_Corr{demand_correlation}_NP{neighbor_picking}/seed{seed}
    
    # We need to handle these two patterns.
    
    data = []
    
    for k in k_values:
        # 1. Baseline
        costs = []
        for seed in seeds:
            # Baseline Path Pattern
            # Note: run_baseline in train_single_config.py doesn't put _Corr or _NP.
            # Let's verify Step 165/188/212.
            # Step 128/146/219 updated train_dnc_or_minmax.
            # Step 146 updated run_baseline too? 
            # Looking at Step 165: 
            # output_path = Path(output_dir) / f'baseline_O{O}_N{n_items}_Scale{scale_reward}_Dist{demand_dist}' / f'seed{seed}'
            # Wait, did I miss updating baseline path to include Corr?
            # Step 122: "I have updated experiments/train_single_config.py ... The new path format is ..._Dist...".
            # Step 146: User updated path to include _Corr. But the diff block (Step 146) only shows changes to line 65 (train_dnc_or_minmax path).
            # It seems run_baseline path (line 165) MIGHT NOT have been updated to include Corr or NP.
            # Let's assume Baseline path is: baseline_O{O}_N{N}_Scale{S}_Dist{D}/seed{seed}
            # Or: baseline_O{O}_N{N}_Scale{S}_Dist{D}_Corr{C}/seed{seed} if it was updated later?
            # Check Step 140 view of train_single_config.py.
            # Line 165: output_path = Path(output_dir) / f'baseline_O{...}_Dist{demand_dist}' / f'seed{seed}'
            # It does NOT have Corr.
            # So Baseline paths don't have Corr or NP.
            
            # Construct path
            scale_str = "True" if scale_reward else "False"
            # Baseline assumes standard dist and default corr (0.0 not in name).
            # But wait, user said "commonCost" experiments.
            # Baseline folder: baseline_O{K}_N{n_actions}_Scale{scale_str}_Dist{standard}
            
            folder_name = f"baseline_O{k}_N{n_actions}_Scale{scale_str}_Diststandard"
            path = Path(base_dir) / folder_name / f"seed{seed}" / "Results" / "evaluation_results.json"
            
            if path.exists():
                with open(path, 'r') as f:
                    res = json.load(f)
                    costs.append(res['mean_cost'])
            else:
                print(f"Missing: {path}")
                costs.append(np.nan)
        
        if costs:
            data.append({
                'Algorithm': 'Baseline (s,S)',
                'K': k,
                'Cost': np.nanmean(costs),
                'Std': np.nanstd(costs),
                'Type': 'Independent'
            })


        # 2. DNC (SA)
        # 3. DNC (Greedy)
        # 4. MinMax
        
        # Path pattern for these:
        # {algo}_O{O}_N{N}_Scale{S}_Dist{D}_Corr{C}_NP{NP}
        
        algos = [
            ('dnc', 'SA', 'DNC (SA)'), 
            ('dnc', 'greedy', 'DNC (Greedy)'),
            ('minmax', 'SA', 'MinMax') # MinMax ignores NP usually, defaults to SA in parser but SA steps=0
        ]
        
        for algo_name, np_type, label in algos:
            costs = []
            for seed in seeds:
                # Path depends on algorithm Type
                scale_str = "True" if scale_reward else "False"
                
                if algo_name == 'dnc':
                     # DNC has NP
                     folder_name = f"{algo_name}_O{k}_N{n_actions}_Scale{scale_str}_Diststandard_Corr0.0_NP{np_type}"
                else:
                     # MinMax (and others) do NOT have NP
                     folder_name = f"{algo_name}_O{k}_N{n_actions}_Scale{scale_str}_Diststandard_Corr0.0"

                path = Path(base_dir) / folder_name / f"seed{seed}" / "Results" / "evaluation_results.json"
                
                if path.exists():
                    with open(path, 'r') as f:
                        res = json.load(f)
                        costs.append(res['mean_cost'])
                else:
                    # Fallback check: maybe previous runs didn't have NP in path?
                    # The user will re-run with new script. So assume new path.
                    print(f"Missing: {path}")
                    costs.append(np.nan)
            
            if costs:
                data.append({
                    'Algorithm': label,
                    'K': k,
                    'Cost': np.nanmean(costs),
                    'Std': np.nanstd(costs),
                    'Type': 'RL'
                })

    return data


def plot_results(data, scale_reward, output_file):
    if not data:
        print("No data to plot.")
        return

    # Extract data for plotting
    ks = sorted(list(set(d['K'] for d in data)))
    algos = sorted(list(set(d['Algorithm'] for d in data)))
    
    # Setup plot
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))
    
    # Create grouped bar chart
    # Position of bars on x-axis
    bar_width = 0.2
    indices = np.arange(len(ks))
    
    # Color palette
    colors = sns.color_palette("muted", len(algos))
    
    for i, algo in enumerate(algos):
        means = []
        stds = []
        for k in ks:
            item = next((d for d in data if d['Algorithm'] == algo and d['K'] == k), None)
            if item:
                means.append(item['Cost'])
                stds.append(item['Std'])
            else:
                means.append(0)
                stds.append(0)
        
        plt.bar(indices + i * bar_width, means, bar_width, label=algo, yerr=stds, capsize=5, color=colors[i])

    plt.xlabel('Common Order Cost (K)', fontsize=12)
    plt.ylabel('Total Cost', fontsize=12)
    plt.title(f'Performance vs Coupling (Scale Reward: {scale_reward})', fontsize=14)
    plt.xticks(indices + bar_width * (len(algos) - 1) / 2, ks)
    plt.legend(title='Algorithm')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Saved plot to {output_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='$SCRATCH/ift6162-project')
    args = parser.parse_args()
    
    # Expand env vars
    results_dir = str(Path(args.results_dir).expanduser()).replace('$SCRATCH', '/scratch/salman/ift6162-project') # Adjust as needed or rely on user shell expansion
    # actually os.path.expandvars is better but scratch var might be shell specific.
    # We will trust the user passes a resolved path or we resolve generic home.
    # For now, let's assume specific path if passed, else try to resolve common patterns.
    import os
    results_dir = os.path.expandvars(args.results_dir)

    # 1. Plot with scale_reward = False
    print("Generating plot for Scale Reward = False...")
    data_false = load_results(results_dir, scale_reward=False)
    plot_results(data_false, False, 'common_cost_noscale.png')
    
    # 2. Plot with scale_reward = True
    print("\nGenerating plot for Scale Reward = True...")
    data_true = load_results(results_dir, scale_reward=True)
    plot_results(data_true, True, 'common_cost_scale.png')

if __name__ == "__main__":
    main()
