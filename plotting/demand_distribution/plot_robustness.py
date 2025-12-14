import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json
import seaborn as sns
import os

def load_scenario_results(base_path, scenario_name, seeds=[42, 43, 44], k=75, n_actions=20, scale_reward=False):
    """
    Load results for a specific scenario (Standard or Heterogeneous).
    
    Args:
        base_path (Path): The directory containing the algorithm folders for this scenario.
        scenario_name (str): "Standard" or "Heterogeneous". Used for determining folder patterns if needed.
                             Standard expects 'Diststandard', Heterogeneous expects 'Distheterogeneous'.
    """
    dist_tag = "standard" if scenario_name == 'Standard' else "heterogeneous"
    scale_str = "True" if scale_reward else "False"
    
    data = []
    
    # Define algorithms to look for
    # Format: (Algorithm Label, Folder Pattern Template)
    
    # helper to check if path exists
    def get_costs(folder_name):
        costs = []
        for seed in seeds:
            path = base_path / folder_name / f"seed{seed}" / "Results" / "evaluation_results.json"
            if path.exists():
                try:
                    with open(path, 'r') as f:
                        res = json.load(f)
                        costs.append(res['mean_cost'])
                except Exception as e:
                    print(f"Error reading {path}: {e}")
                    costs.append(np.nan)
            else:
                # Debug print only if needed
                # print(f"Missing: {path}")
                costs.append(np.nan)
        return costs

    # 1. Baseline
    # Pattern: baseline_O{K}_N{N}_Scale{S}_Dist{D}
    baseline_folder = f"baseline_O{k}_N{n_actions}_Scale{scale_str}_Dist{dist_tag}"
    baseline_costs = get_costs(baseline_folder)
    if any(not np.isnan(c) for c in baseline_costs):
         data.append({
            'Algorithm': 'Baseline',
            'Scenario': scenario_name,
            'Cost': np.nanmean(baseline_costs),
            'Std': np.nanstd(baseline_costs),
            'RawCosts': baseline_costs
        })

    # 2. MinMax
    # Pattern: minmax_O{K}_N{N}_Scale{S}_Dist{D}_Corr0.0
    minmax_folder = f"minmax_O{k}_N{n_actions}_Scale{scale_str}_Dist{dist_tag}_Corr0.0"
    minmax_costs = get_costs(minmax_folder)
    if any(not np.isnan(c) for c in minmax_costs):
        data.append({
            'Algorithm': 'MinMax',
            'Scenario': scenario_name,
            'Cost': np.nanmean(minmax_costs),
            'Std': np.nanstd(minmax_costs),
             'RawCosts': minmax_costs
        })

    # 3. DNC (SA)
    # Pattern: dnc_O{K}_N{N}_Scale{S}_Dist{D}_Corr0.0_NP_SA
    dnc_sa_folder = f"dnc_O{k}_N{n_actions}_Scale{scale_str}_Dist{dist_tag}_Corr0.0_NP_SA"
    dnc_sa_costs = get_costs(dnc_sa_folder)
    if any(not np.isnan(c) for c in dnc_sa_costs):
        data.append({
            'Algorithm': 'DNC',
            'Scenario': scenario_name,
            'Cost': np.nanmean(dnc_sa_costs),
            'Std': np.nanstd(dnc_sa_costs),
            'RawCosts': dnc_sa_costs
        })

    # 4. DNC (Greedy)
    # Pattern: dnc_O{K}_N{N}_Scale{S}_Dist{D}_Corr0.0_NP_greedy
    dnc_greedy_folder = f"dnc_O{k}_N{n_actions}_Scale{scale_str}_Dist{dist_tag}_Corr0.0_NP_greedy"
    dnc_greedy_costs = get_costs(dnc_greedy_folder)
    if any(not np.isnan(c) for c in dnc_greedy_costs):
        data.append({
            'Algorithm': 'DNC Greedy',
            'Scenario': scenario_name,
            'Cost': np.nanmean(dnc_greedy_costs),
            'Std': np.nanstd(dnc_greedy_costs),
             'RawCosts': dnc_greedy_costs
        })
        
    return data

def plot_robustness(data, output_file):
    if not data:
        print("No data available to plot.")
        return

    # Prepare data for seaborn
    # We want grouped bar chart: Inner group = Scenario, Outer group = Algorithm?
    # Prompt: "X-Axis: Scenarios... Grouped by Algorithm"?
    # Actually usually it's "X-Axis: Scenarios" and bars for each Algorithm side-by-side.
    
    # Let's organize it:
    # X-axis: Standard vs Heterogeneous
    # Hue: Algorithm
    
    scenarios = ['Standard', 'Heterogeneous']
    algorithms = ['Baseline', 'MinMax', 'DNC', 'DNC Greedy']
    
    # Filter data to only include these (if present)
    
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))
    
    # We need to restructure data for easier plotting if we want to use direct lists or DataFrame
    # Let's just manually build the bars for full control or use sns.barplot if we had a dataframe.
    # Manual construction is safer for specific ordering.
    
    x = np.arange(len(scenarios))
    width = 0.2
    offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]
    
    colors = sns.color_palette("muted", len(algorithms))
    
    for i, algo in enumerate(algorithms):
        means = []
        stds = []
        for scen in scenarios:
            item = next((d for d in data if d['Algorithm'] == algo and d['Scenario'] == scen), None)
            if item:
                means.append(item['Cost'])
                stds.append(item['Std'])
            else:
                means.append(0)
                stds.append(0)
        
        plt.bar(x + offsets[i], means, width, label=algo, yerr=stds, capsize=5, color=colors[i])

    plt.xlabel('Scenario', fontsize=12)
    plt.ylabel('Total Cost', fontsize=12)
    plt.title('Robustness Comparison: Standard vs Heterogeneous Demand', fontsize=14)
    plt.xticks(x, scenarios)
    plt.legend(title='Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Saved plot to {output_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='$SCRATCH/ift6162-project')
    parser.add_argument('--k', type=int, default=75, help='Common ordering cost K to plot')
    args = parser.parse_args()
    
    results_dir = Path(os.path.expandvars(args.results_dir))
    
    print(f"Loading results from {results_dir}...")
    
    all_data = []
    
    # 1. Load Standard Results
    # Standard results are likely directly in results_dir or in a specific subfolder if user organized it that way.
    # Based on slurm scripts: experiments/commonCost/slurm... -> output_dir $SCRATCH/ift6162-project
    # So Standard folders are in root of results_dir.
    print("Loading Standard Scenario...")
    std_data = load_scenario_results(results_dir, 'Standard', k=args.k)
    all_data.extend(std_data)
    
    # 2. Load Heterogeneous Results
    # Based on slurm scripts: experiments/demandHeterogeneity/slurm... -> output_dir $SCRATCH/ift6162-project/heterogeneous_demand
    hetero_path = results_dir / 'heterogeneous_demand'
    print(f"Loading Heterogeneous Scenario from {hetero_path}...")
    if hetero_path.exists():
        hetero_data = load_scenario_results(hetero_path, 'Heterogeneous', k=args.k)
        all_data.extend(hetero_data)
    else:
        print(f"Warning: Heterogeneous directory {hetero_path} not found.")

    plot_robustness(all_data, 'robustness_comparison.png')

if __name__ == "__main__":
    main()
