"""
Analysis and Plotting Script
Creates bar chart comparing DNC vs Baseline at different coupling levels
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_results(filename='experiment_results.json'):
    """
    Load experiment results from JSON file
    """
    results_dir = Path(__file__).parent / 'results'
    filepath = results_dir / filename
    
    if not filepath.exists():
        raise FileNotFoundError(f"Results file not found: {filepath}")
    
    with open(filepath, 'r') as f:
        results = json.load(f)
    
    return results


def create_bar_chart(results, save_path=None):
    """
    Create bar chart comparing DNC vs Baseline
    
    Args:
        results: Dictionary with experiment results
        save_path: Path to save the figure (optional)
    """
    config_names = ['Decoupled\n(K=0)', 'Standard\n(K=75)', 'Strong Coupling\n(K=200)']
    config_keys = ['decoupled', 'standard', 'strong_coupling']
    
    dnc_means = []
    dnc_stds = []
    baseline_means = []
    baseline_stds = []
    
    for key in config_keys:
        if key in results:
            dnc_means.append(results[key]['dnc_mean'])
            dnc_stds.append(results[key]['dnc_std'])
            baseline_means.append(results[key]['baseline_mean'])
            baseline_stds.append(results[key]['baseline_std'])
    
    x = np.arange(len(config_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create bars
    bars1 = ax.bar(x - width/2, baseline_means, width, 
                   yerr=baseline_stds, label='Baseline (s,S)', 
                   color='#FF6B6B', alpha=0.8, capsize=5)
    bars2 = ax.bar(x + width/2, dnc_means, width,
                   yerr=dnc_stds, label='DNC', 
                   color='#4ECDC4', alpha=0.8, capsize=5)
    
    # Add value labels on bars
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom', fontsize=9)
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    
    # Customize plot
    ax.set_ylabel('Total Cost', fontsize=12, fontweight='bold')
    ax.set_xlabel('Coupling Level', fontsize=12, fontweight='bold')
    ax.set_title('DNC vs Baseline (s,S) Policy: Performance by Coupling Level', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(config_names, fontsize=11)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    else:
        plt.show()
    
    return fig, ax


def print_detailed_analysis(results):
    """
    Print detailed analysis of results
    """
    print("\n" + "="*70)
    print("DETAILED ANALYSIS")
    print("="*70)
    
    config_keys = ['decoupled', 'standard', 'strong_coupling']
    config_labels = {
        'decoupled': 'Decoupled (K=0)',
        'standard': 'Standard (K=75)',
        'strong_coupling': 'Strong Coupling (K=200)'
    }
    
    for key in config_keys:
        if key not in results:
            continue
        
        res = results[key]
        print(f"\n{config_labels[key]}:")
        print(f"  Baseline (s,S):")
        print(f"    Mean Cost: {res['baseline_mean']:.2f}")
        print(f"    Std Dev:   {res['baseline_std']:.2f}")
        print(f"    Optimal s: {res['baseline_s']}")
        print(f"    Optimal S: {res['baseline_S']}")
        
        print(f"  DNC:")
        print(f"    Mean Cost: {res['dnc_mean']:.2f}")
        print(f"    Std Dev:   {res['dnc_std']:.2f}")
        
        improvement = ((res['baseline_mean'] - res['dnc_mean']) / res['baseline_mean']) * 100
        print(f"  Improvement: {improvement:.1f}%")
        
        # Hypothesis check
        if key == 'decoupled':
            print(f"  Hypothesis: DNC ≈ Baseline (expected)")
            if abs(improvement) < 5:
                print(f"  ✓ Confirmed: DNC performs similarly to baseline")
            else:
                print(f"  ✗ Not confirmed: DNC differs by {abs(improvement):.1f}%")
        elif key == 'standard':
            print(f"  Hypothesis: DNC < Baseline (expected)")
            if improvement > 0:
                print(f"  ✓ Confirmed: DNC outperforms baseline by {improvement:.1f}%")
            else:
                print(f"  ✗ Not confirmed: DNC underperforms by {abs(improvement):.1f}%")
        elif key == 'strong_coupling':
            print(f"  Hypothesis: DNC << Baseline (expected)")
            if improvement > 10:
                print(f"  ✓ Confirmed: DNC significantly outperforms by {improvement:.1f}%")
            else:
                print(f"  ✗ Partially confirmed: DNC improves by {improvement:.1f}%")


def main():
    """
    Main analysis function
    """
    try:
        results = load_results()
        
        # Print detailed analysis
        print_detailed_analysis(results)
        
        # Create and save plot
        results_dir = Path(__file__).parent / 'results'
        results_dir.mkdir(exist_ok=True)
        save_path = results_dir / 'dnc_vs_baseline_comparison.png'
        
        create_bar_chart(results, save_path=save_path)
        
        print(f"\n{'='*70}")
        print("Analysis complete!")
        print(f"{'='*70}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease run the experiment first:")
        print("  python experiments/run_experiment.py")


if __name__ == "__main__":
    main()

