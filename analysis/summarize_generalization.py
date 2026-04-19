"""
Summarize all generalization experiment results into a clean table.
Reads JSON files from checkpoints/gen_*/generalization_results.json.
"""

import json
import numpy as np
from pathlib import Path
import os

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

VARIANTS = ['ffn', 'glu', 'moe', 'moe_glu']
SEEDS = [42, 137, 256, 512, 1024]


def collect_results(experiment_prefix):
    """Collect results for one experiment across all variants and seeds."""
    all_results = {}

    for variant in VARIANTS:
        all_results[variant] = {}
        for seed in SEEDS:
            path = Path(f"checkpoints/{experiment_prefix}_{variant}_s{seed}/generalization_results.json")
            if not path.exists():
                continue
            with open(path) as f:
                data = json.load(f)
            all_results[variant][seed] = data['results']

    return all_results


def print_summary(title, results, key_subsets):
    """Print a summary table for one experiment."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")

    header = f"{'Variant':<10}"
    for name in key_subsets:
        header += f" {name:<20}"
    print(header)
    print("-" * 80)

    for variant in VARIANTS:
        row = f"{variant:<10}"
        for name in key_subsets:
            accs = []
            for seed in SEEDS:
                if seed in results[variant] and name in results[variant][seed]:
                    accs.append(results[variant][seed][name]['acc'])
            if accs:
                m, s = np.mean(accs), np.std(accs)
                row += f" {m:.1%} +/- {s:.1%}     "
            else:
                row += f" {'N/A':<20}"
        print(row)

    # Print per-seed details
    print(f"\nPer-seed details:")
    for variant in VARIANTS:
        for seed in SEEDS:
            if seed not in results[variant]:
                continue
            r = results[variant][seed]
            detail = f"  {variant} s{seed}:"
            for name in key_subsets:
                if name in r:
                    detail += f"  {name}={r[name]['acc']:.1%}"
            print(detail)


def main():
    # Experiment 1: Exclude carry >= 3
    # Check both 3-digit and 5-digit runs
    results1 = collect_results("gen5_excludecarry3")
    if not any(len(v) > 0 for v in results1.values()):
        results1 = collect_results("gen_excludecarry3")
    has_results1 = any(len(v) > 0 for v in results1.values())

    if has_results1:
        print_summary(
            "Experiment 1: Train on L<=2, Test on L>=3",
            results1,
            ['train_pool', 'held_out_pool', 'L=0', 'L=1', 'L=2', 'L=3']
        )
    else:
        print("Experiment 1 (exclude carry >= 3): No results found yet.")

    # Experiment 2: Exclude ones >= 3
    results2 = collect_results("gen5_excludeones3")
    if not any(len(v) > 0 for v in results2.values()):
        results2 = collect_results("gen_excludeones3")
    has_results2 = any(len(v) > 0 for v in results2.values())

    if has_results2:
        print_summary(
            "Experiment 2: Train without carry-triggering inputs, Test on them",
            results2,
            ['train_pool', 'held_out_pool', 'ones_lt3', 'ones_ge3', 'L=0', 'L=1']
        )
    else:
        print("Experiment 2 (exclude ones >= 3): No results found yet.")

    # Save combined summary
    combined = {
        'exclude_carry_ge3': {
            v: {str(s): results1[v].get(s, {}) for s in SEEDS}
            for v in VARIANTS
        },
        'exclude_ones_ge3': {
            v: {str(s): results2[v].get(s, {}) for s in SEEDS}
            for v in VARIANTS
        },
    }
    out_path = Path("results_generalization.json")
    with open(out_path, 'w') as f:
        json.dump(combined, f, indent=2)
    print(f"\nCombined results saved to {out_path}")


if __name__ == "__main__":
    main()
