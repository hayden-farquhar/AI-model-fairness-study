#!/usr/bin/env python3
"""
04_performance_disparity_analysis.py

Quantify the contribution of view type, age, and sex to performance disparity.

Author: Hayden Farquhar, MBBS MPHTM
"""

import numpy as np
import pandas as pd
from pathlib import Path

RESULTS_DIR = Path("results")


def calculate_disparity_contribution(metrics_df: pd.DataFrame, model: str, dataset: str) -> dict:
    """
    Calculate performance disparity analysis for a model-dataset combination.
    
    Measures the proportion of total sensitivity range explained by each factor.
    """
    
    subset = metrics_df[
        (metrics_df["model"] == model) & 
        (metrics_df["dataset"] == dataset) &
        (metrics_df["group_type"] != "overall")
    ].copy()
    
    # Calculate sensitivity range for each factor
    ranges = {}
    for factor in ["view_type", "age_group", "sex"]:
        factor_data = subset[subset["group_type"] == factor]
        if len(factor_data) >= 2:
            sens_range = factor_data["sensitivity"].max() - factor_data["sensitivity"].min()
            ranges[factor] = sens_range
        else:
            ranges[factor] = 0
    
    # Total range is max across all factors
    total_range = max(ranges.values()) if ranges else 0
    
    # Calculate proportion explained by each factor
    contributions = {}
    for factor, range_val in ranges.items():
        contributions[f"{factor}_range"] = range_val
        contributions[f"{factor}_contribution"] = range_val / total_range if total_range > 0 else 0
    
    contributions["total_range"] = total_range
    contributions["model"] = model
    contributions["dataset"] = dataset
    
    # View dominance = % of disparity attributable to view type
    if total_range > 0:
        contributions["view_dominance"] = ranges.get("view_type", 0) / total_range
    else:
        contributions["view_dominance"] = 0
    
    return contributions


def main():
    """Main performance disparity analysis analysis."""
    
    # Load performance metrics
    metrics_df = pd.read_csv(RESULTS_DIR / "performance_metrics.csv")
    
    # Get unique model-dataset combinations
    combos = metrics_df[["model", "dataset"]].drop_duplicates()
    
    results = []
    for _, row in combos.iterrows():
        model, dataset = row["model"], row["dataset"]
        
        # Skip overall-only entries
        subset = metrics_df[
            (metrics_df["model"] == model) & 
            (metrics_df["dataset"] == dataset)
        ]
        if len(subset) <= 1:
            continue
        
        contrib = calculate_disparity_contribution(metrics_df, model, dataset)
        results.append(contrib)
        
        print(f"\n=== {dataset} - {model} ===")
        print(f"  View type range: {contrib['view_type_range']*100:.1f}%")
        print(f"  Age range: {contrib['age_group_range']*100:.1f}%")
        print(f"  Sex range: {contrib['sex_range']*100:.1f}%")
        print(f"  View dominance: {contrib['view_dominance']*100:.0f}%")
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_DIR / "performance_disparity_analysis.csv", index=False)
    
    # Summary statistics
    print("\n=== Summary Across All Models ===")
    for dataset in results_df["dataset"].unique():
        subset = results_df[results_df["dataset"] == dataset]
        print(f"\n{dataset}:")
        print(f"  Mean view dominance: {subset['view_dominance'].mean()*100:.0f}%")
        print(f"  Range: {subset['view_dominance'].min()*100:.0f}% - {subset['view_dominance'].max()*100:.0f}%")
    
    print(f"\nPerformance disparity analysis saved: {RESULTS_DIR / 'performance_disparity_analysis.csv'}")


if __name__ == "__main__":
    main()
