#!/usr/bin/env python3
"""
07_generate_figures.py

Generate publication-quality figures for the manuscript.

Author: Hayden Farquhar, MBBS MPHTM
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configuration
RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

# Style settings
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'sans-serif',
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
})


def figure1_variance_decomposition():
    """Figure 1: Variance decomposition showing view type dominance."""
    
    var_df = pd.read_csv(RESULTS_DIR / "variance_decomposition.csv")
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    for i, dataset in enumerate(["RSNA", "NIH"]):
        ax = axes[i]
        subset = var_df[var_df["dataset"] == dataset].copy()
        
        if len(subset) == 0:
            ax.text(0.5, 0.5, f"No data for {dataset}", ha='center', va='center')
            continue
        
        # Stacked bar chart
        models = subset["model"].values
        view_contrib = subset["view_type_contribution"].values * 100
        age_contrib = subset["age_group_contribution"].values * 100
        sex_contrib = subset["sex_contribution"].values * 100
        
        x = np.arange(len(models))
        width = 0.6
        
        ax.bar(x, view_contrib, width, label='View Type', color='#2ecc71')
        ax.bar(x, age_contrib, width, bottom=view_contrib, label='Age', color='#3498db')
        ax.bar(x, sex_contrib, width, bottom=view_contrib+age_contrib, label='Sex', color='#9b59b6')
        
        ax.set_ylabel('Variance Contribution (%)')
        ax.set_title(f'{dataset} Dataset')
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('densenet-', '') for m in models], rotation=45, ha='right')
        ax.set_ylim(0, 105)
        ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
        
        if i == 0:
            ax.legend(loc='upper right')
    
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "figure1_variance_decomposition.png", dpi=300, bbox_inches='tight')
    fig.savefig(FIGURES_DIR / "figure1_variance_decomposition.pdf", bbox_inches='tight')
    plt.close()
    
    print("Figure 1: Variance decomposition saved")


def figure2_view_gap():
    """Figure 2: View type sensitivity gap across models."""
    
    metrics_df = pd.read_csv(RESULTS_DIR / "performance_metrics.csv")
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    for i, dataset in enumerate(["RSNA", "NIH"]):
        ax = axes[i]
        
        subset = metrics_df[
            (metrics_df["dataset"] == dataset) & 
            (metrics_df["group_type"] == "view_type")
        ].copy()
        
        if len(subset) == 0:
            ax.text(0.5, 0.5, f"No data for {dataset}", ha='center', va='center')
            continue
        
        models = subset["model"].unique()
        
        ap_sens = []
        pa_sens = []
        
        for model in models:
            model_data = subset[subset["model"] == model]
            ap_row = model_data[model_data["subgroup"] == "AP"]
            pa_row = model_data[model_data["subgroup"] == "PA"]
            
            if len(ap_row) > 0 and len(pa_row) > 0:
                ap_sens.append(ap_row["sensitivity"].values[0] * 100)
                pa_sens.append(pa_row["sensitivity"].values[0] * 100)
            else:
                ap_sens.append(0)
                pa_sens.append(0)
        
        x = np.arange(len(models))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, ap_sens, width, label='AP Views', color='#e74c3c')
        bars2 = ax.bar(x + width/2, pa_sens, width, label='PA Views', color='#3498db')
        
        ax.set_ylabel('Sensitivity (%)')
        ax.set_title(f'{dataset} Dataset')
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('densenet-', '') for m in models], rotation=45, ha='right')
        ax.set_ylim(0, 100)
        ax.legend()
        
        # Add gap annotations
        for j, (ap, pa) in enumerate(zip(ap_sens, pa_sens)):
            gap = ap - pa
            ax.annotate(f'{gap:.0f}%', xy=(j, max(ap, pa) + 2), ha='center', fontsize=8)
    
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "figure2_view_gap.png", dpi=300, bbox_inches='tight')
    fig.savefig(FIGURES_DIR / "figure2_view_gap.pdf", bbox_inches='tight')
    plt.close()
    
    print("Figure 2: View gap saved")


def figure3_true_negative_analysis():
    """Figure 3: True negative analysis showing severity confounding refutation."""
    
    tn_df = pd.read_csv(RESULTS_DIR / "true_negative_analysis.csv")
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Group by dataset
    datasets = tn_df["dataset"].unique()
    colors = {'RSNA': '#e74c3c', 'NIH': '#3498db'}
    
    for dataset in datasets:
        subset = tn_df[tn_df["dataset"] == dataset]
        models = subset["model"].values
        cohens_d = subset["cohens_d"].values
        
        x_offset = -0.2 if dataset == "RSNA" else 0.2
        x = np.arange(len(models)) + x_offset
        
        ax.bar(x, cohens_d, 0.35, label=dataset, color=colors.get(dataset, 'gray'))
    
    # Effect size thresholds
    ax.axhline(y=0.2, color='green', linestyle='--', alpha=0.5, label='Small (0.2)')
    ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Medium (0.5)')
    ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='Large (0.8)')
    
    ax.set_ylabel("Cohen's d (AP vs PA in True Negatives)")
    ax.set_xlabel("Model")
    ax.set_xticks(np.arange(len(models)))
    ax.set_xticklabels([m.replace('densenet-', '') for m in models], rotation=45, ha='right')
    ax.legend(loc='upper right')
    ax.set_title("True Negative Analysis: Severity Confounding Test")
    
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "figure3_true_negative.png", dpi=300, bbox_inches='tight')
    fig.savefig(FIGURES_DIR / "figure3_true_negative.pdf", bbox_inches='tight')
    plt.close()
    
    print("Figure 3: True negative analysis saved")


def main():
    """Generate all figures."""
    
    print("Generating publication figures...")
    
    try:
        figure1_variance_decomposition()
    except Exception as e:
        print(f"Figure 1 error: {e}")
    
    try:
        figure2_view_gap()
    except Exception as e:
        print(f"Figure 2 error: {e}")
    
    try:
        figure3_true_negative_analysis()
    except Exception as e:
        print(f"Figure 3 error: {e}")
    
    print(f"\nFigures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
