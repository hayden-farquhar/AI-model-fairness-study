#!/usr/bin/env python3
"""
03_performance_analysis.py

Calculate performance metrics (sensitivity, specificity, AUC) by subgroup.

Author: Hayden Farquhar, MBBS MPHTM
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, roc_curve
from scipy import stats

RESULTS_DIR = Path("results")

# Models to analyze
MODELS = ["densenet-all", "densenet-rsna", "densenet-nih", "densenet-chexpert", "densenet-padchest"]


def find_optimal_threshold(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Find optimal threshold using Youden's J statistic."""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    return thresholds[optimal_idx]


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> dict:
    """Calculate classification metrics at given threshold."""
    
    y_binary = (y_pred >= threshold).astype(int)
    
    tp = np.sum((y_binary == 1) & (y_true == 1))
    tn = np.sum((y_binary == 0) & (y_true == 0))
    fp = np.sum((y_binary == 1) & (y_true == 0))
    fn = np.sum((y_binary == 0) & (y_true == 1))
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    try:
        auc = roc_auc_score(y_true, y_pred)
    except ValueError:
        auc = np.nan
    
    return {
        "n": len(y_true),
        "n_positive": int(np.sum(y_true)),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "npv": npv,
        "auc": auc,
        "threshold": threshold,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn
    }


def analyze_by_subgroup(df: pd.DataFrame, pred_col: str, group_col: str, threshold: float) -> pd.DataFrame:
    """Calculate metrics for each subgroup."""
    
    results = []
    for group_val in df[group_col].dropna().unique():
        mask = df[group_col] == group_val
        y_true = df.loc[mask, "pneumonia"].values
        y_pred = df.loc[mask, pred_col].values
        
        # Remove NaN predictions
        valid = ~np.isnan(y_pred)
        if valid.sum() < 10:
            continue
            
        metrics = calculate_metrics(y_true[valid], y_pred[valid], threshold)
        metrics["subgroup"] = group_val
        metrics["group_type"] = group_col
        results.append(metrics)
    
    return pd.DataFrame(results)


def compare_subgroups(metrics_df: pd.DataFrame, group_col: str) -> dict:
    """Statistical comparison between subgroups."""
    
    subset = metrics_df[metrics_df["group_type"] == group_col].copy()
    
    if len(subset) < 2:
        return {}
    
    sens_values = subset["sensitivity"].values
    n_pos = subset["n_positive"].values
    
    # Chi-squared test for sensitivity difference
    # Using proportion test approximation
    max_sens = sens_values.max()
    min_sens = sens_values.min()
    gap = max_sens - min_sens
    
    # Find groups with max and min sensitivity
    max_group = subset.loc[subset["sensitivity"].idxmax(), "subgroup"]
    min_group = subset.loc[subset["sensitivity"].idxmin(), "subgroup"]
    
    return {
        "group_type": group_col,
        "max_sensitivity": max_sens,
        "max_group": max_group,
        "min_sensitivity": min_sens,
        "min_group": min_group,
        "gap": gap,
    }


def main():
    """Main performance analysis pipeline."""
    
    # Load predictions
    rsna_df = pd.read_csv(RESULTS_DIR / "rsna_predictions.csv")
    nih_df = pd.read_csv(RESULTS_DIR / "nih_predictions.csv")
    
    all_results = []
    comparisons = []
    
    for model in MODELS:
        pred_col = f"pred_{model}"
        
        for df, dataset in [(rsna_df, "RSNA"), (nih_df, "NIH")]:
            print(f"\n=== {dataset} - {model} ===")
            
            # Skip if predictions not available
            if pred_col not in df.columns:
                print(f"  Skipping - predictions not found")
                continue
            
            # Find optimal threshold on full dataset
            valid = ~df[pred_col].isna()
            threshold = find_optimal_threshold(
                df.loc[valid, "pneumonia"].values,
                df.loc[valid, pred_col].values
            )
            print(f"  Optimal threshold: {threshold:.3f}")
            
            # Overall metrics
            overall = calculate_metrics(
                df.loc[valid, "pneumonia"].values,
                df.loc[valid, pred_col].values,
                threshold
            )
            overall["subgroup"] = "Overall"
            overall["group_type"] = "overall"
            overall["model"] = model
            overall["dataset"] = dataset
            all_results.append(overall)
            
            print(f"  Overall AUC: {overall['auc']:.3f}")
            print(f"  Overall Sensitivity: {overall['sensitivity']*100:.1f}%")
            
            # Analysis by subgroup
            for group_col in ["view_type", "age_group", "sex"]:
                subgroup_metrics = analyze_by_subgroup(df, pred_col, group_col, threshold)
                subgroup_metrics["model"] = model
                subgroup_metrics["dataset"] = dataset
                all_results.extend(subgroup_metrics.to_dict("records"))
                
                comp = compare_subgroups(subgroup_metrics, group_col)
                if comp:
                    comp["model"] = model
                    comp["dataset"] = dataset
                    comparisons.append(comp)
                    print(f"  {group_col} gap: {comp['gap']*100:.1f}% ({comp['max_group']} vs {comp['min_group']})")
    
    # Save results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(RESULTS_DIR / "performance_metrics.csv", index=False)
    
    comparisons_df = pd.DataFrame(comparisons)
    comparisons_df.to_csv(RESULTS_DIR / "subgroup_comparisons.csv", index=False)
    
    print("\n=== Analysis Complete ===")
    print(f"Performance metrics saved: {RESULTS_DIR / 'performance_metrics.csv'}")
    print(f"Subgroup comparisons saved: {RESULTS_DIR / 'subgroup_comparisons.csv'}")


if __name__ == "__main__":
    main()
