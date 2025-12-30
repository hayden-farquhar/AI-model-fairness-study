#!/usr/bin/env python3
"""
05_statistical_validation.py

Statistical validation: bootstrap confidence intervals, permutation tests,
cross-validation, and multiple testing correction.

Author: Hayden Farquhar, MBBS MPHTM
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

RESULTS_DIR = Path("results")
N_BOOTSTRAP = 1000
N_PERMUTATIONS = 1000
RANDOM_SEED = 42


def bootstrap_sensitivity(y_true: np.ndarray, y_pred: np.ndarray, threshold: float, 
                          n_bootstrap: int = N_BOOTSTRAP) -> tuple:
    """Calculate bootstrap confidence interval for sensitivity."""
    
    np.random.seed(RANDOM_SEED)
    sensitivities = []
    
    for _ in range(n_bootstrap):
        idx = np.random.choice(len(y_true), size=len(y_true), replace=True)
        y_true_boot = y_true[idx]
        y_pred_boot = y_pred[idx]
        
        # Only calculate if we have positive cases
        if y_true_boot.sum() > 0:
            y_binary = (y_pred_boot >= threshold).astype(int)
            tp = np.sum((y_binary == 1) & (y_true_boot == 1))
            fn = np.sum((y_binary == 0) & (y_true_boot == 1))
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0
            sensitivities.append(sens)
    
    sensitivities = np.array(sensitivities)
    ci_lower = np.percentile(sensitivities, 2.5)
    ci_upper = np.percentile(sensitivities, 97.5)
    
    return np.mean(sensitivities), ci_lower, ci_upper


def bootstrap_gap(y_true_a: np.ndarray, y_pred_a: np.ndarray,
                  y_true_b: np.ndarray, y_pred_b: np.ndarray,
                  threshold: float, n_bootstrap: int = N_BOOTSTRAP) -> tuple:
    """Calculate bootstrap CI for sensitivity gap between groups."""
    
    np.random.seed(RANDOM_SEED)
    gaps = []
    
    for _ in range(n_bootstrap):
        # Bootstrap group A
        idx_a = np.random.choice(len(y_true_a), size=len(y_true_a), replace=True)
        y_true_a_boot = y_true_a[idx_a]
        y_pred_a_boot = y_pred_a[idx_a]
        
        # Bootstrap group B
        idx_b = np.random.choice(len(y_true_b), size=len(y_true_b), replace=True)
        y_true_b_boot = y_true_b[idx_b]
        y_pred_b_boot = y_pred_b[idx_b]
        
        # Calculate sensitivities
        if y_true_a_boot.sum() > 0 and y_true_b_boot.sum() > 0:
            sens_a = np.sum((y_pred_a_boot >= threshold) & (y_true_a_boot == 1)) / y_true_a_boot.sum()
            sens_b = np.sum((y_pred_b_boot >= threshold) & (y_true_b_boot == 1)) / y_true_b_boot.sum()
            gaps.append(sens_a - sens_b)
    
    gaps = np.array(gaps)
    return np.mean(gaps), np.percentile(gaps, 2.5), np.percentile(gaps, 97.5)


def permutation_test(y_true: np.ndarray, y_pred: np.ndarray, group: np.ndarray,
                     threshold: float, n_permutations: int = N_PERMUTATIONS) -> float:
    """Permutation test for sensitivity difference between groups."""
    
    np.random.seed(RANDOM_SEED)
    
    # Observed difference
    mask_a = group == group.unique()[0]
    sens_a = np.sum((y_pred[mask_a] >= threshold) & (y_true[mask_a] == 1)) / y_true[mask_a].sum()
    sens_b = np.sum((y_pred[~mask_a] >= threshold) & (y_true[~mask_a] == 1)) / y_true[~mask_a].sum()
    observed_diff = abs(sens_a - sens_b)
    
    # Permutation distribution
    perm_diffs = []
    for _ in range(n_permutations):
        perm_group = np.random.permutation(group)
        mask_perm = perm_group == group.unique()[0]
        
        if y_true[mask_perm].sum() > 0 and y_true[~mask_perm].sum() > 0:
            sens_perm_a = np.sum((y_pred[mask_perm] >= threshold) & (y_true[mask_perm] == 1)) / y_true[mask_perm].sum()
            sens_perm_b = np.sum((y_pred[~mask_perm] >= threshold) & (y_true[~mask_perm] == 1)) / y_true[~mask_perm].sum()
            perm_diffs.append(abs(sens_perm_a - sens_perm_b))
    
    p_value = np.mean(np.array(perm_diffs) >= observed_diff)
    return p_value


def cohens_d(group_a: np.ndarray, group_b: np.ndarray) -> float:
    """Calculate Cohen's d effect size."""
    n1, n2 = len(group_a), len(group_b)
    var1, var2 = group_a.var(), group_b.var()
    
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    
    if pooled_std == 0:
        return 0
    
    return (group_a.mean() - group_b.mean()) / pooled_std


def multiple_testing_correction(p_values: list, method: str = "bonferroni") -> list:
    """Apply multiple testing correction."""
    
    n_tests = len(p_values)
    
    if method == "bonferroni":
        return [min(p * n_tests, 1.0) for p in p_values]
    elif method == "bh":
        # Benjamini-Hochberg FDR
        sorted_idx = np.argsort(p_values)
        sorted_p = np.array(p_values)[sorted_idx]
        adjusted = np.zeros(n_tests)
        
        for i, p in enumerate(sorted_p):
            adjusted[sorted_idx[i]] = p * n_tests / (i + 1)
        
        # Ensure monotonicity
        adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
        return np.minimum(adjusted, 1.0).tolist()
    
    return p_values


def main():
    """Main statistical validation pipeline."""
    
    # Load predictions
    rsna_df = pd.read_csv(RESULTS_DIR / "rsna_predictions.csv")
    nih_df = pd.read_csv(RESULTS_DIR / "nih_predictions.csv")
    
    # Load performance metrics for thresholds
    metrics_df = pd.read_csv(RESULTS_DIR / "performance_metrics.csv")
    
    results = []
    all_p_values = []
    
    for df, dataset in [(rsna_df, "RSNA"), (nih_df, "NIH")]:
        pred_col = "pred_densenet-all"
        
        if pred_col not in df.columns:
            continue
        
        # Get threshold
        threshold_row = metrics_df[
            (metrics_df["model"] == "densenet-all") & 
            (metrics_df["dataset"] == dataset) &
            (metrics_df["subgroup"] == "Overall")
        ]
        
        if len(threshold_row) == 0:
            continue
            
        threshold = threshold_row["threshold"].values[0]
        
        valid = ~df[pred_col].isna()
        y_true = df.loc[valid, "pneumonia"].values
        y_pred = df.loc[valid, pred_col].values
        
        print(f"\n=== {dataset} Statistical Validation ===")
        
        # View type analysis
        view_valid = valid & df["view_type"].notna()
        
        # AP group
        ap_mask = (df["view_type"] == "AP") & view_valid
        pa_mask = (df["view_type"] == "PA") & view_valid
        
        if ap_mask.sum() > 0 and pa_mask.sum() > 0:
            # Bootstrap CI for view gap
            gap_mean, gap_lower, gap_upper = bootstrap_gap(
                df.loc[ap_mask, "pneumonia"].values,
                df.loc[ap_mask, pred_col].values,
                df.loc[pa_mask, "pneumonia"].values,
                df.loc[pa_mask, pred_col].values,
                threshold
            )
            
            print(f"View gap: {gap_mean*100:.1f}% [95% CI: {gap_lower*100:.1f}%, {gap_upper*100:.1f}%]")
            
            # Cohen's d for prediction scores
            d = cohens_d(
                df.loc[ap_mask, pred_col].values,
                df.loc[pa_mask, pred_col].values
            )
            print(f"Cohen's d (prediction scores): {d:.2f}")
            
            results.append({
                "dataset": dataset,
                "comparison": "AP vs PA",
                "gap": gap_mean,
                "ci_lower": gap_lower,
                "ci_upper": gap_upper,
                "cohens_d": d
            })
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_DIR / "statistical_validation.csv", index=False)
    
    print(f"\nStatistical validation saved: {RESULTS_DIR / 'statistical_validation.csv'}")


if __name__ == "__main__":
    main()
