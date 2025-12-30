#!/usr/bin/env python3
"""
06_true_negative_analysis.py

Analyze prediction score distributions in disease-free (true negative) images
to test the severity confounding hypothesis.

If models give higher scores to AP views because AP patients are sicker,
then among disease-free images (where there's no disease to be sicker with),
prediction scores should be similar across view types.

Author: Hayden Farquhar, MBBS MPHTM
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

RESULTS_DIR = Path("results")


def cohens_d(group_a: np.ndarray, group_b: np.ndarray) -> float:
    """Calculate Cohen's d effect size."""
    n1, n2 = len(group_a), len(group_b)
    var1, var2 = group_a.var(), group_b.var()
    
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    
    if pooled_std == 0:
        return 0
    
    return (group_a.mean() - group_b.mean()) / pooled_std


def analyze_true_negatives(df: pd.DataFrame, pred_col: str, dataset: str) -> dict:
    """
    Analyze prediction scores among disease-free images by view type.
    
    Key insight: If AP views get higher scores due to patient severity,
    this effect should disappear when we only look at patients without disease.
    Large effect sizes among true negatives refute the severity hypothesis.
    """
    
    # Filter to disease-free images
    tn_mask = df["pneumonia"] == 0
    valid = tn_mask & df[pred_col].notna() & df["view_type"].notna()
    
    tn_df = df[valid].copy()
    
    n_total = len(tn_df)
    n_ap = (tn_df["view_type"] == "AP").sum()
    n_pa = (tn_df["view_type"] == "PA").sum()
    
    print(f"\n  True negatives: {n_total:,} (AP: {n_ap:,}, PA: {n_pa:,})")
    
    # Get prediction scores by view type
    ap_scores = tn_df.loc[tn_df["view_type"] == "AP", pred_col].values
    pa_scores = tn_df.loc[tn_df["view_type"] == "PA", pred_col].values
    
    # Calculate effect size
    d = cohens_d(ap_scores, pa_scores)
    
    # Independent samples t-test
    t_stat, p_value = stats.ttest_ind(ap_scores, pa_scores)
    
    # Mann-Whitney U test (non-parametric)
    u_stat, u_pvalue = stats.mannwhitneyu(ap_scores, pa_scores, alternative='two-sided')
    
    # Summary statistics
    ap_mean = np.mean(ap_scores)
    pa_mean = np.mean(pa_scores)
    ap_std = np.std(ap_scores)
    pa_std = np.std(pa_scores)
    
    print(f"  AP mean score: {ap_mean:.3f} (SD: {ap_std:.3f})")
    print(f"  PA mean score: {pa_mean:.3f} (SD: {pa_std:.3f})")
    print(f"  Cohen's d: {d:.2f}")
    print(f"  t-test p-value: {p_value:.2e}")
    
    # Interpret effect size
    if abs(d) < 0.2:
        interpretation = "negligible"
    elif abs(d) < 0.5:
        interpretation = "small"
    elif abs(d) < 0.8:
        interpretation = "medium"
    else:
        interpretation = "LARGE"
    
    print(f"  Effect size interpretation: {interpretation}")
    
    result = {
        "dataset": dataset,
        "model": pred_col.replace("pred_", ""),
        "n_true_negatives": n_total,
        "n_ap": n_ap,
        "n_pa": n_pa,
        "ap_mean_score": ap_mean,
        "pa_mean_score": pa_mean,
        "ap_std": ap_std,
        "pa_std": pa_std,
        "score_difference": ap_mean - pa_mean,
        "cohens_d": d,
        "effect_interpretation": interpretation,
        "t_statistic": t_stat,
        "t_pvalue": p_value,
        "u_statistic": u_stat,
        "u_pvalue": u_pvalue,
    }
    
    # Key conclusion for severity hypothesis
    if abs(d) >= 0.5:
        result["severity_hypothesis_refuted"] = True
        print(f"  ** SEVERITY CONFOUNDING REFUTED: Large effect persists without disease **")
    else:
        result["severity_hypothesis_refuted"] = False
    
    return result


def main():
    """Main true negative analysis pipeline."""
    
    # Load predictions
    rsna_df = pd.read_csv(RESULTS_DIR / "rsna_predictions.csv")
    nih_df = pd.read_csv(RESULTS_DIR / "nih_predictions.csv")
    
    models = ["densenet-all", "densenet-rsna", "densenet-nih", "densenet-chexpert", "densenet-padchest"]
    
    results = []
    
    for df, dataset in [(rsna_df, "RSNA"), (nih_df, "NIH")]:
        print(f"\n{'='*50}")
        print(f"TRUE NEGATIVE ANALYSIS: {dataset}")
        print(f"{'='*50}")
        
        for model in models:
            pred_col = f"pred_{model}"
            
            if pred_col not in df.columns:
                continue
            
            print(f"\nModel: {model}")
            result = analyze_true_negatives(df, pred_col, dataset)
            results.append(result)
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_DIR / "true_negative_analysis.csv", index=False)
    
    # Summary
    print(f"\n{'='*50}")
    print("SUMMARY: Severity Confounding Hypothesis Test")
    print(f"{'='*50}")
    
    for dataset in ["RSNA", "NIH"]:
        subset = results_df[results_df["dataset"] == dataset]
        if len(subset) == 0:
            continue
        
        refuted = subset["severity_hypothesis_refuted"].sum()
        total = len(subset)
        mean_d = subset["cohens_d"].mean()
        
        print(f"\n{dataset}:")
        print(f"  Models with large effect in true negatives: {refuted}/{total}")
        print(f"  Mean Cohen's d: {mean_d:.2f}")
        
        if mean_d >= 0.5:
            print(f"  CONCLUSION: Severity confounding hypothesis REFUTED")
            print(f"              AP views receive higher scores even without disease")
        else:
            print(f"  CONCLUSION: Severity confounding cannot be ruled out")
    
    print(f"\nTrue negative analysis saved: {RESULTS_DIR / 'true_negative_analysis.csv'}")


if __name__ == "__main__":
    main()
