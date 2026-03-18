#!/usr/bin/env python3
"""
Expanded Cohen's d effect sizes for all primary comparisons.

Computes Cohen's d on continuous prediction scores for:
- AP vs PA views
- Male vs Female
- Each age group vs reference (youngest)

Addresses: Revision Issue #14 (R2 #4)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from load_data_helper import load_model, compute_optimal_threshold, MODELS

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def cohens_d(group1, group2):
    """Compute Cohen's d between two groups."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return np.nan
    var1, var2 = group1.var(ddof=1), group2.var(ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return np.nan
    return (group1.mean() - group2.mean()) / pooled_std


def interpret_d(d):
    """Interpret Cohen's d magnitude."""
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"


def main():
    all_results = []

    for model_name in MODELS:
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")

        df = load_model(model_name)

        # --- View type: AP vs PA ---
        ap = df[df["view_type"] == "AP"]["prediction"]
        pa = df[df["view_type"] == "PA"]["prediction"]
        d = cohens_d(ap, pa)
        all_results.append({
            "model": model_name, "comparison": "AP vs PA",
            "factor": "view_type", "group1": "AP", "group2": "PA",
            "mean1": ap.mean(), "mean2": pa.mean(),
            "n1": len(ap), "n2": len(pa),
            "cohens_d": d, "interpretation": interpret_d(d),
        })
        print(f"  AP vs PA: d={d:.3f} ({interpret_d(d)})")

        # --- Sex: M vs F ---
        m = df[df["sex"] == "M"]["prediction"]
        f = df[df["sex"] == "F"]["prediction"]
        d = cohens_d(m, f)
        all_results.append({
            "model": model_name, "comparison": "Male vs Female",
            "factor": "sex", "group1": "M", "group2": "F",
            "mean1": m.mean(), "mean2": f.mean(),
            "n1": len(m), "n2": len(f),
            "cohens_d": d, "interpretation": interpret_d(d),
        })
        print(f"  Male vs Female: d={d:.3f} ({interpret_d(d)})")

        # --- Age groups: each vs youngest (<40) ---
        age_order = ["<40", "40-59", "60-79", "≥80"]
        age_groups = [ag for ag in age_order if ag in df["age_group"].values]
        ref_group = age_groups[0]  # <40
        ref_preds = df[df["age_group"] == ref_group]["prediction"]

        for ag in age_groups[1:]:
            ag_preds = df[df["age_group"] == ag]["prediction"]
            d = cohens_d(ag_preds, ref_preds)
            all_results.append({
                "model": model_name, "comparison": f"{ag} vs {ref_group}",
                "factor": "age_group", "group1": ag, "group2": ref_group,
                "mean1": ag_preds.mean(), "mean2": ref_preds.mean(),
                "n1": len(ag_preds), "n2": len(ref_preds),
                "cohens_d": d, "interpretation": interpret_d(d),
            })
            print(f"  {ag} vs {ref_group}: d={d:.3f} ({interpret_d(d)})")

        # --- Also compute for disease-positive subset only (sensitivity-relevant) ---
        pos = df[df["pneumonia"] == 1]
        ap_pos = pos[pos["view_type"] == "AP"]["prediction"]
        pa_pos = pos[pos["view_type"] == "PA"]["prediction"]
        d = cohens_d(ap_pos, pa_pos)
        all_results.append({
            "model": model_name, "comparison": "AP vs PA (pneumonia+)",
            "factor": "view_type_positive", "group1": "AP", "group2": "PA",
            "mean1": ap_pos.mean(), "mean2": pa_pos.mean(),
            "n1": len(ap_pos), "n2": len(pa_pos),
            "cohens_d": d, "interpretation": interpret_d(d),
        })
        print(f"  AP vs PA (pneumonia+): d={d:.3f} ({interpret_d(d)})")

        # --- Disease-free subset (true negative analysis) ---
        neg = df[df["pneumonia"] == 0]
        ap_neg = neg[neg["view_type"] == "AP"]["prediction"]
        pa_neg = neg[neg["view_type"] == "PA"]["prediction"]
        d = cohens_d(ap_neg, pa_neg)
        all_results.append({
            "model": model_name, "comparison": "AP vs PA (pneumonia-)",
            "factor": "view_type_negative", "group1": "AP", "group2": "PA",
            "mean1": ap_neg.mean(), "mean2": pa_neg.mean(),
            "n1": len(ap_neg), "n2": len(pa_neg),
            "cohens_d": d, "interpretation": interpret_d(d),
        })
        print(f"  AP vs PA (pneumonia-): d={d:.3f} ({interpret_d(d)})")

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(RESULTS_DIR / "cohens_d_all_comparisons.csv", index=False)
    print(f"\nSaved to {RESULTS_DIR / 'cohens_d_all_comparisons.csv'}")

    # Summary table
    print("\n\n=== SUMMARY: Mean |Cohen's d| across models ===")
    summary = results_df.groupby("comparison")["cohens_d"].agg(["mean", "std"]).round(3)
    print(summary)


if __name__ == "__main__":
    main()
