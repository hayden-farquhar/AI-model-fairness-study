#!/usr/bin/env python3
"""
Intersectional analysis: sensitivity across age × sex × view type combinations.

Addresses: Revision Issue #7 (R3 Specific #5)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from load_data_helper import load_model, compute_optimal_threshold, MODELS

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def compute_intersectional_sensitivity(df, threshold):
    """Compute sensitivity for every age_group × sex × view_type combination."""
    df = df.copy()
    df["predicted_class"] = (df["prediction"] >= threshold).astype(int)

    positives = df[df["pneumonia"] == 1]

    results = []
    for (ag, sex, vt), group in positives.groupby(["age_group", "sex", "view_type"]):
        if len(group) < 10:  # Skip tiny groups
            continue
        sens = group["predicted_class"].mean()
        results.append({
            "age_group": ag,
            "sex": sex,
            "view_type": vt,
            "sensitivity": sens,
            "n_positive": len(group),
            "n_detected": group["predicted_class"].sum(),
        })

    return pd.DataFrame(results)


def bootstrap_cumulative_disparity(df, threshold, n_boot=1000, seed=42):
    """Bootstrap CI for cumulative disparity (max - min intersectional sensitivity)."""
    rng = np.random.RandomState(seed)
    df = df.copy()
    df["predicted_class"] = (df["prediction"] >= threshold).astype(int)
    positives = df[df["pneumonia"] == 1]

    disparities = []
    for _ in range(n_boot):
        boot = positives.sample(n=len(positives), replace=True, random_state=rng)
        sensitivities = []
        for (ag, sex, vt), group in boot.groupby(["age_group", "sex", "view_type"]):
            if len(group) >= 10:
                sensitivities.append(group["predicted_class"].mean())
        if len(sensitivities) >= 2:
            disparities.append(max(sensitivities) - min(sensitivities))

    return np.percentile(disparities, [2.5, 97.5]) if disparities else (np.nan, np.nan)


def main():
    all_results = []
    summary_results = []

    for model_name in MODELS:
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")

        df = load_model(model_name)
        threshold = compute_optimal_threshold(df)

        inter = compute_intersectional_sensitivity(df, threshold)
        inter["model"] = model_name
        all_results.append(inter)

        if len(inter) >= 2:
            max_row = inter.loc[inter["sensitivity"].idxmax()]
            min_row = inter.loc[inter["sensitivity"].idxmin()]
            disparity = max_row["sensitivity"] - min_row["sensitivity"]

            ci_low, ci_high = bootstrap_cumulative_disparity(df, threshold)

            summary_results.append({
                "model": model_name,
                "max_subgroup": f"{max_row['age_group']}/{max_row['sex']}/{max_row['view_type']}",
                "max_sensitivity": max_row["sensitivity"],
                "min_subgroup": f"{min_row['age_group']}/{min_row['sex']}/{min_row['view_type']}",
                "min_sensitivity": min_row["sensitivity"],
                "cumulative_disparity": disparity,
                "disparity_ci_low": ci_low,
                "disparity_ci_high": ci_high,
            })

            print(f"  Best subgroup: {max_row['age_group']}/{max_row['sex']}/{max_row['view_type']} "
                  f"= {max_row['sensitivity']*100:.1f}%")
            print(f"  Worst subgroup: {min_row['age_group']}/{min_row['sex']}/{min_row['view_type']} "
                  f"= {min_row['sensitivity']*100:.1f}%")
            print(f"  Cumulative disparity: {disparity*100:.1f}% "
                  f"[95% CI: {ci_low*100:.1f}%-{ci_high*100:.1f}%]")

    # Save
    pd.concat(all_results).to_csv(RESULTS_DIR / "intersectional_sensitivity.csv", index=False)
    pd.DataFrame(summary_results).to_csv(RESULTS_DIR / "intersectional_summary.csv", index=False)

    print(f"\nSaved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
