#!/usr/bin/env python3
"""
Generate Table 1 demographics stratified by AP/PA view type.

Addresses: Revision Issue #16 (Editor #4)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from load_data_helper import load_model, MODELS

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def compute_demographics(df, label="All"):
    """Compute demographic summary statistics for a subset."""
    return {
        "subset": label,
        "n": len(df),
        "age_mean": df["age"].mean(),
        "age_sd": df["age"].std(),
        "age_median": df["age"].median(),
        "pct_male": (df["sex"] == "M").mean() * 100,
        "pct_female": (df["sex"] == "F").mean() * 100,
        "pneumonia_n": df["pneumonia"].sum(),
        "pneumonia_pct": df["pneumonia"].mean() * 100,
        "age_lt40_n": (df["age_group"] == "<40").sum(),
        "age_lt40_pct": (df["age_group"] == "<40").mean() * 100,
        "age_40_59_n": (df["age_group"] == "40-59").sum(),
        "age_40_59_pct": (df["age_group"] == "40-59").mean() * 100,
        "age_60_79_n": (df["age_group"] == "60-79").sum(),
        "age_60_79_pct": (df["age_group"] == "60-79").mean() * 100,
        "age_ge80_n": (df["age_group"] == "≥80").sum(),
        "age_ge80_pct": (df["age_group"] == "≥80").mean() * 100,
    }


def main():
    # Use any model (same patients across all) - use densenet121-all as reference
    df = load_model("densenet121-all")

    results = []

    # Overall
    results.append(compute_demographics(df, "Total"))

    # By view type
    for vt in ["AP", "PA"]:
        subset = df[df["view_type"] == vt]
        results.append(compute_demographics(subset, vt))

    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_DIR / "table1_by_view_type.csv", index=False)

    # Print formatted table
    print("=" * 80)
    print("TABLE 1: Dataset characteristics stratified by view type")
    print("=" * 80)
    print(f"\n{'Characteristic':<25} {'Total':>12} {'AP':>12} {'PA':>12}")
    print("-" * 65)

    for _, row in results_df.iterrows():
        if row["subset"] == "Total":
            continue

    total = results_df[results_df["subset"] == "Total"].iloc[0]
    ap = results_df[results_df["subset"] == "AP"].iloc[0]
    pa = results_df[results_df["subset"] == "PA"].iloc[0]

    print(f"{'N':<25} {int(total['n']):>12,} {int(ap['n']):>12,} {int(pa['n']):>12,}")
    print(f"{'Age, mean (SD)':<25} {total['age_mean']:.1f} ({total['age_sd']:.1f})   {ap['age_mean']:.1f} ({ap['age_sd']:.1f})   {pa['age_mean']:.1f} ({pa['age_sd']:.1f})")
    print(f"{'Male, n (%)':<25} {int(total['n']*total['pct_male']/100):>6,} ({total['pct_male']:.1f}%) {int(ap['n']*ap['pct_male']/100):>6,} ({ap['pct_male']:.1f}%) {int(pa['n']*pa['pct_male']/100):>6,} ({pa['pct_male']:.1f}%)")
    print(f"{'Pneumonia, n (%)':<25} {int(total['pneumonia_n']):>6,} ({total['pneumonia_pct']:.1f}%) {int(ap['pneumonia_n']):>6,} ({ap['pneumonia_pct']:.1f}%) {int(pa['pneumonia_n']):>6,} ({pa['pneumonia_pct']:.1f}%)")

    print(f"\n{'Age group':<25}")
    for label, n_col, pct_col in [("<40", "age_lt40_n", "age_lt40_pct"),
                                    ("40-59", "age_40_59_n", "age_40_59_pct"),
                                    ("60-79", "age_60_79_n", "age_60_79_pct"),
                                    ("≥80", "age_ge80_n", "age_ge80_pct")]:
        print(f"  {label:<23} {int(total[n_col]):>6,} ({total[pct_col]:.1f}%) {int(ap[n_col]):>6,} ({ap[pct_col]:.1f}%) {int(pa[n_col]):>6,} ({pa[pct_col]:.1f}%)")

    print(f"\nSaved to {RESULTS_DIR / 'table1_by_view_type.csv'}")


if __name__ == "__main__":
    main()
