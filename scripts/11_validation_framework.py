#!/usr/bin/env python3
"""
Validation framework: classify each model-dataset pair as Internal/External.
Recompute view dominance statistics segregated by validation type.

Addresses: Revision Issue #1 (R3 Major #1, Editor #2)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from load_data_helper import load_model, compute_optimal_threshold, MODELS, TRAINING_SOURCES

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# All predictions are on RSNA test data
DATASET = "RSNA"


def classify_validation(model_name, dataset):
    """Classify a model-dataset pair as Internal, External, or Semi-internal."""
    sources = TRAINING_SOURCES[model_name]
    dataset_lower = dataset.lower()
    if dataset_lower in sources and len(sources) == 1:
        return "Internal"
    elif dataset_lower in sources and len(sources) > 1:
        return "Semi-internal"
    else:
        return "External"


def compute_view_dominance(df):
    """Compute view type sensitivity gap and dominance metrics."""
    threshold = compute_optimal_threshold(df)
    df = df.copy()
    df["predicted_class"] = (df["prediction"] >= threshold).astype(int)

    positives = df[df["pneumonia"] == 1]

    metrics = {}
    for factor in ["view_type", "age_group", "sex"]:
        sensitivities = {}
        for level, group in positives.groupby(factor):
            sensitivities[level] = group["predicted_class"].mean()
        vals = list(sensitivities.values())
        metrics[f"{factor}_range"] = max(vals) - min(vals) if len(vals) >= 2 else 0
        metrics[f"{factor}_details"] = sensitivities

    total = sum(metrics[f"{f}_range"] for f in ["view_type", "age_group", "sex"])
    metrics["view_dominance"] = metrics["view_type_range"] / total if total > 0 else 0
    metrics["threshold"] = threshold
    return metrics


def main():
    results = []

    for model_name in MODELS:
        df = load_model(model_name)
        val_type = classify_validation(model_name, DATASET)
        metrics = compute_view_dominance(df)

        vt_details = metrics["view_type_details"]
        ap_sens = vt_details.get("AP", np.nan)
        pa_sens = vt_details.get("PA", np.nan)

        result = {
            "model": model_name,
            "dataset": DATASET,
            "validation_type": val_type,
            "threshold": metrics["threshold"],
            "AP_sensitivity": ap_sens,
            "PA_sensitivity": pa_sens,
            "view_gap": ap_sens - pa_sens if not np.isnan(ap_sens) else np.nan,
            "view_type_range": metrics["view_type_range"],
            "age_group_range": metrics["age_group_range"],
            "sex_range": metrics["sex_range"],
            "view_dominance": metrics["view_dominance"],
        }
        results.append(result)

        print(f"{model_name} ({val_type}): "
              f"AP sens={ap_sens:.3f}, PA sens={pa_sens:.3f}, "
              f"gap={ap_sens-pa_sens:.3f}, view dominance={metrics['view_dominance']*100:.0f}%")

    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_DIR / "validation_framework.csv", index=False)

    # Summary by validation type
    print("\n=== Summary by Validation Type ===")
    for vt in ["External", "Semi-internal", "Internal"]:
        subset = results_df[results_df["validation_type"] == vt]
        if len(subset) == 0:
            continue
        print(f"\n{vt} (n={len(subset)}):")
        print(f"  Mean view gap: {subset['view_gap'].mean()*100:.1f}%")
        print(f"  Mean view dominance: {subset['view_dominance'].mean()*100:.0f}%")
        print(f"  Range of view gaps: {subset['view_gap'].min()*100:.1f}% - {subset['view_gap'].max()*100:.1f}%")

    print(f"\nKey finding: View type dominance holds across external validation "
          f"(mean={results_df[results_df['validation_type']=='External']['view_dominance'].mean()*100:.0f}%)")
    print(f"\nSaved to {RESULTS_DIR / 'validation_framework.csv'}")


if __name__ == "__main__":
    main()
