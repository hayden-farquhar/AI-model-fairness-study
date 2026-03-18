#!/usr/bin/env python3
"""
Formal ANOVA-style decomposition of performance disparities.

Uses logistic regression on per-patient correct/incorrect classification
to compute the proportion of deviance explained by view type, age group, and sex.
Also computes traditional one-way ANOVA on prediction scores.

Addresses: Revision Issues #2, #19 (R1 #1, R2 #2/#5, R3 Specific #2)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import logit, ols

sys.path.insert(0, str(Path(__file__).parent))
from load_data_helper import load_model, compute_optimal_threshold, MODELS

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def compute_anova_on_predictions(df):
    """One-way ANOVA on continuous prediction scores by each factor."""
    results = []

    for factor, col in [("view_type", "view_type"), ("age_group", "age_group"), ("sex", "sex")]:
        groups = [g["prediction"].values for _, g in df.groupby(col)]
        if len(groups) < 2:
            continue
        f_stat, p_val = stats.f_oneway(*groups)

        # Eta-squared
        grand_mean = df["prediction"].mean()
        ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
        ss_total = ((df["prediction"] - grand_mean) ** 2).sum()
        eta_sq = ss_between / ss_total if ss_total > 0 else 0

        results.append({
            "factor": factor,
            "F_statistic": f_stat,
            "p_value": p_val,
            "eta_squared": eta_sq,
            "n_groups": len(groups),
        })

    return pd.DataFrame(results)


def compute_deviance_decomposition(df):
    """Logistic regression decomposition of correct classification."""
    # Binary: was the prediction correct?
    df = df.copy()
    df["correct"] = (df["predicted_class"] == df["pneumonia"]).astype(int)

    results = []

    # Null model — use llf (log-likelihood) to compute deviance: -2 * llf
    null_model = logit("correct ~ 1", data=df).fit(disp=0)
    null_deviance = -2 * null_model.llf

    # Single-factor models
    for factor in ["view_type", "age_group", "sex"]:
        try:
            model = logit(f"correct ~ C({factor})", data=df).fit(disp=0)
            model_deviance = -2 * model.llf
            deviance_explained = null_deviance - model_deviance
            prop_explained = deviance_explained / null_deviance if null_deviance > 0 else 0
            results.append({
                "factor": factor,
                "null_deviance": null_deviance,
                "model_deviance": model_deviance,
                "deviance_explained": deviance_explained,
                "proportion_deviance_explained": prop_explained,
                "aic": model.aic,
                "bic": model.bic,
            })
        except Exception as e:
            print(f"  Warning: logistic regression failed for {factor}: {e}")

    # Full model
    try:
        full_model = logit("correct ~ C(view_type) + C(age_group) + C(sex)", data=df).fit(disp=0)
        full_deviance = -2 * full_model.llf
        full_deviance_explained = null_deviance - full_deviance
        results.append({
            "factor": "all_combined",
            "null_deviance": null_deviance,
            "model_deviance": full_deviance,
            "deviance_explained": full_deviance_explained,
            "proportion_deviance_explained": full_deviance_explained / null_deviance if null_deviance > 0 else 0,
            "aic": full_model.aic,
            "bic": full_model.bic,
        })
    except Exception as e:
        print(f"  Warning: full logistic regression failed: {e}")

    return pd.DataFrame(results)


def compute_sensitivity_ranges(df):
    """Compute sensitivity range for each factor (the original 'variance decomposition' approach)."""
    results = []

    for factor in ["view_type", "age_group", "sex"]:
        sensitivities = []
        for level, group in df[df["pneumonia"] == 1].groupby(factor):
            sens = group["predicted_class"].mean()
            sensitivities.append({"level": level, "sensitivity": sens, "n": len(group)})

        if len(sensitivities) >= 2:
            sens_vals = [s["sensitivity"] for s in sensitivities]
            results.append({
                "factor": factor,
                "sensitivity_range": max(sens_vals) - min(sens_vals),
                "max_sensitivity": max(sens_vals),
                "min_sensitivity": min(sens_vals),
                "max_level": sensitivities[np.argmax(sens_vals)]["level"],
                "min_level": sensitivities[np.argmin(sens_vals)]["level"],
            })

    total_range = sum(r["sensitivity_range"] for r in results)
    for r in results:
        r["relative_contribution"] = r["sensitivity_range"] / total_range if total_range > 0 else 0

    return pd.DataFrame(results)


def main():
    all_anova = []
    all_deviance = []
    all_ranges = []

    for model_name in MODELS:
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")

        df = load_model(model_name)

        # Compute optimal threshold for this model
        threshold = compute_optimal_threshold(df)
        df["predicted_class"] = (df["prediction"] >= threshold).astype(int)
        print(f"  Optimal threshold: {threshold:.3f}")

        # ANOVA on prediction scores
        anova = compute_anova_on_predictions(df)
        anova["model"] = model_name
        all_anova.append(anova)
        print(f"\n  ANOVA on prediction scores:")
        for _, row in anova.iterrows():
            print(f"    {row['factor']}: F={row['F_statistic']:.1f}, p={row['p_value']:.2e}, η²={row['eta_squared']:.4f}")

        # Deviance decomposition
        deviance = compute_deviance_decomposition(df)
        deviance["model"] = model_name
        all_deviance.append(deviance)
        print(f"\n  Deviance decomposition (logistic regression):")
        for _, row in deviance.iterrows():
            print(f"    {row['factor']}: deviance explained={row['deviance_explained']:.1f}, "
                  f"proportion={row['proportion_deviance_explained']:.4f}")

        # Sensitivity ranges
        ranges = compute_sensitivity_ranges(df)
        ranges["model"] = model_name
        all_ranges.append(ranges)
        print(f"\n  Sensitivity ranges:")
        for _, row in ranges.iterrows():
            print(f"    {row['factor']}: range={row['sensitivity_range']*100:.1f}%, "
                  f"relative contribution={row['relative_contribution']*100:.0f}%")

    # Save results
    pd.concat(all_anova).to_csv(RESULTS_DIR / "anova_prediction_scores.csv", index=False)
    pd.concat(all_deviance).to_csv(RESULTS_DIR / "deviance_decomposition.csv", index=False)
    pd.concat(all_ranges).to_csv(RESULTS_DIR / "sensitivity_ranges.csv", index=False)

    print(f"\n\nResults saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
