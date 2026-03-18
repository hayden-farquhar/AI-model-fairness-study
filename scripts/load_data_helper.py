#!/usr/bin/env python3
"""
Shared data loader for revision analyses.
Loads all 5 model prediction CSVs and merges with view type lookup.
"""

import pandas as pd
from pathlib import Path

# When run from repo: data/ and results/ are siblings of scripts/
# Override BASE with environment variable DATA_DIR if needed
import os
BASE = Path(os.environ.get("DATA_DIR", Path(__file__).resolve().parent.parent / "data"))
PREDICTIONS = BASE / "predictions"
VIEW_LOOKUP = BASE / "view_type_lookup.csv"

MODELS = {
    "densenet121-all": "model_densenet121-all/rsna_predictions.csv",
    "densenet121_rsna": "model_densenet121_res224_rsna/predictions.csv",
    "densenet121_nih": "model_densenet121_res224_nih/predictions.csv",
    "densenet121_chex": "model_densenet121_res224_chex/predictions.csv",
    "padchest": "model_padchest/predictions.csv",
}

TRAINING_SOURCES = {
    "densenet121-all": ["rsna", "nih", "chexpert", "mimic"],
    "densenet121_rsna": ["rsna"],
    "densenet121_nih": ["nih"],
    "densenet121_chex": ["chexpert"],
    "padchest": ["padchest"],
}


def load_view_lookup():
    return pd.read_csv(VIEW_LOOKUP)


def load_model(model_name):
    """Load a single model's predictions with view type merged in."""
    path = PREDICTIONS / MODELS[model_name]
    df = pd.read_csv(path)

    if "view" in df.columns and "view_type" not in df.columns:
        df = df.rename(columns={"view": "view_type"})

    if "view_type" not in df.columns:
        view = load_view_lookup()
        df = df.merge(view, on="patientId", how="left")

    if "predicted_class" not in df.columns:
        df["predicted_class"] = (df["prediction"] >= 0.5).astype(int)

    df["model"] = model_name
    return df


def load_all_models():
    """Load all 5 models into a single DataFrame."""
    return pd.concat([load_model(name) for name in MODELS], ignore_index=True)


def compute_optimal_threshold(df):
    """Compute Youden's J optimal threshold."""
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(df["pneumonia"], df["prediction"])
    j = tpr - fpr
    return thresholds[j.argmax()]
