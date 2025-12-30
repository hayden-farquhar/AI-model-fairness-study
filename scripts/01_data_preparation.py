#!/usr/bin/env python3
"""
01_data_preparation.py

Load and preprocess RSNA and NIH datasets for analysis.
Extracts DICOM metadata including view type, age, and sex.

Author: Hayden Farquhar, MBBS MPHTM
"""

import os
import pandas as pd
import pydicom
from pathlib import Path
from tqdm import tqdm

# Configuration
DATA_DIR = Path("data")
RSNA_DIR = DATA_DIR / "rsna"
NIH_DIR = DATA_DIR / "nih"
OUTPUT_DIR = Path("results")


def load_rsna_data(rsna_dir: Path) -> pd.DataFrame:
    """Load RSNA Pneumonia Detection Challenge data with DICOM metadata."""
    
    # Load labels
    labels_path = rsna_dir / "stage_2_train_labels.csv"
    labels = pd.read_csv(labels_path)
    
    # Aggregate to patient level (some patients have multiple boxes)
    patient_labels = labels.groupby("patientId").agg({
        "Target": "max"  # 1 if any pneumonia annotation
    }).reset_index()
    
    # Extract DICOM metadata
    dicom_dir = rsna_dir / "stage_2_train_images"
    metadata = []
    
    for patient_id in tqdm(patient_labels["patientId"], desc="Loading RSNA DICOM"):
        dcm_path = dicom_dir / f"{patient_id}.dcm"
        if dcm_path.exists():
            dcm = pydicom.dcmread(dcm_path)
            metadata.append({
                "patientId": patient_id,
                "age": int(dcm.PatientAge.replace("Y", "")) if hasattr(dcm, "PatientAge") else None,
                "sex": dcm.PatientSex if hasattr(dcm, "PatientSex") else None,
                "view": dcm.ViewPosition if hasattr(dcm, "ViewPosition") else None,
            })
    
    metadata_df = pd.DataFrame(metadata)
    rsna_df = patient_labels.merge(metadata_df, on="patientId")
    
    # Standardize column names
    rsna_df = rsna_df.rename(columns={
        "Target": "pneumonia",
        "view": "view_type"
    })
    
    # Create age groups
    rsna_df["age_group"] = pd.cut(
        rsna_df["age"],
        bins=[0, 40, 60, 80, 100],
        labels=["<40", "40-59", "60-79", "≥80"],
        right=False
    )
    
    rsna_df["dataset"] = "RSNA"
    return rsna_df


def load_nih_data(nih_dir: Path) -> pd.DataFrame:
    """Load NIH ChestX-ray14 data."""
    
    # Load data entry file
    data_entry = pd.read_csv(nih_dir / "Data_Entry_2017_v2020.csv")
    
    # Extract pneumonia label
    data_entry["pneumonia"] = data_entry["Finding Labels"].str.contains("Pneumonia").astype(int)
    
    # Parse view position
    data_entry["view_type"] = data_entry["View Position"]
    
    # Parse age (already numeric in NIH)
    data_entry["age"] = data_entry["Patient Age"]
    
    # Create age groups
    data_entry["age_group"] = pd.cut(
        data_entry["age"],
        bins=[0, 40, 60, 80, 100],
        labels=["<40", "40-59", "60-79", "≥80"],
        right=False
    )
    
    # Standardize sex
    data_entry["sex"] = data_entry["Patient Gender"]
    
    # Select relevant columns
    nih_df = data_entry[[
        "Image Index", "pneumonia", "view_type", "age", "age_group", "sex"
    ]].copy()
    nih_df = nih_df.rename(columns={"Image Index": "image_id"})
    nih_df["dataset"] = "NIH"
    
    return nih_df


def main():
    """Main data preparation pipeline."""
    
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Load datasets
    print("Loading RSNA dataset...")
    rsna_df = load_rsna_data(RSNA_DIR)
    print(f"  RSNA: {len(rsna_df)} images, {rsna_df['pneumonia'].mean()*100:.1f}% pneumonia")
    
    print("Loading NIH dataset...")
    nih_df = load_nih_data(NIH_DIR)
    print(f"  NIH: {len(nih_df)} images, {nih_df['pneumonia'].mean()*100:.1f}% pneumonia")
    
    # Save processed data
    rsna_df.to_csv(OUTPUT_DIR / "rsna_processed.csv", index=False)
    nih_df.to_csv(OUTPUT_DIR / "nih_processed.csv", index=False)
    
    # Print summary statistics
    print("\n=== Dataset Summary ===")
    for df, name in [(rsna_df, "RSNA"), (nih_df, "NIH")]:
        print(f"\n{name}:")
        print(f"  Total images: {len(df):,}")
        print(f"  View distribution: {df['view_type'].value_counts().to_dict()}")
        print(f"  Sex distribution: {df['sex'].value_counts().to_dict()}")
        print(f"  Age groups: {df['age_group'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
