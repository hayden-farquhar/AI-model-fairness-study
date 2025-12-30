#!/usr/bin/env python3
"""
02_model_inference.py

Run inference using torchxrayvision pre-trained models on RSNA and NIH datasets.

Author: Hayden Farquhar, MBBS MPHTM
"""

import torch
import torchxrayvision as xrv
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import pydicom
from skimage import exposure

# Configuration
DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Models to evaluate
MODELS = {
    "densenet-all": "densenet121-res224-all",
    "densenet-rsna": "densenet121-res224-rsna", 
    "densenet-nih": "densenet121-res224-nih",
    "densenet-chexpert": "densenet121-res224-chex",
    "densenet-padchest": "densenet121-res224-pc",
}


def load_model(model_name: str) -> torch.nn.Module:
    """Load a torchxrayvision model."""
    model = xrv.models.DenseNet(weights=model_name)
    model = model.to(DEVICE)
    model.eval()
    return model


def preprocess_image(image_path: Path, is_dicom: bool = True) -> torch.Tensor:
    """Preprocess image for torchxrayvision models."""
    
    if is_dicom:
        dcm = pydicom.dcmread(image_path)
        img = dcm.pixel_array.astype(np.float32)
    else:
        img = np.array(Image.open(image_path).convert("L")).astype(np.float32)
    
    # Normalize to [0, 1]
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    
    # Apply torchxrayvision preprocessing
    img = xrv.datasets.normalize(img, 255)
    
    # Resize to 224x224
    from skimage.transform import resize
    img = resize(img, (224, 224), preserve_range=True)
    
    # Add channel and batch dimensions
    img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()
    
    return img.to(DEVICE)


def get_pneumonia_prediction(model: torch.nn.Module, img: torch.Tensor, model_name: str) -> float:
    """Get pneumonia prediction score from model."""
    
    with torch.no_grad():
        outputs = model(img)
    
    # Get pathology names for the model
    pathologies = model.pathologies
    
    # Find pneumonia index (or lung opacity as proxy)
    if "Pneumonia" in pathologies:
        idx = pathologies.index("Pneumonia")
    elif "Lung Opacity" in pathologies:
        idx = pathologies.index("Lung Opacity")
    else:
        raise ValueError(f"No pneumonia-related output in model {model_name}")
    
    return outputs[0, idx].item()


def run_inference_rsna(model: torch.nn.Module, model_name: str, df: pd.DataFrame) -> pd.DataFrame:
    """Run inference on RSNA dataset."""
    
    predictions = []
    image_dir = DATA_DIR / "rsna" / "stage_2_train_images"
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"RSNA - {model_name}"):
        img_path = image_dir / f"{row['patientId']}.dcm"
        if img_path.exists():
            img = preprocess_image(img_path, is_dicom=True)
            pred = get_pneumonia_prediction(model, img, model_name)
            predictions.append(pred)
        else:
            predictions.append(np.nan)
    
    df[f"pred_{model_name}"] = predictions
    return df


def run_inference_nih(model: torch.nn.Module, model_name: str, df: pd.DataFrame) -> pd.DataFrame:
    """Run inference on NIH dataset."""
    
    predictions = []
    image_dirs = [DATA_DIR / "nih" / f"images_{i:03d}" for i in range(1, 13)]
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"NIH - {model_name}"):
        img_found = False
        for img_dir in image_dirs:
            img_path = img_dir / "images" / row["image_id"]
            if img_path.exists():
                img = preprocess_image(img_path, is_dicom=False)
                pred = get_pneumonia_prediction(model, img, model_name)
                predictions.append(pred)
                img_found = True
                break
        
        if not img_found:
            predictions.append(np.nan)
    
    df[f"pred_{model_name}"] = predictions
    return df


def main():
    """Main inference pipeline."""
    
    print(f"Using device: {DEVICE}")
    
    # Load processed data
    rsna_df = pd.read_csv(RESULTS_DIR / "rsna_processed.csv")
    nih_df = pd.read_csv(RESULTS_DIR / "nih_processed.csv")
    
    # Run inference for each model
    for model_name, weights in MODELS.items():
        print(f"\n=== Loading {model_name} ===")
        model = load_model(weights)
        
        print("Running RSNA inference...")
        rsna_df = run_inference_rsna(model, model_name, rsna_df)
        
        print("Running NIH inference...")
        nih_df = run_inference_nih(model, model_name, nih_df)
        
        # Clear GPU memory
        del model
        torch.cuda.empty_cache()
    
    # Save predictions
    rsna_df.to_csv(RESULTS_DIR / "rsna_predictions.csv", index=False)
    nih_df.to_csv(RESULTS_DIR / "nih_predictions.csv", index=False)
    
    print("\n=== Inference Complete ===")
    print(f"RSNA predictions saved: {RESULTS_DIR / 'rsna_predictions.csv'}")
    print(f"NIH predictions saved: {RESULTS_DIR / 'nih_predictions.csv'}")


if __name__ == "__main__":
    main()
