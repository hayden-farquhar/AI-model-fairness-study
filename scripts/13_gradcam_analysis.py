#!/usr/bin/env python3
"""
Grad-CAM saliency analysis: Compare AP vs PA attention patterns.

Generates saliency maps for a sample of AP and PA images to visualise
what image features the model attends to for each view type.

Addresses: Revision Issue #18 (R1 #2)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

import torch
import torch.nn.functional as F
import torchxrayvision as xrv
import pydicom

PROJECT = Path(__file__).resolve().parent.parent.parent
RSNA_IMAGES = Path("/Users/haydenfarquhar/My Drive/Research Data/rsna/stage_2_train_images")
PREDICTIONS = PROJECT / "04 Data" / "predictions" / "model_densenet121-all" / "rsna_predictions.csv"
VIEW_LOOKUP = PROJECT / "04 Data" / "view_type_lookup.csv"
OUTPUT = PROJECT / "07 Figures"


def load_dicom_as_tensor(dcm_path):
    """Load a DICOM file and preprocess for torchxrayvision."""
    ds = pydicom.dcmread(str(dcm_path))
    img = ds.pixel_array.astype(np.float32)

    # Normalise to [0, 1]
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    # torchxrayvision expects [-1024, 1024] range
    img = img * 2048 - 1024

    # Resize to 224x224
    from torchvision import transforms
    pil_img = Image.fromarray(((img + 1024) / 2048 * 255).astype(np.uint8))
    pil_img = pil_img.resize((224, 224), Image.LANCZOS)
    img_resized = np.array(pil_img).astype(np.float32) / 255.0 * 2048 - 1024

    tensor = torch.from_numpy(img_resized).unsqueeze(0).unsqueeze(0)  # [1, 1, 224, 224]
    return tensor, ds.pixel_array


class GradCAM:
    """Simple Grad-CAM implementation for DenseNet."""

    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None

        # Hook the last conv layer (features.denseblock4)
        target_layer = model.features.denseblock4
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_idx):
        """Generate Grad-CAM heatmap."""
        self.model.zero_grad()
        output = self.model(input_tensor)

        target = output[0, target_idx]
        target.backward()

        # Pool gradients
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        # Resize to input size
        cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
        cam = cam.squeeze().numpy()

        # Normalise
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam


def select_sample_images(n_per_group=5):
    """Select sample images: AP/PA × pneumonia+/pneumonia- × high/low prediction."""
    preds = pd.read_csv(PREDICTIONS)
    views = pd.read_csv(VIEW_LOOKUP)
    df = preds.merge(views, on="patientId")

    samples = {}
    for view_type in ["AP", "PA"]:
        for disease in [1, 0]:
            subset = df[(df["view_type"] == view_type) & (df["pneumonia"] == disease)]
            if disease == 1:
                # True positives (high prediction) and false negatives (low prediction)
                top = subset.nlargest(n_per_group, "prediction")
            else:
                # For disease-free, take range of predictions
                top = subset.sample(n=min(n_per_group, len(subset)), random_state=42)
            key = f"{view_type}_{'pos' if disease else 'neg'}"
            samples[key] = top
            print(f"  {key}: {len(top)} images selected")

    return samples


def main():
    print("Loading model...")
    model = xrv.models.DenseNet(weights="densenet121-res224-all")
    model.eval()

    pneumonia_idx = list(model.pathologies).index("Pneumonia")
    print(f"  Pneumonia index: {pneumonia_idx}")

    gradcam = GradCAM(model)

    print("\nSelecting sample images...")
    samples = select_sample_images(n_per_group=5)

    print("\nGenerating Grad-CAM heatmaps...")
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))

    row_labels = [
        "AP (pneumonia+)",
        "AP (disease-free)",
        "PA (pneumonia+)",
        "PA (disease-free)",
    ]
    row_keys = ["AP_pos", "AP_neg", "PA_pos", "PA_neg"]

    for row_idx, (label, key) in enumerate(zip(row_labels, row_keys)):
        df_subset = samples[key]
        for col_idx, (_, patient) in enumerate(df_subset.iterrows()):
            if col_idx >= 5:
                break

            ax = axes[row_idx, col_idx]
            dcm_path = RSNA_IMAGES / f"{patient['patientId']}.dcm"

            if not dcm_path.exists():
                ax.text(0.5, 0.5, "Image\nnot found", ha='center', va='center',
                        transform=ax.transAxes, fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            try:
                input_tensor, raw_img = load_dicom_as_tensor(dcm_path)

                with torch.enable_grad():
                    input_tensor.requires_grad_(True)
                    cam = gradcam.generate(input_tensor, pneumonia_idx)

                # Get prediction score
                with torch.no_grad():
                    pred = model(input_tensor)[0, pneumonia_idx].item()

                # Plot
                raw_resized = np.array(Image.fromarray(raw_img).resize((224, 224)))
                ax.imshow(raw_resized, cmap='gray', alpha=0.7)
                ax.imshow(cam, cmap='jet', alpha=0.3)
                ax.set_title(f"Score: {pred:.2f}", fontsize=9)
                ax.set_xticks([])
                ax.set_yticks([])
            except Exception as e:
                ax.text(0.5, 0.5, f"Error:\n{str(e)[:30]}", ha='center', va='center',
                        transform=ax.transAxes, fontsize=8)
                ax.set_xticks([])
                ax.set_yticks([])

        axes[row_idx, 0].set_ylabel(label, fontsize=12, fontweight='bold')

    fig.suptitle(
        "Grad-CAM Saliency Maps: AP vs PA View Type\n"
        "(DenseNet-All, Pneumonia Detection)",
        fontsize=14, fontweight='bold', y=0.98
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save as both PNG and TIF
    png_path = OUTPUT / "S13_Fig_GradCAM_Saliency.png"
    tif_path = OUTPUT / "S13_Fig_GradCAM_Saliency.tif"
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(tif_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nSaved: {png_path}")
    print(f"Saved: {tif_path}")


if __name__ == "__main__":
    main()
