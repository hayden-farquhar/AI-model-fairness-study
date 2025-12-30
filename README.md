# Technical Acquisition Parameters Dominate Demographic Factors in Chest X-ray AI Performance Disparities

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository contains the analysis code for the study: **"Technical Acquisition Parameters Dominate Demographic Factors in Chest X-ray AI Performance Disparities: Evidence from Two Independent Datasets"**

**Author:** Hayden Farquhar, MBBS MPHTM  
**Contact:** hayden.farquhar@icloud.com

## Abstract

We conducted a multi-dataset external validation study analyzing 138,804 chest radiographs from two independent sources (RSNA Pneumonia Detection Challenge and NIH ChestX-ray14) to examine performance disparities in chest X-ray AI systems. View type dominated performance variance in both datasets (87% in RSNA, 69% in NIH), substantially exceeding demographic factors. All models demonstrated PA view underdiagnosis with miss rates of 30-78%. True negative analysis definitively refuted severity confounding.

## Repository Structure

```
cxr-ai-fairness/
├── scripts/
│   ├── 01_data_preparation.py      # Load and preprocess datasets
│   ├── 02_model_inference.py       # Run inference with torchxrayvision models
│   ├── 03_performance_analysis.py  # Calculate sensitivity, specificity, AUC
│   ├── 04_variance_decomposition.py# Quantify factor contributions
│   ├── 05_statistical_validation.py# Bootstrap CIs, permutation tests
│   ├── 06_true_negative_analysis.py# Disease-free subgroup analysis
│   └── 07_generate_figures.py      # Create publication figures
├── data/
│   └── .gitkeep                    # Data not included (see Data Access)
├── figures/
│   └── .gitkeep                    # Generated figures
├── results/
│   └── .gitkeep                    # Analysis outputs
├── requirements.txt
├── LICENSE
└── README.md
```

## Requirements

```
torch>=1.12.0
torchxrayvision>=0.0.45
torchvision>=0.13.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scipy>=1.10.0
statsmodels>=0.14.0
pydicom>=2.4.0
tqdm>=4.65.0
```

## Data Access

The datasets used in this study are publicly available:

1. **RSNA Pneumonia Detection Challenge** (n=26,684)
   - Download: https://www.kaggle.com/c/rsna-pneumonia-detection-challenge
   - Extract to `data/rsna/`

2. **NIH ChestX-ray14** (n=112,120)
   - Download: https://nihcc.app.box.com/v/ChestXray-NIHCC
   - Extract to `data/nih/`

## Models

We evaluated five pre-trained DenseNet-121 models from the [torchxrayvision](https://github.com/mlmed/torchxrayvision) library:

- DenseNet-All (combined training)
- DenseNet-RSNA
- DenseNet-NIH
- DenseNet-CheXpert
- DenseNet-PadChest

## Quick Start

```bash
# Clone repository
git clone https://github.com/[username]/cxr-ai-fairness.git
cd cxr-ai-fairness

# Install dependencies
pip install -r requirements.txt

# Download and prepare data (see Data Access section)

# Run analysis pipeline
python scripts/01_data_preparation.py
python scripts/02_model_inference.py
python scripts/03_performance_analysis.py
python scripts/04_variance_decomposition.py
python scripts/05_statistical_validation.py
python scripts/06_true_negative_analysis.py
python scripts/07_generate_figures.py
```

## Key Findings

| Dataset | View Dominance | PA Miss Rate | OR (PA vs AP) |
|---------|---------------|--------------|---------------|
| RSNA    | 87%           | 43%          | 6.69          |
| NIH     | 69%           | 78%          | 13.02         |

- View type explained 69-87% of performance variance vs 1-30% for demographics
- True negative analysis (n=131,361) showed large effect sizes (d=1.19-1.33), refuting severity confounding
- 100% replication across 10 model-dataset combinations

## Citation

If you use this code, please cite:

```
Farquhar H. Technical Acquisition Parameters Dominate Demographic Factors 
in Chest X-ray AI Performance Disparities: Evidence from Two Independent 
Datasets. [Journal]. [Year];[Volume]:[Pages].
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [torchxrayvision](https://github.com/mlmed/torchxrayvision) for pre-trained models
- RSNA and NIH for making datasets publicly available
