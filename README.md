# Lightweight Explainable Network Anomaly Detection Framework with Zero-Day Attack Evaluation

Project Work Phase –1 (CSE 784) — Christ University

## Team

| Sl.No | Student Name | RegNo | Class | Guide Name |
|-------|-------------|-------|-------|------------|
| 1 | Aditya Baheti | 2360498 | 7BTCS-C | Prof. Benedict Tephila |
| 2 | Travis Rego | 2360475 | 7BTCS-C | Prof. Benedict Tephila |
| 3 | Kuncham Yaswanth Reddy | 2360406 | 7BTCS-C | Dr. Gokulapriya R |

## Project Overview

This project builds a **lightweight, explainable network anomaly detection framework** using the UNSW-NB15 dataset. It addresses four key gaps in existing ML-based intrusion detection systems:

1. **No zero-day testing** — Existing models are not tested against unseen attack types
2. **Data leakage** — Improper train/test splitting inflates accuracy metrics
3. **Computationally heavy** — Models are too large for real-time deployment
4. **No explainability** — No mechanism to explain why traffic was flagged

### Key Features

- Leakage-free train/test split
- Feature reduction from ~49 to ~15-20 features
- Model comparison: Random Forest, XGBoost, Deep Learning (MLP)
- Leave-one-attack-out zero-day simulation
- SHAP-based explainability for every alert
- Inference time and memory benchmarking

## Project Structure

```
├── config/                  # Configuration files
│   └── config.yaml
├── data/                    # Dataset storage
│   ├── raw/                 # Place UNSW-NB15 CSV files here
│   ├── processed/           # Processed data artifacts
│   └── external/            # External resources
├── src/                     # Source code
│   ├── data/                # Data loading, cleaning, splitting
│   ├── features/            # Feature engineering & selection
│   ├── models/              # RF, XGBoost, Deep Learning
│   ├── evaluation/          # Metrics, zero-day, benchmarking
│   ├── explainability/       # SHAP explainer
│   └── visualization/       # Plotting utilities
├── notebooks/               # Jupyter notebooks for exploration
├── tests/                   # Unit tests
├── results/                 # Output artifacts
│   ├── models/              # Trained model files
│   ├── figures/             # Generated plots
│   └── tables/              # CSV result tables
├── docs/                    # Documentation & reports
├── pipeline.py              # End-to-end pipeline script
├── requirements.txt         # Python dependencies
└── implementation_plan.md   # Detailed implementation plan
```

## Setup

### Prerequisites
- Python 3.10+
- pip

### Installation

```bash
pip install -r requirements.txt
```

### Dataset

1. Download UNSW-NB15 from: https://research.unsw.edu.au/projects/unsw-nb15-dataset
2. Place the CSV files in `data/raw/`:
   - `UNSW_NB15_training-set.csv`
   - `UNSW_NB15_testing-set.csv`

## Usage

### Full Pipeline

```bash
python pipeline.py
```

### Selective Execution

```bash
python pipeline.py --skip-zero-day    # Skip zero-day simulation (faster)
python pipeline.py --skip-benchmark   # Skip benchmarking
python pipeline.py --skip-dl          # Skip deep learning (CPU-only)
```

### Jupyter Notebooks

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

## Expected Outcomes

1. Working leakage-free baseline pipeline with honest metrics
2. Model comparison table (RF, XGBoost, DL)
3. Validated lightweight feature set (15-20 features)
4. Quantified zero-day detection rate across all 9 attack categories
5. Benchmarked inference time and memory usage
6. Explainable alert system with SHAP
