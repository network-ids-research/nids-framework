# Objective 2: Model Training & Comparison

**Goal:** Train and compare 2–3 candidate models from literature to select the best suitable model for lightweight NIDS deployment.

## Candidate models

| Model | Hyperparameter tuning | Source in literature |
|-------|----------------------|---------------------|
| Random Forest | Optuna (30 trials) | Standard ensemble baseline |
| XGBoost | Optuna (30 trials) | State-of-the-art on UNSW-NB15 |
| Deep Learning | Early stopping | Optional, modern DNN baseline |

## Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Random Forest | 0.8693 | 0.8144 | 0.9878 | 0.8927 | 0.9811 |
| XGBoost | 0.8700 | 0.8180 | 0.9826 | **0.8928** | **0.9833** |

## Selection

**XGBoost is selected** as the best model because:
- Highest F1 (0.8928) and ROC-AUC (0.9833) — marginal but consistent
- **60× faster inference** than RF (1.1ms vs 68.3ms per sample)
- **55× smaller** model size (3MB vs 164MB)
- XGBoost is the most widely cited model in UNSW-NB15 literature

## Comparison with published results

| Study | Model | Accuracy | F1 | Notes |
|-------|-------|----------|----|-------|
| MDPI Computers (2026) | XGBoost | 0.97 | 0.96 | All 47 features, 5-fold CV |
| Ćirković & Milošević (2025) | XGBoost | 0.978 | 0.978 | Default params, all features |
| Springer (2024) | RF / XGBoost | ~0.93 | ~0.93 | 70/15/15 split |
| **This work** | **XGBoost** | **0.870** | **0.893** | **Top-25 features only** |

Our ~0.89 F1 is competitive given we use **only top-25 features** (vs all 47 in literature) and **no resampling** — reflecting real-world deployment constraints.

## Relevant code
- `src/models/random_forest.py`
- `src/models/xgboost_model.py`
- `src/models/deep_learning.py`
- `pipeline.py:109-170`
