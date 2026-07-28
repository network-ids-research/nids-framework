# Implementation Plan: Lightweight Explainable Network Anomaly Detection with Zero-Day Attack Evaluation

## Team Members
- Aditya Baheti (2360498)
- Travis Rego (2360475)
- Kuncham Yaswanth Reddy (2360406)
- Guide: Prof. Benedict Tephila / Dr. Gokulapriya R

---

## Timeline Overview (8 Weeks)

| Week | Phase | Deliverables |
|------|-------|-------------|
| 1 | Data Pipeline | Loaded & cleaned UNSW-NB15, correct train/test split |
| 2-3 | Feature Evaluation | Feature importance ranking, reduced feature set (~15-20) |
| 3-4 | Model Training | Trained RF, XGBoost, DL models with hyperparameter tuning |
| 5 | Zero-Day Simulation | Leave-one-attack-out results across all 9 attack categories |
| 6 | Explainability | SHAP analysis integrated for every alert |
| 7 | Benchmarking | Inference time & memory usage measurements |
| 8 | Documentation & Report | Final report, presentation, code cleanup |

---

## Task Breakdown

### Phase 1: Data Pipeline (Week 1) — Aditya

| Task | Description | Output |
|------|-------------|--------|
| 1.1 | Download UNSW-NB15 dataset (CSV files) | Raw data in `data/raw/` |
| 1.2 | Parse and merge training + test partitions | Unified DataFrame |
| 1.3 | Handle missing values (drop/median-impute) | Clean DataFrame |
| 1.4 | Encode categorical fields (proto, service, state) | Numeric DataFrame |
| 1.5 | Correct train/test split BEFORE balancing | `X_train, X_test, y_train, y_test` |
| 1.6 | Exploratory data analysis (class distribution, correlations) | `notebooks/01_data_exploration.ipynb` |

### Phase 2: Feature Evaluation (Week 2-3) — Yaswanth

| Task | Description | Output |
|------|-------------|--------|
| 2.1 | Train baseline RF on all ~49 features | Full-feature accuracy baseline |
| 2.2 | Extract feature importance (Gini importance / permutation) | Feature importance ranking |
| 2.3 | Select top-N features (try N = 15, 20, 25) | Reduced feature subsets |
| 2.4 | Compare accuracy vs. number of features | Feature reduction report |
| 2.5 | Finalize optimized feature set | `features/selected_features.json` |

### Phase 3: Model Training (Week 3-4) — Travis

| Task | Description | Output |
|------|-------------|--------|
| 3.1 | Train Random Forest with Optuna tuning | `models/rf_best.pkl` |
| 3.2 | Train XGBoost with Optuna tuning | `models/xgb_best.pkl` |
| 3.3 | Train Deep Learning (MLP/Autoencoder) with TensorFlow | `models/dl_best.keras` |
| 3.4 | Evaluate all models: accuracy, precision, recall, F1, confusion matrix | `results/tables/model_comparison.csv` |
| 3.5 | Cross-validation on training set | Reproducible scores |

### Phase 4: Zero-Day Simulation (Week 5) — Aditya + Yaswanth

| Task | Description | Output |
|------|-------------|--------|
| 4.1 | List all 9 attack categories in UNSW-NB15 | Attack category mapping |
| 4.2 | Implement leave-one-attack-out loop | Iterative training script |
| 4.3 | For each held-out attack: train on others, test on held-out | Detection rate per attack type |
| 4.4 | Aggregate results across all 9 iterations | `results/tables/zero_day_results.csv` |
| 4.5 | Compare zero-day detection across RF, XGBoost, DL | Best model for unseen attacks |

### Phase 5: Explainability (Week 6) — Travis

| Task | Description | Output |
|------|-------------|--------|
| 5.1 | Implement SHAP explainer wrapper | `src/explainability/shap_explainer.py` |
| 5.2 | Generate SHAP summary plots (global importance) | `results/figures/shap_summary.png` |
| 5.3 | Generate SHAP waterfall plots (per-sample explanation) | `results/figures/shap_waterfall/` |
| 5.4 | Build "Explain Alert" function | Returns top-3 features for any prediction |
| 5.5 | Test on flagged malicious traffic | Sample explanations in notebook |

### Phase 6: Benchmarking (Week 7) — All

| Task | Description | Output |
|------|-------------|--------|
| 6.1 | Measure inference time (per-sample, batch) | `results/tables/inference_time.csv` |
| 6.2 | Measure memory usage (model size, RAM at inference) | `results/tables/memory_usage.csv` |
| 6.3 | Compare lightweight model vs. full-feature baseline | Speed vs. accuracy trade-off table |

### Phase 7: Documentation & Integration (Week 8) — All

| Task | Description | Output |
|------|-------------|--------|
| 7.1 | Integrate all modules into pipeline script | `pipeline.py` |
| 7.2 | Run end-to-end pipeline | Complete results |
| 7.3 | Write final report | `docs/report/` |
| 7.4 | Prepare presentation | Review-1 slides |

---

## Architecture Overview

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────┐
│ UNSW-NB15   │───▶│ Data Loader  │───▶│ Feature     │───▶│ Model        │
│ (CSV Files) │    │ & Cleaner    │    │ Selector    │    │ Trainer      │
└─────────────┘    └──────────────┘    └─────────────┘    └──────┬───────┘
                                                                 │
                         ┌───────────────────────────────────────┘
                         ▼
              ┌─────────────────────┐
              │   Model Evaluation  │
              │ (Zero-Day + Normal) │
              └─────────┬───────────┘
                        │
              ┌─────────▼───────────┐
              │  SHAP Explainability│
              │  + Benchmarking     │
              └─────────────────────┘
```

## Dependencies

- Python 3.10+
- pandas, numpy, scikit-learn
- xgboost, lightgbm (optional)
- tensorflow or pytorch (for DL model)
- shap, matplotlib, seaborn
- optuna (hyperparameter tuning)
- psutil (memory benchmarking)

## Risk Mitigation

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| UNSW-NB15 download failure | Low | Mirrors available; use cached version |
| DL model too slow on CPU | Medium | Use simpler MLP; fallback to sklearn MLP |
| SHAP slow on full test set | Medium | Use KernelSHAP on sample subset |
| Memory exhaustion | Low | Process in chunks; use data generators |

---

## Communication Plan

- Weekly sync meetings (Wednesdays)
- Shared WhatsApp group for daily updates
- GitHub for code version control
- Each member pushes working code by end of their assigned week
- Peer review before merging to main
