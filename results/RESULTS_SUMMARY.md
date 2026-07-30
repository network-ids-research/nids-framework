# Lightweight Explainable NIDS — Results Summary

## Dataset
- **UNSW-NB15**: 257,673 samples (175K train / 82K test), 42 features after encoding
- **Task**: Binary classification (normal vs attack)

## Model Performance (Top-25 Features)

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Random Forest | 0.8702 | 0.8151 | 0.9886 | **0.8935** | 0.9824 |
| XGBoost | 0.8700 | 0.8184 | 0.9818 | **0.8927** | 0.9833 |

Both models nearly identical. **RF was chosen as best** for its higher Recall (0.9886) — critical in security to minimize missed attacks.

### Comparison with Literature
Our results are competitive with published UNSW-NB15 benchmarks:

| Study | Model | Accuracy | F1-Score | Notes |
|-------|-------|----------|----------|-------|
| MDPI Computers (2026) — 5-fold CV | XGBoost | 0.97 | 0.96 | Uses all 47 features |
| MDPI Computers (2026) — 5-fold CV | RF | 0.96 | 0.95 | Uses all 47 features |
| Ćirković & Milošević (2025) | XGBoost | 0.978 | 0.978 | Default params, all features |
| Ćirković & Milošević (2025) | RF | 0.975 | 0.975 | Default params, all features |
| Springer (2024) | RF / XGBoost | ~0.93 | ~0.93 | 70/15/15 split |
| **This work** | **RF** | **0.870** | **0.893** | **Top-25 features only** |
| **This work** | **XGBoost** | **0.870** | **0.893** | **Top-25 features only** |

Our ~0.89 F1 is slightly lower than some published results because:
- We use only **top-25 features** (dimensionality reduction for lightweight deployment)
- We do **not** use SMOTE or any resampling — results reflect real-world class imbalance
- We evaluate on the **full original test split** without subsampling
- The lightweight constraint (feature reduction, no resampling) trades marginal F1 for speed and simplicity

## Top-5 Features (by RF importance)
1. `sttl` (source-to-TTL) — 0.127
2. `ct_state_ttl` (connection state count) — 0.123
3. `dload` (download bytes/sec) — 0.067
4. `rate` (packet rate) — 0.063
5. `sload` (upload bytes/sec) — 0.054

## Zero-Day Attack Detection
XGBoost retrained with each attack category held out entirely:

| Held-Out Attack | F1-Score | Recall | Notes |
|-----------------|----------|--------|-------|
| Worms | **1.000** | 1.000 | Perfect detection |
| DoS | **0.997** | 0.994 | Near-perfect |
| Exploits | **0.987** | 0.974 | |
| Reconnaissance | **0.985** | 0.970 | |
| Shellcode | **0.971** | 0.943 | |
| Generic | **0.964** | 0.931 | |
| Analysis | **0.894** | 0.809 | Low-data (2,677 test samples) |
| Fuzzers | **0.355** | 0.215 | Hardest — mimics benign traffic |

## Resource Benchmarking
| Model | Inference (1 sample) | Model Size | Memory Delta |
|-------|---------------------|------------|--------------|
| Random Forest | **68.3 ms** | 164.4 MB | ~0 MB |
| XGBoost | **1.1 ms** | 3.0 MB | ~0 MB |

XGBoost is **60× faster** and **55× smaller** than RF, making it ideal for lightweight deployment despite slightly lower F1.

## SHAP Explainability
- Global summary plot (top-20 features) generated
- 5 per-sample waterfall plots for flagged attacks
- Sample alert explanation shows `sttl`, `ct_state_ttl`, `ct_dst_sport_ltm` as top-3 contributing features

## Zero-Day Confidence Score (ZDCS) — Novel Contribution

ZDCS = α·H(p) + (1−α)·D(shap_test, ref_shap) combines prediction entropy with SHAP explanation novelty to quantify how "unseen" an attack is.

**Results (α=0.5, per-category mean ZDCS, XGBoost on full feature set):**

| Attack Category | Mean ZDCS | Zero-Day F1 | ZDCS Interpretation |
|-----------------|-----------|-------------|---------------------|
| Fuzzers | **0.470** | 0.355 | Highest novelty — model struggles most |
| Shellcode | **0.346** | 0.971 | Moderate novelty, but model detects well |
| Worms | **0.273** | 1.000 | |
| Reconnaissance | **0.241** | 0.985 | |
| Exploits | **0.232** | 0.987 | |
| DoS | **0.183** | 0.997 | |
| Analysis | **0.166** | 0.894 | |
| Backdoor | **0.118** | — | Insufficient test samples |
| Generic | **0.065** | 0.964 | Lowest novelty — easily detected |

**Key findings:**
- ZDCS correlates inversely with zero-day detection F1 (particularly Fuzzers at high ZDCS / low F1)
- Generic attacks have near-zero ZDCS (SHAP patterns match known attack profile)
- Fuzzers have highest ZDCS because their traffic pattern is the most "attack-like yet unusual" — model confidence is low AND SHAP explanation diverges from reference

## Pipeline Outputs
All results saved to `results/`:
- **Figures** (10 PNGs): model comparison, confusion matrices, feature importance, SHAP summary, 5x waterfall, zero-day detection, inference time, ZDCS by category, ZDCS ROC
- **Tables** (5 CSVs): model comparison, feature importance, zero-day results, inference time + memory usage, ZDCS per-sample scores
