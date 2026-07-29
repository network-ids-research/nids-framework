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

## Pipeline Outputs
All results saved to `results/`:
- **Figures** (8 PNGs): model comparison, confusion matrices, feature importance, SHAP summary, 5x waterfall, zero-day detection, inference time
- **Tables** (4 CSVs): model comparison, feature importance, zero-day results, inference time + memory usage
