# Bonus: Zero-Day Confidence Score (ZDCS)

**Novel contribution** — Combines prediction uncertainty with SHAP explanation novelty to quantify how "unseen" an attack is.

## Formula

```
ZDCS = α · H(p) + (1 − α) · D(shap_test, ref_shap)
```

- **H(p)**: Normalized entropy of prediction probability (0 = certain, 1 = maximally uncertain)
- **D(shap_test, ref_shap)**: Cosine distance between sample's SHAP vector and mean SHAP of known attacks
- **α = 0.5**: Equal weighting

Higher ZDCS = more likely to be a novel / previously unseen attack pattern.

## Method

1. Compute reference SHAP signature from 300 training attack samples
2. For each test sample, compute ZDCS using the trained XGBoost model
3. Group by attack category and report mean ZDCS
4. Correlate with zero-day detection F1 from Objective 4

## Results

| Attack Category | Mean ZDCS | Zero-Day F1 | Interpretation |
|-----------------|-----------|-------------|----------------|
| Fuzzers | **0.470** | 0.355 | Highest novelty — model struggles most |
| Shellcode | **0.346** | 0.971 | High novelty, but model still detects |
| Worms | **0.273** | 1.000 | Moderate novelty |
| Reconnaissance | **0.241** | 0.985 | |
| Exploits | **0.232** | 0.987 | |
| DoS | **0.183** | 0.997 | |
| Analysis | **0.166** | 0.894 | |
| Backdoor | **0.118** | — | Insufficient samples |
| Generic | **0.065** | 0.964 | Lowest novelty |

## Key insight

ZDCS correlates inversely with zero-day detection F1:

- **Fuzzers** (ZDCS=0.470, F1=0.355) — both low confidence AND unusual SHAP pattern
- **Generic** (ZDCS=0.065, F1=0.964) — matches known attack SHAP profile, easily detected

This suggests ZDCS could be used to **flag potentially missed zero-day attacks** in production: samples with high ZDCS but low prediction confidence may warrant manual review.

## Outputs
- `results/tables/zdcs_results.csv` — per-sample ZDCS, entropy, SHAP distance
- `results/figures/zdcs_by_category.png` — boxplot + correlation with zero-day F1
- `results/figures/zdcs_roc.png` — ROC curve: ZDCS vs raw probability for hard attack detection

## Relevant code
- `src/evaluation/zero_day_confidence.py` — `ZeroDayConfidenceScore` class
- `pipeline.py:217-245`
