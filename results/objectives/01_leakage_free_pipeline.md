# Objective 1: Leakage-Free Pipeline

**Goal:** Build a clean traffic classification pipeline on UNSW-NB15 where no information leaks from test to training data.

## Implementation

Data splitting is done **before** any feature engineering, balancing, or model training:

```
raw CSVs → merge → clean → split_by_partition() → train / test
```

## Key design decisions

| Decision | Detail |
|----------|--------|
| Split strategy | `split_by_partition()` uses the original UNSW-NB15 train/test partitions — no random shuffle across partitions |
| When splitting occurs | Before balancing, feature selection, or any model training (`pipeline.py:66`) |
| Feature selection | Importance computed on train only; reduced sets evaluated on test only |
| SHAP explainer | Fitted on test data (explanation, not training) |

## Dataset sizes

| Partition | Samples | Attack | Normal |
|-----------|---------|--------|--------|
| Train | 175,341 | 119,341 | 56,000 |
| Test | 82,332 | 45,332 | 37,000 |

No cross-partition contamination. All downstream phases operate on cleanly separated data.

## Relevant code
- `split_by_partition()` in `src/data/data_splitter.py`
- Pipeline orchestration at `pipeline.py:56-72`
