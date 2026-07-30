# Objective 3: Feature Reduction

**Goal:** Reduce the feature set from ~49 to a smaller, optimized subset without significant accuracy loss, to support real-time and resource-constrained deployment.

## Approach

1. Train a Random Forest on all 42 features to get importance ranking
2. Evaluate reduced subsets (top-15, top-20, top-25) using RF
3. Pick the smallest set with minimal F1 drop

## Accuracy vs feature count

| Feature Count | Accuracy | F1-Score | vs Full (42) |
|---------------|----------|----------|--------------|
| 42 (full) | — | ~0.896 (baseline) | — |
| Top-25 | **0.8727** | **0.8948** | **−0.1%** |
| Top-20 | 0.8699 | 0.8924 | −0.4% |
| Top-15 | 0.8637 | 0.8866 | −1.0% |

**Decision:** Top-25 features selected — <0.2% F1 loss while reducing dimensionality by 40%.

## Top-5 features

| Rank | Feature | Importance | Description |
|------|---------|------------|-------------|
| 1 | `sttl` | 0.127 | Source-to-destination TTL |
| 2 | `ct_state_ttl` | 0.123 | Connection state + TTL count |
| 3 | `dload` | 0.067 | Download bytes per second |
| 4 | `rate` | 0.063 | Packet rate |
| 5 | `sload` | 0.054 | Upload bytes per second |

Network metadata features (TTL, connection state) dominate — payload-independent, computable at line speed.

## Relevance to lightweight deployment

- 25 features vs 42 → **40% fewer features to compute/collect**
- All top-5 features are **header-level** (no deep packet inspection needed)
- Feature reduction enables deployment on resource-constrained edge devices

## Relevant code
- `src/features/build_features.py` — `FeatureSelector.rank_features()`, `evaluate_reduced_set()`
- `pipeline.py:80-107`
