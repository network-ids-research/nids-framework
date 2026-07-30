# Lightweight Explainable NIDS — Results Summary

## Objectives

| # | Objective | Result | File |
|---|-----------|--------|------|
| 1 | Leakage-free pipeline | Clean train/test split before any processing | [View](objectives/01_leakage_free_pipeline.md) |
| 2 | Model comparison (2–3 models) | XGBoost selected (F1=0.893, 1.1ms inference) | [View](objectives/02_model_comparison.md) |
| 3 | Feature reduction (~49 → optimized) | Top-25 features, <0.2% F1 loss | [View](objectives/03_feature_reduction.md) |
| 4 | Zero-day simulation (leave-one-out) | Fuzzers hardest (F1=0.355), Worms easiest (1.0) | [View](objectives/04_zero_day_simulation.md) |
| 5 | Inference time & resource usage | XGBoost: 1.1ms/sample, 3MB | [View](objectives/05_resource_benchmarking.md) |
| 6 | SHAP explainability per alert | Summary + waterfall plots + alert API | [View](objectives/06_shap_explainability.md) |
| — | **Bonus:** ZDCS (novel contribution) | Entropy + SHAP novelty for zero-day scoring | [View](objectives/07_zdcs_novel_contribution.md) |

## Output files

| Directory | Contents |
|-----------|----------|
| `results/figures/` | 10 PNGs: CM, feature importance, SHAP summary/waterfall, zero-day detection, inference time, ZDCS plots |
| `results/tables/` | 5 CSVs: model comparison, feature importance, zero-day results, inference time, ZDCS scores |
| `results/models/` | Trained RF + XGBoost `.pkl` files |
