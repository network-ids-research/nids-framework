#!/usr/bin/env python3
"""
End-to-end pipeline for Lightweight Explainable Network Anomaly Detection.

Usage:
    python pipeline.py                    # Full pipeline
    python pipeline.py --skip-zero-day    # Skip zero-day simulation
    python pipeline.py --skip-benchmark   # Skip benchmarking
    python pipeline.py --skip-dl          # Skip deep learning training
"""

import argparse
import logging
import sys
import yaml
import pandas as pd
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("pipeline")


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    logger.info(f"Config loaded from {config_path}")
    return config


def run_pipeline(config: dict, skip_zero_day: bool = False,
                 skip_benchmark: bool = False, skip_dl: bool = False):
    from src.data import DataLoader, DataCleaner, DataSplitter
    from src.features import FeatureBuilder, FeatureSelector
    from src.models import RandomForestTrainer, XGBoostTrainer, DeepLearningTrainer
    from src.models import ModelPredictor
    from src.evaluation import MetricsEvaluator, ZeroDaySimulator, Benchmarker
    from src.explainability import SHAPExplainer
    from src.visualization import Plotter

    # =========================================================================
    # Phase 1: Data Pipeline
    # =========================================================================
    logger.info("=" * 60)
    logger.info("PHASE 1: Data Loading & Cleaning")
    logger.info("=" * 60)

    loader = DataLoader(config)
    cleaner = DataCleaner(config)
    splitter = DataSplitter(config)

    try:
        train_df, test_df = loader.load_raw()
    except FileNotFoundError as e:
        logger.error(f"Dataset not found: {e}")
        logger.error("Place UNSW-NB15 CSV files in data/raw/ and update config.")
        sys.exit(1)

    df = loader.merge_partitions(train_df, test_df)
    df = cleaner.clean(df)

    # Correct split: split before balancing to avoid data leakage
    train_df, test_df = splitter.split_by_partition(df)
    X_train, y_train, attack_cat_train = cleaner.separate_features_target(train_df)
    X_test, y_test, _ = cleaner.separate_features_target(test_df)

    logger.info(f"Final train: {X_train.shape}, test: {X_test.shape}")
    logger.info(f"Train attack distribution:\n{y_train.value_counts()}")

    # =========================================================================
    # Phase 2: Feature Evaluation
    # =========================================================================
    logger.info("=" * 60)
    logger.info("PHASE 2: Feature Evaluation & Selection")
    logger.info("=" * 60)

    feature_selector = FeatureSelector(config)
    plotter = Plotter()

    importance = feature_selector.rank_features(X_train, y_train)
    plotter.feature_importance(importance, top_n=20)
    importance.to_csv("results/tables/feature_importance.csv")
    logger.info("Feature importance saved to results/tables/feature_importance.csv")

    # Evaluate reduced feature sets
    reduction_results = []
    for n in config["features"]["reduced_sizes"]:
        selected = feature_selector.select_top_n(importance, n)
        metrics = feature_selector.evaluate_reduced_set(
            X_train, y_train, X_test, y_test, selected
        )
        reduction_results.append(metrics)
        logger.info(f"Top-{n} features: Accuracy={metrics['accuracy']:.4f}, "
                     f"F1={metrics['f1']:.4f}")

    best_n = max(reduction_results, key=lambda x: x["f1"])["n_features"]
    logger.info(f"Best performing reduced set: top-{best_n} features")

    # Use full feature set for model comparison (as baseline)
    selected_features = feature_selector.select_top_n(importance, best_n)
    feature_selector.save_selected_features(selected_features)

    X_train_red = X_train[selected_features]
    X_test_red = X_test[selected_features]

    # =========================================================================
    # Phase 3: Model Training & Comparison
    # =========================================================================
    logger.info("=" * 60)
    logger.info("PHASE 3: Model Training & Comparison")
    logger.info("=" * 60)

    evaluator = MetricsEvaluator()
    trained_models = {}

    # Random Forest
    rf_trainer = RandomForestTrainer(config)
    logger.info("Training Random Forest...")
    rf_model, rf_params, rf_f1 = rf_trainer.train(X_train, y_train)
    trained_models["Random Forest"] = rf_model

    y_pred_rf = rf_model.predict(X_test)
    y_prob_rf = rf_model.predict_proba(X_test)[:, 1]
    evaluator.evaluate(y_test, y_pred_rf, y_prob_rf, "Random Forest")
    plotter.confusion_matrix(
        evaluator.results["Random Forest"]["confusion_matrix"], "Random Forest"
    )

    # XGBoost
    xgb_trainer = XGBoostTrainer(config)
    logger.info("Training XGBoost...")
    xgb_model, xgb_params, xgb_f1 = xgb_trainer.train(X_train, y_train)
    trained_models["XGBoost"] = xgb_model

    y_pred_xgb = xgb_model.predict(X_test)
    y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]
    evaluator.evaluate(y_test, y_pred_xgb, y_prob_xgb, "XGBoost")
    plotter.confusion_matrix(
        evaluator.results["XGBoost"]["confusion_matrix"], "XGBoost"
    )

    # Deep Learning (optional)
    if not skip_dl:
        dl_trainer = DeepLearningTrainer(config)
        logger.info("Training Deep Learning model...")
        dl_model, dl_params, dl_f1 = dl_trainer.train(X_train_red.values, y_train.values)
        trained_models["Deep Learning"] = dl_model

        y_pred_dl = (dl_model.predict(X_test_red.values, verbose=0) > 0.5).astype(int)
        y_prob_dl = dl_model.predict(X_test_red.values, verbose=0).flatten()
        evaluator.evaluate(y_test, y_pred_dl, y_prob_dl, "Deep Learning")
        plotter.confusion_matrix(
            evaluator.results["Deep Learning"]["confusion_matrix"], "Deep Learning"
        )

    # Save comparison table
    comparison_df = evaluator.comparison_table()
    evaluator.save_comparison("results/tables/model_comparison.csv")
    plotter.model_comparison(comparison_df)
    logger.info(f"\nModel Comparison:\n{comparison_df.to_string()}")

    # Identify best model
    best_model_name = max(evaluator.results,
                          key=lambda k: evaluator.results[k]["f1"])
    best_model = trained_models[best_model_name]
    logger.info(f"Best model: {best_model_name} "
                f"(F1={evaluator.results[best_model_name]['f1']:.4f})")

    # =========================================================================
    # Phase 4: Zero-Day Simulation
    # =========================================================================
    if not skip_zero_day:
        logger.info("=" * 60)
        logger.info("PHASE 4: Zero-Day Attack Simulation")
        logger.info("=" * 60)

        zero_day = ZeroDaySimulator(config)
        zero_day_df = zero_day.simulate(
            pd.concat([train_df, test_df], ignore_index=True),
            config["data"]["label_col"],
            config["data"]["attack_cat_col"],
            lambda Xz, yz: xgb_trainer.train(Xz, yz)[0],
            selected_features if not skip_dl else X_train.columns.tolist()
        )
        zero_day_df.to_csv("results/tables/zero_day_results.csv", index=False)
        plotter.zero_detection_rate(zero_day_df)
        logger.info(f"Zero-day results saved to results/tables/zero_day_results.csv")

    # =========================================================================
    # Phase 5: Explainability (SHAP)
    # =========================================================================
    logger.info("=" * 60)
    logger.info("PHASE 5: Explainability with SHAP")
    logger.info("=" * 60)

    explainer = SHAPExplainer(config)
    explainer.fit(best_model, X_test)
    explainer.summary_plot(X_test, max_display=20)

    # Generate per-sample explanations for flagged alerts
    anomalous_idx = y_test[y_test == 1].index[:5]
    if len(anomalous_idx) > 0:
        anomalous_samples = X_test.loc[anomalous_idx]
        explainer.waterfall_plot(
            anomalous_samples,
            indices=list(range(len(anomalous_samples)))
        )

    # Sample explain_alert
    sample = X_test.iloc[[0]]
    alert = explainer.explain_alert(best_model, sample)
    logger.info(f"Sample alert explanation:\n{alert}")

    # =========================================================================
    # Phase 6: Resource & Speed Benchmarking
    # =========================================================================
    if not skip_benchmark:
        logger.info("=" * 60)
        logger.info("PHASE 6: Resource & Speed Benchmarking")
        logger.info("=" * 60)

        benchmarker = Benchmarker(config)
        X_sample = X_test.values[:100]

        inf_df, mem_df = benchmarker.benchmark_all(trained_models, X_sample)
        inf_df.to_csv("results/tables/inference_time.csv", index=False)
        mem_df.to_csv("results/tables/memory_usage.csv", index=False)
        plotter.inference_time(inf_df)
        logger.info(f"Inference times:\n{inf_df.to_string()}")
        logger.info(f"Memory usage:\n{mem_df.to_string()}")

    # =========================================================================
    # Summary
    # =========================================================================
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Best model: {best_model_name}")
    logger.info("Results saved to:")
    logger.info("  - results/tables/model_comparison.csv")
    logger.info("  - results/tables/feature_importance.csv")
    logger.info("  - results/figures/")
    if not skip_zero_day:
        logger.info("  - results/tables/zero_day_results.csv")
    if not skip_benchmark:
        logger.info("  - results/tables/inference_time.csv")
        logger.info("  - results/tables/memory_usage.csv")


def main():
    parser = argparse.ArgumentParser(
        description="Lightweight Explainable Network Anomaly Detection Pipeline"
    )
    parser.add_argument("--config", default="config/config.yaml",
                        help="Path to configuration file")
    parser.add_argument("--skip-zero-day", action="store_true",
                        help="Skip zero-day simulation")
    parser.add_argument("--skip-benchmark", action="store_true",
                        help="Skip resource benchmarking")
    parser.add_argument("--skip-dl", action="store_true",
                        help="Skip deep learning training")
    args = parser.parse_args()

    config = load_config(args.config)
    run_pipeline(config, args.skip_zero_day, args.skip_benchmark, args.skip_dl)


if __name__ == "__main__":
    main()
