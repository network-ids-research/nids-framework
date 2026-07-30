#!/usr/bin/env python3
"""
End-to-end pipeline for Lightweight Explainable Network Anomaly Detection.

Usage:
    python pipeline.py                          # Full pipeline
    python pipeline.py --phase 1                # Phase 1 only
    python pipeline.py --phase 1-3              # Phases 1 through 3
    python pipeline.py --phase 4                # Phase 4 only (loads checkpoint)
    python pipeline.py --phase 5,5.5            # SHAP + ZDCS (loads checkpoint)
    python pipeline.py --skip-zero-day          # Skip zero-day simulation
    python pipeline.py --skip-benchmark         # Skip benchmarking
    python pipeline.py --skip-dl                # Skip deep learning training
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
                 skip_benchmark: bool = False, skip_dl: bool = False,
                 phase_spec: str = None):
    from src.data import DataLoader, DataCleaner, DataSplitter
    from src.features import FeatureBuilder, FeatureSelector
    from src.models import RandomForestTrainer, XGBoostTrainer, DeepLearningTrainer
    from src.models import ModelPredictor
    from src.evaluation import MetricsEvaluator, ZeroDaySimulator, Benchmarker, ZeroDayConfidenceScore
    from src.explainability import SHAPExplainer
    from src.visualization import Plotter
    from src.utils.checkpoint import save_phase1, load_phase1, checkpoint_exists, parse_phase

    phases = parse_phase(phase_spec)
    run_all = phases is None
    check = lambda p: run_all or p in phases
    must_checkpoint = lambda: phases is not None and 1 not in phases

    # Load checkpoint if skipping Phase 1
    if must_checkpoint():
        if not checkpoint_exists():
            logger.error("No checkpoint found. Run --phase 1 first or omit --phase for full pipeline.")
            sys.exit(1)
        X_train, y_train, X_test, y_test, attack_cat_test, train_df, test_df = load_phase1()
        logger.info(f"Loaded: train {X_train.shape}, test {X_test.shape}")
    else:
        # =========================================================================
        # Phase 1: Data Pipeline
        # =========================================================================
        if check(1):
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

            train_df, test_df = splitter.split_by_partition(df)
            X_train, y_train, attack_cat_train = cleaner.separate_features_target(train_df)
            X_test, y_test, attack_cat_test = cleaner.separate_features_target(test_df)

            logger.info(f"Final train: {X_train.shape}, test: {X_test.shape}")
            logger.info(f"Train attack distribution:\n{y_train.value_counts()}")

            save_phase1(X_train, y_train, X_test, y_test, attack_cat_test, train_df, test_df)

            if not run_all:
                logger.info("Phase 1 complete. Checkpoint saved.")
                return
        else:
            logger.error("Phase 1 must be included or a checkpoint must exist.")
            sys.exit(1)
    # =========================================================================
    # Phase 2: Feature Evaluation
    # =========================================================================
    if check(2):
        logger.info("=" * 60)
        logger.info("PHASE 2: Feature Evaluation & Selection")
        logger.info("=" * 60)

        feature_selector = FeatureSelector(config)
        plotter = Plotter()

        importance = feature_selector.rank_features(X_train, y_train)
        plotter.feature_importance(importance, top_n=20)
        importance.to_csv("results/tables/feature_importance.csv")
        logger.info("Feature importance saved to results/tables/feature_importance.csv")

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

        selected_features = feature_selector.select_top_n(importance, best_n)
        feature_selector.save_selected_features(selected_features)

        X_train_red = X_train[selected_features]
        X_test_red = X_test[selected_features]

        if not run_all:
            logger.info("Phase 2 complete.")
            return

    # Load Phase 2 outputs if skipped
    if not check(2) and not run_all:
        import json
        sf_path = Path("data/processed/selected_features.json")
        if not sf_path.exists():
            logger.error("Phase 2 results not found. Run --phase 2 first.")
            sys.exit(1)
        with open(sf_path) as f:
            selected_features = json.load(f)["selected_features"]
        X_train_red = X_train[selected_features]
        X_test_red = X_test[selected_features]
        logger.info(f"Loaded selected features ({len(selected_features)} features)")

    # =========================================================================
    # Phase 3: Model Training & Comparison
    # =========================================================================
    if check(3):
        logger.info("=" * 60)
        logger.info("PHASE 3: Model Training & Comparison")
        logger.info("=" * 60)

        if not check(2):
            from src.visualization import Plotter as _Plotter
            plotter = _Plotter()

        evaluator = MetricsEvaluator()
        trained_models = {}

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

        comparison_df = evaluator.comparison_table()
        evaluator.save_comparison("results/tables/model_comparison.csv")
        plotter.model_comparison(comparison_df)
        logger.info(f"\nModel Comparison:\n{comparison_df.to_string()}")

        best_model_name = max(evaluator.results,
                              key=lambda k: evaluator.results[k]["f1"])
        best_model = trained_models[best_model_name]
        logger.info(f"Best model: {best_model_name} "
                    f"(F1={evaluator.results[best_model_name]['f1']:.4f})")

        if not run_all:
            logger.info("Phase 3 complete.")
            return

    # =========================================================================
    # Phase 4: Zero-Day Simulation
    # =========================================================================
    if check(4) and not skip_zero_day:
        logger.info("=" * 60)
        logger.info("PHASE 4: Zero-Day Attack Simulation")
        logger.info("=" * 60)

        xgb_trainer = XGBoostTrainer(config)
        feature_list = selected_features if not skip_dl else X_train.columns.tolist()

        zero_day = ZeroDaySimulator(config)
        zero_day_df = zero_day.simulate(
            pd.concat([train_df, test_df], ignore_index=True),
            config["data"]["label_col"],
            config["data"]["attack_cat_col"],
            lambda Xz, yz: xgb_trainer.train(Xz, yz)[0],
            feature_list
        )
        zero_day_df.to_csv("results/tables/zero_day_results.csv", index=False)
        plotter = Plotter()
        plotter.zero_detection_rate(zero_day_df)
        logger.info(f"Zero-day results saved to results/tables/zero_day_results.csv")

        if not run_all:
            logger.info("Phase 4 complete.")
            return

    # =========================================================================
    # Phase 5: Explainability (SHAP)
    # =========================================================================
    if check(5):
        logger.info("=" * 60)
        logger.info("PHASE 5: Explainability with SHAP")
        logger.info("=" * 60)

        predictor = ModelPredictor()
        best_model = None
        for path, name, mtype in [
            ("results/models/xgb_best.pkl", "XGBoost", "xgboost"),
            ("results/models/rf_best.pkl", "Random Forest", "sklearn"),
        ]:
            pp = Path(path)
            if pp.exists():
                best_model = predictor.load_model(name, path, mtype)
                best_model_name = name
                logger.info(f"Loaded model: {name}")
                break
        if best_model is None:
            logger.error("No trained model found. Run --phase 3 first.")
            sys.exit(1)

        plotter = Plotter()
        explainer = SHAPExplainer(config)
        explainer.fit(best_model, X_test)
        explainer.summary_plot(X_test, max_display=20)

        anomalous_idx = y_test[y_test == 1].index[:5]
        if len(anomalous_idx) > 0:
            anomalous_samples = X_test.loc[anomalous_idx]
            explainer.waterfall_plot(
                anomalous_samples,
                indices=list(range(len(anomalous_samples)))
            )

        sample = X_test.iloc[[0]]
        alert = explainer.explain_alert(best_model, sample)
        logger.info(f"Sample alert explanation:\n{alert}")

        if not run_all:
            logger.info("Phase 5 complete.")
            return

    # =========================================================================
    # Phase 5.5: Zero-Day Confidence Score (ZDCS) Evaluation
    # =========================================================================
    if check(5.5):
        logger.info("=" * 60)
        logger.info("PHASE 5.5: Zero-Day Confidence Score (ZDCS)")
        logger.info("=" * 60)

        predictor = ModelPredictor()
        best_model = None
        for path, name, mtype in [
            ("results/models/xgb_best.pkl", "XGBoost", "xgboost"),
            ("results/models/rf_best.pkl", "Random Forest", "sklearn"),
        ]:
            pp = Path(path)
            if pp.exists():
                best_model = predictor.load_model(name, path, mtype)
                logger.info(f"Loaded model: {name}")
                break
        if best_model is None:
            logger.error("No trained model found. Run --phase 3 first.")
            sys.exit(1)

        explainer = SHAPExplainer(config)
        explainer.fit(best_model, X_test)

        zdcs_eval = ZeroDayConfidenceScore(alpha=0.5)
        zdcs_df = zdcs_eval.evaluate_by_category(
            best_model, explainer,
            X_train, y_train, X_test, attack_cat_test,
            max_ref=300, max_per_cat=100,
        )

        zdcs_df.to_csv("results/tables/zdcs_results.csv", index=False)
        logger.info("ZDCS by category computed. Head:\n%s", zdcs_df.groupby("attack_category")["zdcs"].describe().to_string())

        zero_day_f1 = None
        try:
            zd = pd.read_csv("results/tables/zero_day_results.csv")
            if "f1" in zd.columns and "held_out_attack" in zd.columns:
                zero_day_f1 = zd[["held_out_attack", "f1"]]
                logger.info("Loaded existing zero-day F1 scores for ZDCS correlation")
        except (FileNotFoundError, KeyError):
            pass

        plotter = Plotter()
        plotter.zdcs_comparison(zdcs_df, zero_day_f1)
        plotter.zdcs_roc(zdcs_df)

        if not run_all:
            logger.info("Phase 5.5 complete.")
            return

    # =========================================================================
    # Phase 6: Resource & Speed Benchmarking
    # =========================================================================
    if check(6) and not skip_benchmark:
        logger.info("=" * 60)
        logger.info("PHASE 6: Resource & Speed Benchmarking")
        logger.info("=" * 60)

        predictor = ModelPredictor()
        trained_models = {}
        for name, path, mtype in [
            ("XGBoost", "results/models/xgb_best.pkl", "xgboost"),
            ("Random Forest", "results/models/rf_best.pkl", "sklearn"),
        ]:
            pp = Path(path)
            if pp.exists():
                trained_models[name] = predictor.load_model(name, path, mtype)
        if not trained_models:
            logger.error("No trained models found. Run --phase 3 first.")
            sys.exit(1)

        benchmarker = Benchmarker(config)
        X_sample = X_test.values[:100]

        inf_df, mem_df = benchmarker.benchmark_all(trained_models, X_sample)
        inf_df.to_csv("results/tables/inference_time.csv", index=False)
        mem_df.to_csv("results/tables/memory_usage.csv", index=False)
        plotter = Plotter()
        plotter.inference_time(inf_df)
        logger.info(f"Inference times:\n{inf_df.to_string()}")
        logger.info(f"Memory usage:\n{mem_df.to_string()}")

        if not run_all:
            logger.info("Phase 6 complete.")
            return

    # =========================================================================
    # Summary
    # =========================================================================
    if run_all:
        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Best model: {best_model_name}")
        logger.info("Results saved to:")
        logger.info("  - results/tables/model_comparison.csv")
        logger.info("  - results/tables/feature_importance.csv")
        logger.info("  - results/tables/zdcs_results.csv")
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
    parser.add_argument("--phase", default=None,
                        help="Phase(s) to run: '1', '1-3', '4,5,5.5', '5.5' (default: all)")
    parser.add_argument("--skip-zero-day", action="store_true",
                        help="Skip zero-day simulation")
    parser.add_argument("--skip-benchmark", action="store_true",
                        help="Skip resource benchmarking")
    parser.add_argument("--skip-dl", action="store_true",
                        help="Skip deep learning training")
    args = parser.parse_args()

    config = load_config(args.config)
    run_pipeline(config, args.skip_zero_day, args.skip_benchmark, args.skip_dl, args.phase)


if __name__ == "__main__":
    main()
