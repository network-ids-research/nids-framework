import logging
import numpy as np
import pandas as pd
from copy import deepcopy
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

logger = logging.getLogger(__name__)


class ZeroDaySimulator:
    def __init__(self, config: dict):
        self.config = config
        self.attack_categories = config["evaluation"]["zero_day"]["attack_categories"]

    def simulate(self, df: pd.DataFrame, label_col: str, attack_cat_col: str,
                 train_model_fn, feature_cols: list) -> pd.DataFrame:
        """
        Leave-one-attack-out simulation.

        For each attack category, hide it from training, train on all others,
        and test detection on the held-out category.

        Args:
            df: Full dataframe with label and attack_cat columns
            label_col: Name of binary label column
            attack_cat_col: Name of attack category column
            train_model_fn: Function that takes (X_train, y_train) and returns a model
            feature_cols: List of feature columns to use

        Returns:
            DataFrame with zero-day detection results per category
        """
        results = []

        for held_out_cat in self.attack_categories:
            if held_out_cat == "Normal":
                continue

            logger.info(f"=== Zero-Day Simulation: Hiding '{held_out_cat}' ===")

            train_mask = df[attack_cat_col] != held_out_cat
            test_mask = df[attack_cat_col] == held_out_cat

            if test_mask.sum() == 0:
                logger.warning(f"No samples for '{held_out_cat}', skipping")
                continue

            X_train = df[train_mask][feature_cols]
            y_train = df[train_mask][label_col]
            X_test = df[test_mask][feature_cols]
            y_test = df[test_mask][label_col]

            model = train_model_fn(X_train, y_train)
            y_pred = model.predict(X_test)

            y_prob = None
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]

            metrics = {
                "held_out_attack": held_out_cat,
                "n_train": len(X_train),
                "n_test": len(X_test),
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1": f1_score(y_test, y_pred, zero_division=0)
            }
            results.append(metrics)
            logger.info(f"  {held_out_cat}: F1={metrics['f1']:.4f}, Recall={metrics['recall']:.4f}")

        results_df = pd.DataFrame(results)
        logger.info("\n=== Zero-Day Detection Summary ===")
        logger.info(f"\n{results_df.to_string()}")

        return results_df
