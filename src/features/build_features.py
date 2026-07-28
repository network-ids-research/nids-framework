import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

logger = logging.getLogger(__name__)


class FeatureBuilder:
    def __init__(self, config: dict):
        self.config = config

    def prepare_data(self, X_train: pd.DataFrame, y_train: pd.Series,
                     X_test: pd.DataFrame) -> tuple:
        feature_cols = [c for c in X_train.columns if c not in self.config["data"]["id_cols"]
                        and c != self.config["data"]["label_col"]
                        and c != self.config["data"]["attack_cat_col"]]
        return X_train[feature_cols], X_test[feature_cols]


class FeatureSelector:
    def __init__(self, config: dict):
        self.config = config
        self.reduced_sizes = config["features"]["reduced_sizes"]
        self.selected_features_path = Path(config["features"]["selected_features_file"])

    def rank_features(self, X_train: pd.DataFrame, y_train: pd.Series) -> pd.Series:
        logger.info("Training RF for feature importance ranking...")
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        importances = pd.Series(rf.feature_importances_, index=X_train.columns)
        importances = importances.sort_values(ascending=False)
        logger.info(f"Top 10 features:\n{importances.head(10)}")
        return importances

    def select_top_n(self, importance_series: pd.Series, n: int) -> list:
        return importance_series.head(n).index.tolist()

    def evaluate_reduced_set(self, X_train: pd.DataFrame, y_train: pd.Series,
                              X_test: pd.DataFrame, y_test: pd.Series,
                              selected_features: list) -> dict:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        X_train_sub = X_train[selected_features]
        X_test_sub = X_test[selected_features]

        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train_sub, y_train)
        y_pred = rf.predict(X_test_sub)

        return {
            "n_features": len(selected_features),
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred)
        }

    def compute_permutation_importance(self, X_train: pd.DataFrame, y_train: pd.Series) -> pd.Series:
        logger.info("Computing permutation importance...")
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        perm = permutation_importance(rf, X_train, y_train, n_repeats=5, random_state=42, n_jobs=-1)
        importances = pd.Series(perm.importances_mean, index=X_train.columns).sort_values(ascending=False)
        return importances

    def save_selected_features(self, features: list):
        self.selected_features_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.selected_features_path, "w") as f:
            json.dump({"selected_features": features}, f, indent=2)
        logger.info(f"Selected features saved to {self.selected_features_path}")

    def load_selected_features(self) -> list:
        if self.selected_features_path.exists():
            with open(self.selected_features_path) as f:
                return json.load(f)["selected_features"]
        return []
