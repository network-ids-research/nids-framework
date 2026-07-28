import joblib
import optuna
import logging
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

logger = logging.getLogger(__name__)


class RandomForestTrainer:
    def __init__(self, config: dict):
        self.config = config
        self.model_config = config["models"]["random_forest"]
        self.save_path = Path(self.model_config["save_path"])
        self.best_model = None

    def _objective(self, trial, X_train, y_train, X_val, y_val):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=100),
            "max_depth": trial.suggest_int("max_depth", 10, 40, step=10),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 4),
            "random_state": 42,
            "n_jobs": -1
        }
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        return f1_score(y_val, y_pred)

    def train(self, X_train, y_train, X_val=None, y_val=None):
        logger.info("Training Random Forest with Optuna...")
        if X_val is None or y_val is None:
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42
            )

        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: self._objective(trial, X_train, y_train, X_val, y_val),
            n_trials=self.model_config["n_trials"]
        )

        logger.info(f"Best RF params: {study.best_params}")
        logger.info(f"Best RF F1: {study.best_value:.4f}")

        self.best_model = RandomForestClassifier(**study.best_params, random_state=42, n_jobs=-1)
        self.best_model.fit(X_train, y_train)

        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.best_model, self.save_path)
        logger.info(f"RF model saved to {self.save_path}")

        return self.best_model, study.best_params, study.best_value
