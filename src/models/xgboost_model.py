import optuna
import logging
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import xgboost as xgb

logger = logging.getLogger(__name__)


class XGBoostTrainer:
    def __init__(self, config: dict):
        self.config = config
        self.model_config = config["models"]["xgboost"]
        self.save_path = Path(self.model_config["save_path"])
        self.best_model = None

    def _objective(self, trial, X_train, y_train, X_val, y_val):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 300, step=100),
            "max_depth": trial.suggest_int("max_depth", 4, 10, step=2),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "random_state": 42,
            "use_label_encoder": False,
            "eval_metric": "logloss",
            "verbosity": 0
        }
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        return f1_score(y_val, y_pred)

    def train(self, X_train, y_train, X_val=None, y_val=None):
        logger.info("Training XGBoost with Optuna...")
        if X_val is None or y_val is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42
            )

        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: self._objective(trial, X_train, y_train, X_val, y_val),
            n_trials=self.model_config["n_trials"]
        )

        logger.info(f"Best XGBoost params: {study.best_params}")
        logger.info(f"Best XGBoost F1: {study.best_value:.4f}")

        self.best_model = xgb.XGBClassifier(**study.best_params, random_state=42, verbosity=0)
        self.best_model.fit(X_train, y_train)

        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        self.best_model.save_model(str(self.save_path))
        logger.info(f"XGBoost model saved to {self.save_path}")

        return self.best_model, study.best_params, study.best_value
