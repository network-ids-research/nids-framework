import logging
import joblib
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelPredictor:
    def __init__(self):
        self.models = {}

    def load_model(self, name: str, path: str, model_type: str = "sklearn"):
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Model not found: {path}")

        if model_type == "sklearn":
            self.models[name] = joblib.load(path_obj)
        elif model_type == "keras":
            from tensorflow.keras.models import load_model
            self.models[name] = load_model(path_obj)
        elif model_type == "xgboost":
            import xgboost as xgb
            model = xgb.XGBClassifier()
            model.load_model(str(path_obj))
            self.models[name] = model

        logger.info(f"Loaded model '{name}' from {path}")
        return self.models[name]

    def predict(self, name: str, X, threshold: float = 0.5):
        if name not in self.models:
            raise ValueError(f"Model '{name}' not loaded")

        model = self.models[name]
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)[:, 1]
            return (probs >= threshold).astype(int), probs
        else:
            return model.predict(X), None

    def predict_batch(self, name: str, X_batches, threshold: float = 0.5):
        predictions = []
        probabilities = []
        for X_batch in X_batches:
            pred, prob = self.predict(name, X_batch, threshold)
            predictions.append(pred)
            if prob is not None:
                probabilities.append(prob)
        return np.concatenate(predictions), np.concatenate(probabilities) if probabilities else None
