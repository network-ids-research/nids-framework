import logging
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from pathlib import Path

logger = logging.getLogger(__name__)


class SHAPExplainer:
    def __init__(self, config: dict):
        self.config = config
        self.max_samples = config["explainability"]["max_samples"]
        self.summary_plot_path = Path(config["explainability"]["summary_plot"])
        self.waterfall_dir = Path(config["explainability"]["waterfall_dir"])
        self.explainer = None

    def fit(self, model, X_background: pd.DataFrame):
        if str(type(model)).find("xgboost") >= 0:
            self.explainer = shap.TreeExplainer(model)
        elif str(type(model)).find("RandomForest") >= 0:
            self.explainer = shap.TreeExplainer(model)
        elif str(type(model)).find("keras") >= 0 or str(type(model)).find("tensorflow") >= 0:
            X_sample = X_background.sample(n=min(100, len(X_background)), random_state=42)
            self.explainer = shap.KernelExplainer(model.predict, X_sample)
        else:
            X_sample = X_background.sample(n=min(100, len(X_background)), random_state=42)
            self.explainer = shap.KernelExplainer(
                lambda x: model.predict_proba(x)[:, 1], X_sample
            )
        logger.info(f"SHAP explainer initialized: {type(self.explainer).__name__}")

    def explain(self, X: pd.DataFrame) -> np.ndarray:
        if self.explainer is None:
            raise ValueError("Explainer not fitted. Call .fit() first.")

        X_sample = X.sample(n=min(self.max_samples, len(X)), random_state=42)
        shap_values = self.explainer.shap_values(X_sample)

        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        if hasattr(shap_values, "ndim") and shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]

        logger.info(f"SHAP values computed for {len(X_sample)} samples, shape={shap_values.shape}")
        return shap_values, X_sample

    def summary_plot(self, X: pd.DataFrame, max_display: int = 20):
        shap_values, X_sample = self.explain(X)
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_sample, max_display=max_display, show=False)
        self.summary_plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(self.summary_plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"SHAP summary plot saved to {self.summary_plot_path}")

    def waterfall_plot(self, X: pd.DataFrame, indices: list = None, max_display: int = 10):
        shap_values, X_sample = self.explain(X)
        self.waterfall_dir.mkdir(parents=True, exist_ok=True)

        if indices is None:
            indices = [0]

        expected_val = self.explainer.expected_value
        if isinstance(expected_val, (list, np.ndarray)):
            if hasattr(expected_val, "ndim") and expected_val.ndim == 2:
                expected_val = expected_val[1]
            elif hasattr(expected_val, "ndim") and expected_val.ndim == 1 and len(expected_val) > 1:
                expected_val = expected_val[1]
            elif isinstance(expected_val, list) and len(expected_val) > 1:
                expected_val = expected_val[1]

        for idx in indices:
            if idx >= len(X_sample):
                continue

            sv = shap_values[idx]
            if hasattr(sv, "ndim") and sv.ndim > 1:
                if sv.shape[-1] == 2:
                    sv = sv[..., 1]
                else:
                    sv = sv.flatten()

            plt.figure()
            shap.waterfall_plot(
                shap.Explanation(
                    values=sv,
                    base_values=expected_val,
                    data=X_sample.iloc[idx].values,
                    feature_names=X_sample.columns.tolist()
                ),
                max_display=max_display,
                show=False
            )
            save_path = self.waterfall_dir / f"waterfall_sample_{idx}.png"
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
            logger.info(f"Waterfall plot saved to {save_path}")

    def explain_alert(self, model, X_sample: pd.DataFrame, threshold: float = 0.5) -> dict:
        if self.explainer is None:
            X_bg = X_sample.sample(n=min(100, len(X_sample)), random_state=42) if len(X_sample) > 100 else X_sample
            self.fit(model, X_bg)

        shap_values = self.explainer.shap_values(X_sample)

        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        if hasattr(shap_values, "ndim") and shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]

        if len(shap_values.shape) == 1:
            shap_values = shap_values.reshape(1, -1)

        shap_values_row = shap_values[0]
        feature_contributions = pd.Series(shap_values_row, index=X_sample.columns)
        top_features = feature_contributions.abs().sort_values(ascending=False).head(3)

        expected = self.explainer.expected_value
        if isinstance(expected, (list, np.ndarray)):
            expected = expected[1] if len(expected) > 1 else expected[0]

        pred_score = float(expected + shap_values_row.sum())
        return {
            "prediction": int(pred_score > threshold),
            "prediction_score": pred_score,
            "top_features": top_features.to_dict(),
            "shap_values": shap_values_row.tolist(),
            "base_value": float(expected)
        }
