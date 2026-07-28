import logging
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_auc_score,
                             classification_report)

logger = logging.getLogger(__name__)


class MetricsEvaluator:
    def __init__(self):
        self.results = {}

    def evaluate(self, y_true, y_pred, y_prob=None, model_name: str = "model"):
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0)
        }

        if y_prob is not None:
            try:
                metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
            except ValueError:
                metrics["roc_auc"] = None

        metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
        self.results[model_name] = metrics

        logger.info(f"\n=== {model_name} ===")
        logger.info(f"Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall:    {metrics['recall']:.4f}")
        logger.info(f"F1-Score:  {metrics['f1']:.4f}")
        if metrics.get("roc_auc"):
            logger.info(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
        logger.info(f"Confusion Matrix:\n{metrics['confusion_matrix']}")

        return metrics

    def comparison_table(self) -> pd.DataFrame:
        rows = []
        for name, metrics in self.results.items():
            row = {
                "Model": name,
                "Accuracy": f"{metrics['accuracy']:.4f}",
                "Precision": f"{metrics['precision']:.4f}",
                "Recall": f"{metrics['recall']:.4f}",
                "F1-Score": f"{metrics['f1']:.4f}",
            }
            if metrics.get("roc_auc"):
                row["ROC-AUC"] = f"{metrics['roc_auc']:.4f}"
            rows.append(row)
        return pd.DataFrame(rows)

    def save_comparison(self, path: str):
        df = self.comparison_table()
        df.to_csv(path, index=False)
        logger.info(f"Comparison table saved to {path}")
