import logging
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)


class Plotter:
    def __init__(self, results_dir: str = "results/figures"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def confusion_matrix(self, cm: np.ndarray, model_name: str, labels: list = None):
        plt.figure(figsize=(6, 5))
        if labels is None:
            labels = ["Benign", "Attack"]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels, yticklabels=labels)
        plt.title(f"Confusion Matrix - {model_name}")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        save_path = self.results_dir / f"cm_{model_name.lower().replace(' ', '_')}.png"
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        logger.info(f"Confusion matrix saved to {save_path}")

    def feature_importance(self, importance_series: pd.Series, top_n: int = 20,
                           title: str = "Feature Importance"):
        plt.figure(figsize=(10, 8))
        top_features = importance_series.head(top_n)
        sns.barplot(x=top_features.values, y=top_features.index, palette="viridis")
        plt.title(title)
        plt.xlabel("Importance")
        plt.tight_layout()
        save_path = self.results_dir / "feature_importance.png"
        plt.savefig(save_path, dpi=150)
        plt.close()
        logger.info(f"Feature importance plot saved to {save_path}")

    def zero_detection_rate(self, results_df: pd.DataFrame):
        plt.figure(figsize=(12, 6))
        x = np.arange(len(results_df))
        width = 0.25

        plt.bar(x - width, results_df["recall"], width, label="Recall", alpha=0.8)
        plt.bar(x, results_df["precision"], width, label="Precision", alpha=0.8)
        plt.bar(x + width, results_df["f1"], width, label="F1-Score", alpha=0.8)

        plt.xlabel("Held-Out Attack Category")
        plt.ylabel("Score")
        plt.title("Zero-Day Detection Performance by Attack Category")
        plt.xticks(x, results_df["held_out_attack"], rotation=45, ha="right")
        plt.legend()
        plt.ylim(0, 1.05)
        plt.tight_layout()
        save_path = self.results_dir / "zero_day_detection.png"
        plt.savefig(save_path, dpi=150)
        plt.close()
        logger.info(f"Zero-day detection plot saved to {save_path}")

    def model_comparison(self, comparison_df: pd.DataFrame):
        metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
        plt.figure(figsize=(10, 6))
        x = np.arange(len(metrics))
        width = 0.25

        for i, row in comparison_df.iterrows():
            values = [float(row[m]) for m in metrics]
            plt.bar(x + i * width, values, width, label=row["Model"], alpha=0.8)

        plt.xlabel("Metric")
        plt.ylabel("Score")
        plt.title("Model Performance Comparison")
        plt.xticks(x + width, metrics)
        plt.legend()
        plt.ylim(0, 1.05)
        plt.tight_layout()
        save_path = self.results_dir / "model_comparison.png"
        plt.savefig(save_path, dpi=150)
        plt.close()
        logger.info(f"Model comparison plot saved to {save_path}")

    def inference_time(self, inference_df: pd.DataFrame):
        batch_cols = [c for c in inference_df.columns if "mean_ms" in c and c.startswith("batch_")]
        plt.figure(figsize=(10, 6))
        x = np.arange(len(batch_cols))
        width = 0.25

        for i, row in inference_df.iterrows():
            values = [row[c] for c in batch_cols]
            plt.bar(x + i * width, values, width, label=row["model"], alpha=0.8)

        plt.xlabel("Batch Size")
        plt.ylabel("Inference Time (ms)")
        plt.title("Model Inference Time Comparison")
        labels = [c.replace("batch_", "").replace("_mean_ms", "") for c in batch_cols]
        plt.xticks(x + width, labels)
        plt.legend()
        plt.tight_layout()
        save_path = self.results_dir / "inference_time.png"
        plt.savefig(save_path, dpi=150)
        plt.close()
        logger.info(f"Inference time plot saved to {save_path}")
