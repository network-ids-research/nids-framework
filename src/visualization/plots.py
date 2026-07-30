import logging
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_curve, auc

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

    def zdcs_comparison(self, zdcs_df: pd.DataFrame, zero_day_f1: pd.DataFrame = None):
        categories = zdcs_df["attack_category"].unique()
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax = axes[0]
        order = sorted(categories)
        data = [zdcs_df[zdcs_df["attack_category"] == c]["zdcs"].values for c in order]
        bp = ax.boxplot(data, tick_labels=order, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_alpha(0.6)
        ax.set_title("ZDCS Distribution by Attack Category")
        ax.set_ylabel("Zero-Day Confidence Score")
        ax.set_xlabel("Attack Category")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

        ax = axes[1]
        if zero_day_f1 is not None:
            mean_zdcs = zdcs_df.groupby("attack_category")["zdcs"].mean()
            merged = zero_day_f1.merge(
                mean_zdcs.rename("mean_zdcs").reset_index(),
                left_on="held_out_attack",
                right_on="attack_category",
            )
            ax.scatter(merged["mean_zdcs"], merged["f1"], s=60, c="crimson", alpha=0.8)
            for _, row in merged.iterrows():
                ax.annotate(
                    row["held_out_attack"],
                    (row["mean_zdcs"], row["f1"]),
                    fontsize=8,
                    alpha=0.8,
                )
            r = merged["mean_zdcs"].corr(merged["f1"])
            ax.set_title(f"ZDCS vs Zero-Day F1 (r = {r:.3f})")
            ax.set_xlabel("Mean ZDCS")
            ax.set_ylabel("Zero-Day F1-Score")

        plt.tight_layout()
        path = self.results_dir / "zdcs_by_category.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"ZDCS by category plot saved to {path}")

    def zdcs_roc(self, zdcs_df: pd.DataFrame):
        categories = zdcs_df["attack_category"].unique()
        if len(categories) < 2:
            return
        zdcs_df = zdcs_df.copy()
        cat_counts = zdcs_df["attack_category"].value_counts()
        majority_cat = cat_counts.idxmax()
        zdcs_df["is_hard"] = (zdcs_df["attack_category"] != majority_cat).astype(int)

        fig, ax = plt.subplots(figsize=(7, 6))
        for score_col, label, color in [
            ("zdcs", "ZDCS", "crimson"),
            ("prediction_prob", "Raw Probability", "steelblue"),
        ]:
            fpr, tpr, _ = roc_curve(zdcs_df["is_hard"], zdcs_df[score_col])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f"{label} (AUC = {roc_auc:.3f})", color=color, lw=2)

        ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ZDCS vs Raw Probability for Hard Attack Detection")
        ax.legend(loc="lower right")
        plt.tight_layout()
        path = self.results_dir / "zdcs_roc.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"ZDCS ROC plot saved to {path}")
