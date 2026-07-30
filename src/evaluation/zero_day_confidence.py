import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_curve, auc

logger = logging.getLogger(__name__)


class ZeroDayConfidenceScore:
    """Zero-Day Confidence Score (ZDCS).

    Combines prediction entropy and SHAP explanation novelty to quantify
    how "unusual" a sample appears relative to known attacks.

    ZDCS = alpha * H(p) + (1-alpha) * D(shap, ref)

    Higher ZDCS = more likely to be a zero-day / novel attack.
    """

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        self.ref_shap_mean_ = None

    def _extract_positive(self, shap_values):
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        if hasattr(shap_values, "ndim") and shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]
        return shap_values

    def _normalized_entropy(self, proba: float) -> float:
        p = np.clip(proba, 1e-10, 1 - 1e-10)
        return float(-(p * np.log(p) + (1 - p) * np.log(1 - p)) / np.log(2))

    def _cosine_distance(self, shap_vector: np.ndarray) -> float:
        sim = cosine_similarity(
            shap_vector.reshape(1, -1), self.ref_shap_mean_.reshape(1, -1)
        )[0, 0]
        return float(1 - sim)

    def fit(self, explainer, X_attacks: pd.DataFrame):
        shap_values = explainer.explainer.shap_values(X_attacks)
        shap_values = self._extract_positive(shap_values)
        self.ref_shap_mean_ = shap_values.mean(axis=0)
        logger.info(
            "Reference SHAP signature computed from %d known attack samples",
            len(X_attacks),
        )
        return self

    def compute(self, model, explainer, X: pd.DataFrame) -> pd.DataFrame:
        proba = model.predict_proba(X)[:, 1]
        shap_values = explainer.explainer.shap_values(X)
        shap_values = self._extract_positive(shap_values)

        rows = []
        for i in range(len(X)):
            entropy = self._normalized_entropy(proba[i])
            shap_dist = self._cosine_distance(shap_values[i])
            zdcs = self.alpha * entropy + (1 - self.alpha) * shap_dist
            rows.append(
                {
                    "entropy": entropy,
                    "shap_distance": shap_dist,
                    "zdcs": zdcs,
                    "prediction_prob": proba[i],
                }
            )
        return pd.DataFrame(rows, index=X.index)

    def evaluate_by_category(
        self,
        model,
        explainer,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        attack_cat_test: pd.Series,
        max_ref: int = 300,
        max_per_cat: int = 100,
    ) -> pd.DataFrame:
        train_attacks = X_train[y_train == 1]
        train_sample = train_attacks.sample(
            n=min(max_ref, len(train_attacks)), random_state=42
        )
        logger.info("Fitting ZDCS reference from %d training attack samples...", len(train_sample))
        self.fit(explainer, train_sample)

        all_dfs = []
        for cat in sorted(attack_cat_test.unique()):
            if cat == "Normal" or pd.isna(cat):
                continue
            mask = attack_cat_test == cat
            cat_X = X_test[mask]
            if len(cat_X) == 0:
                continue
            if len(cat_X) > max_per_cat:
                cat_X = cat_X.sample(n=max_per_cat, random_state=42)
            logger.info("Computing ZDCS for %s (%d samples)...", cat, len(cat_X))
            df = self.compute(model, explainer, cat_X)
            df["attack_category"] = cat
            all_dfs.append(df)

        return pd.concat(all_dfs, ignore_index=True)


def plot_zdcs_by_category(
    zdcs_df: pd.DataFrame,
    zero_day_f1: pd.DataFrame = None,
    save_dir: str = "results/figures",
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    categories = zdcs_df["attack_category"].unique()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    order = sorted(categories)
    data = [zdcs_df[zdcs_df["attack_category"] == c]["zdcs"].values for c in order]
    bp = ax.boxplot(data, labels=order, patch_artist=True)
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
    else:
        mean_zdcs = zdcs_df.groupby("attack_category")["zdcs"].mean().sort_values()
        ax.barh(range(len(mean_zdcs)), mean_zdcs.values, color="steelblue", alpha=0.7)
        ax.set_yticks(range(len(mean_zdcs)))
        ax.set_yticklabels(mean_zdcs.index)
        ax.set_title("Mean ZDCS by Attack Category")
        ax.set_xlabel("Mean ZDCS")

    plt.tight_layout()
    path = save_dir / "zdcs_by_category.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("ZDCS by category plot saved to %s", path)


def plot_zdcs_roc(
    zdcs_df: pd.DataFrame,
    save_dir: str = "results/figures",
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

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
    path = save_dir / "zdcs_roc.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("ZDCS ROC plot saved to %s", path)
