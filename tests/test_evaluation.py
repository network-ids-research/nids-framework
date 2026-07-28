import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation import MetricsEvaluator, ZeroDaySimulator


@pytest.fixture
def sample_predictions():
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1, 0, 0])
    y_pred = np.array([0, 1, 1, 1, 0, 1, 0, 0, 0, 0])
    y_prob = np.array([0.1, 0.6, 0.9, 0.8, 0.2, 0.7, 0.3, 0.4, 0.1, 0.2])
    return y_true, y_pred, y_prob


def test_metrics_evaluator_returns_all_metrics(sample_predictions):
    y_true, y_pred, y_prob = sample_predictions
    evaluator = MetricsEvaluator()
    metrics = evaluator.evaluate(y_true, y_pred, y_prob, "test_model")
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "roc_auc" in metrics
    assert "confusion_matrix" in metrics


def test_metrics_evaluator_comparison_table(sample_predictions):
    y_true, y_pred, y_prob = sample_predictions
    evaluator = MetricsEvaluator()
    evaluator.evaluate(y_true, y_pred, y_prob, "model_a")
    evaluator.evaluate(y_true, y_pred, y_prob, "model_b")
    df = evaluator.comparison_table()
    assert len(df) == 2
    assert "Model" in df.columns
    assert "F1-Score" in df.columns


def test_zero_day_simulator_config():
    config = {
        "evaluation": {
            "zero_day": {
                "attack_categories": [
                    "Normal", "Fuzzers", "Analysis", "Backdoors",
                    "DoS", "Exploits", "Generic", "Reconnaissance",
                    "Shellcode", "Worms"
                ]
            }
        }
    }
    sim = ZeroDaySimulator(config)
    assert len(sim.attack_categories) == 10
    assert "Normal" in sim.attack_categories
    assert "Worms" in sim.attack_categories
