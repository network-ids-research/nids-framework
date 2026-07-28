import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import RandomForestTrainer, XGBoostTrainer, ModelPredictor


@pytest.fixture
def sample_data():
    np.random.seed(42)
    n = 200
    X = pd.DataFrame({
        f"feat_{i}": np.random.randn(n) for i in range(10)
    })
    y = pd.Series(np.random.randint(0, 2, n))
    return X, y


@pytest.fixture
def model_config():
    return {
        "models": {
            "random_forest": {
                "enabled": True,
                "n_trials": 2,
                "save_path": "tests/fixtures/rf_test.pkl"
            },
            "xgboost": {
                "enabled": True,
                "n_trials": 2,
                "save_path": "tests/fixtures/xgb_test.pkl"
            },
            "deep_learning": {
                "enabled": False,
                "n_trials": 2,
                "save_path": "tests/fixtures/dl_test.keras"
            }
        },
        "evaluation": {
            "test_size": 0.2,
            "random_state": 42
        }
    }


def test_random_forest_trains(model_config, sample_data):
    X, y = sample_data
    trainer = RandomForestTrainer(model_config)
    model, params, f1 = trainer.train(X, y)
    assert model is not None
    assert params is not None
    assert 0 <= f1 <= 1


def test_random_forest_predicts(model_config, sample_data):
    X, y = sample_data
    trainer = RandomForestTrainer(model_config)
    model, _, _ = trainer.train(X, y)
    preds = model.predict(X[:5])
    assert len(preds) == 5
    assert set(preds).issubset({0, 1})


def test_xgboost_trains(model_config, sample_data):
    X, y = sample_data
    trainer = XGBoostTrainer(model_config)
    model, params, f1 = trainer.train(X, y)
    assert model is not None
    assert params is not None
    assert 0 <= f1 <= 1


def test_xgboost_predicts(model_config, sample_data):
    X, y = sample_data
    trainer = XGBoostTrainer(model_config)
    model, _, _ = trainer.train(X, y)
    preds = model.predict(X[:5])
    assert len(preds) == 5
    assert set(preds).issubset({0, 1})


def test_model_predictor_sklearn(model_config, sample_data):
    X, y = sample_data
    trainer = RandomForestTrainer(model_config)
    model, _, _ = trainer.train(X, y)

    predictor = ModelPredictor()
    predictor.models["rf"] = model
    preds, probs = predictor.predict("rf", X[:5])
    assert len(preds) == 5
    assert probs is not None
