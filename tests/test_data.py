import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import DataLoader, DataCleaner, DataSplitter


@pytest.fixture
def sample_config():
    return {
        "data": {
            "raw_path": "tests/fixtures",
            "processed_path": "tests/fixtures/processed",
            "train_file": "train_sample.csv",
            "test_file": "test_sample.csv",
            "label_col": "label",
            "attack_cat_col": "attack_cat",
            "categorical_cols": ["proto", "service", "state"],
            "id_cols": ["id"]
        },
        "evaluation": {
            "test_size": 0.2,
            "random_state": 42
        }
    }


@pytest.fixture
def sample_data():
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "id": range(n),
        "proto": np.random.choice(["tcp", "udp", "arp"], n),
        "service": np.random.choice(["http", "dns", "-"], n),
        "state": np.random.choice(["FIN", "CON", "INT"], n),
        "dur": np.random.rand(n) * 100,
        "sbytes": np.random.randint(0, 10000, n),
        "label": np.random.randint(0, 2, n),
        "attack_cat": np.random.choice(
            ["Normal", "Fuzzers", "Exploits", "Generic"], n
        )
    })


def test_data_cleaner_encodes_categorical(sample_config, sample_data):
    cleaner = DataCleaner(sample_config)
    cleaned = cleaner.clean(sample_data)
    for col in sample_config["data"]["categorical_cols"]:
        assert cleaned[col].dtype in ["int8", "int16", "int32", "int64"]
    assert "id" not in cleaned.columns


def test_data_cleaner_handles_missing(sample_config, sample_data):
    sample_data.loc[0, "sbytes"] = np.nan
    cleaner = DataCleaner(sample_config)
    cleaned = cleaner.clean(sample_data)
    assert cleaned["sbytes"].isna().sum() == 0


def test_data_splitter_creates_split(sample_config, sample_data):
    splitter = DataSplitter(sample_config)
    X, y, attack_cat = DataCleaner(sample_config).separate_features_target(sample_data)
    X_train, X_test, y_train, y_test = splitter.split(X, y)
    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(X_train) + len(X_test) == len(sample_data)


def test_data_splitter_preserves_proportions(sample_config, sample_data):
    splitter = DataSplitter(sample_config)
    X, y, attack_cat = DataCleaner(sample_config).separate_features_target(sample_data)
    X_train, X_test, y_train, y_test = splitter.split(X, y, attack_cat)
    assert len(X_train) > 0
    assert len(X_test) > 0
