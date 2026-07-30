import logging
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)

DIR = Path("data/interim")
DIR.mkdir(parents=True, exist_ok=True)

def save_phase1(X_train, y_train, X_test, y_test, attack_cat_test, train_df, test_df):
    X_train.to_parquet(DIR / "X_train.parquet")
    y_train.to_frame("label").to_parquet(DIR / "y_train.parquet")
    X_test.to_parquet(DIR / "X_test.parquet")
    y_test.to_frame("label").to_parquet(DIR / "y_test.parquet")
    attack_cat_test.to_frame("attack_cat").to_parquet(DIR / "attack_cat_test.parquet")
    train_df.to_parquet(DIR / "train_df.parquet")
    test_df.to_parquet(DIR / "test_df.parquet")
    logger.info("Phase 1 checkpoint saved to data/interim/")

def load_phase1():
    X_train = pd.read_parquet(DIR / "X_train.parquet")
    y_train = pd.read_parquet(DIR / "y_train.parquet")["label"]
    X_test = pd.read_parquet(DIR / "X_test.parquet")
    y_test = pd.read_parquet(DIR / "y_test.parquet")["label"]
    attack_cat_test = pd.read_parquet(DIR / "attack_cat_test.parquet")["attack_cat"]
    train_df = pd.read_parquet(DIR / "train_df.parquet")
    test_df = pd.read_parquet(DIR / "test_df.parquet")
    logger.info("Phase 1 checkpoint loaded from data/interim/")
    return X_train, y_train, X_test, y_test, attack_cat_test, train_df, test_df

def checkpoint_exists():
    return (DIR / "X_train.parquet").exists()

def parse_phase(value: str | None):
    if value is None:
        return None
    value = value.strip()
    if "-" in value:
        parts = value.split("-")
        return list(range(int(parts[0]), int(parts[1]) + 1))
    parts = [p.strip() for p in value.split(",")]
    result = []
    for p in parts:
        if p == "5.5":
            result.append(5.5)
        else:
            result.append(int(p))
    return sorted(result)
