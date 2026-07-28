import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class DataLoader:
    def __init__(self, config: dict):
        self.raw_path = Path(config["data"]["raw_path"])
        self.processed_path = Path(config["data"]["processed_path"])
        self.train_file = config["data"]["train_file"]
        self.test_file = config["data"]["test_file"]
        self.label_col = config["data"]["label_col"]
        self.attack_cat_col = config["data"]["attack_cat_col"]
        self.categorical_cols = config["data"]["categorical_cols"]
        self.id_cols = config["data"]["id_cols"]

        self.processed_path.mkdir(parents=True, exist_ok=True)

    def load_raw(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train_path = self.raw_path / self.train_file
        test_path = self.raw_path / self.test_file

        if not train_path.exists():
            raise FileNotFoundError(f"Training file not found: {train_path}")
        if not test_path.exists():
            raise FileNotFoundError(f"Testing file not found: {test_path}")

        logger.info(f"Loading training data from {train_path}")
        train_df = pd.read_csv(train_path, low_memory=False)

        logger.info(f"Loading testing data from {test_path}")
        test_df = pd.read_csv(test_path, low_memory=False)

        logger.info(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
        return train_df, test_df

    def merge_partitions(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
        train_df["_partition"] = "train"
        test_df["_partition"] = "test"
        df = pd.concat([train_df, test_df], ignore_index=True)
        logger.info(f"Merged dataset shape: {df.shape}")
        return df


class DataCleaner:
    def __init__(self, config: dict):
        self.categorical_cols = config["data"]["categorical_cols"]
        self.id_cols = config["data"]["id_cols"]
        self.label_col = config["data"]["label_col"]
        self.attack_cat_col = config["data"]["attack_cat_col"]

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df = self._drop_id_columns(df)
        df = self._handle_missing_values(df)
        df = self._encode_categorical(df)

        return df

    def _drop_id_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        cols_to_drop = [c for c in self.id_cols if c in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            logger.info(f"Dropped ID columns: {cols_to_drop}")
        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        initial_shape = df.shape
        df = df.dropna(how="all")
        for col in df.columns:
            if df[col].isna().sum() > 0:
                if df[col].dtype in ["float64", "int64"]:
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else "unknown")
        dropped = initial_shape[0] - df.shape[0]
        if dropped > 0:
            logger.info(f"Dropped {dropped} fully empty rows")
        return df

    def _encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self.categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype("category").cat.codes
                logger.info(f"Encoded categorical column: {col}")
        return df

    def separate_features_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.Series]]:
        X = df.drop(columns=[self.label_col, self.attack_cat_col], errors="ignore")
        y = df[self.label_col].astype(int) if self.label_col in df.columns else None
        attack_cat = df[self.attack_cat_col] if self.attack_cat_col in df.columns else None
        return X, y, attack_cat


class DataSplitter:
    def __init__(self, config: dict):
        self.test_size = config["evaluation"]["test_size"]
        self.random_state = config["evaluation"]["random_state"]

    def split(self, X: pd.DataFrame, y: pd.Series, attack_cat: pd.Series = None):
        from sklearn.model_selection import train_test_split

        if attack_cat is not None:
            stratify = attack_cat
        else:
            stratify = y

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify
        )
        logger.info(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
        return X_train, X_test, y_train, y_test

    def split_by_partition(self, df: pd.DataFrame):
        train_df = df[df["_partition"] == "train"].drop(columns=["_partition"])
        test_df = df[df["_partition"] == "test"].drop(columns=["_partition"])
        logger.info(f"Partition split - Train: {train_df.shape}, Test: {test_df.shape}")
        return train_df, test_df
