import optuna
import logging
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

logger = logging.getLogger(__name__)


class DeepLearningTrainer:
    def __init__(self, config: dict):
        self.config = config
        self.model_config = config["models"]["deep_learning"]
        self.save_path = Path(self.model_config["save_path"])
        self.best_model = None
        self.input_dim = None
        self._tf = None

    @property
    def tf(self):
        if self._tf is None:
            import tensorflow as _tf
            _tf.get_logger().setLevel("ERROR")
            self._tf = _tf
        return self._tf

    def _build_model(self, trial, input_dim):
        tf = self.tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
        from tensorflow.keras.optimizers import Adam

        n_layers = trial.suggest_int("n_layers", 1, 3)
        units = trial.suggest_categorical("units", [64, 128, 256])
        dropout = trial.suggest_float("dropout", 0.2, 0.5)
        lr = trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True)

        model = Sequential()
        model.add(Dense(units, activation="relu", input_shape=(input_dim,)))
        model.add(BatchNormalization())
        model.add(Dropout(dropout))

        for i in range(n_layers - 1):
            units_i = max(units // (2 ** (i + 1)), 16)
            model.add(Dense(units_i, activation="relu"))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))

        model.add(Dense(1, activation="sigmoid"))
        model.compile(
            optimizer=Adam(learning_rate=lr),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        return model

    def _objective(self, trial, X_train, y_train, X_val, y_val):
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

        model = self._build_model(trial, X_train.shape[1])

        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=self.model_config["early_stop_patience"],
            restore_best_weights=True
        )
        reduce_lr = ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6
        )

        batch_size = trial.suggest_categorical("batch_size", self.model_config["batch_size"])
        epochs = self.model_config["epochs"]

        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, reduce_lr],
            verbose=0
        )

        y_pred = (model.predict(X_val, verbose=0) > 0.5).astype(int)
        return f1_score(y_val, y_pred)

    def train(self, X_train, y_train, X_val=None, y_val=None):
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

        logger.info("Training Deep Learning model with Optuna...")
        self.input_dim = X_train.shape[1]

        if X_val is None or y_val is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42
            )

        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: self._objective(trial, X_train, y_train, X_val, y_val),
            n_trials=self.model_config["n_trials"]
        )

        logger.info(f"Best DL params: {study.best_params}")
        logger.info(f"Best DL F1: {study.best_value:.4f}")

        best_params = study.best_params
        self.best_model = self._build_model(study.best_trial, self.input_dim)

        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=self.model_config["early_stop_patience"],
            restore_best_weights=True
        )
        reduce_lr = ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6
        )

        self.best_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.model_config["epochs"],
            batch_size=best_params.get("batch_size", 64),
            callbacks=[early_stop, reduce_lr],
            verbose=0
        )

        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        self.best_model.save(self.save_path)
        logger.info(f"DL model saved to {self.save_path}")

        return self.best_model, study.best_params, study.best_value
