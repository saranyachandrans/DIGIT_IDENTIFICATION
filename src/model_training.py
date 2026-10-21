"""Model training pipeline.

Loads the processed .npz, preprocesses, builds the CNN from config,
trains with optional early stopping, evaluates on the test set, and
saves the model, history, metrics and accuracy plot.
"""
import json
import random

import numpy as np

from src.utils import get_logger, resolve


class ModelTraining:
    def __init__(self, config: dict):
        self.cfg = config["training"]
        self.data_cfg = config["data_ingestion"]
        self.seed = config["project"]["seed"]
        self.log = get_logger("ModelTraining", config["project"]["log_level"])

    # ------------------------------------------------------------------
    def _set_seeds(self):
        import tensorflow as tf
        random.seed(self.seed)
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)

    def _load_data(self):
        path = resolve(self.data_cfg["processed_file"])
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run the ingestion stage first."
            )
        d = np.load(path)
        size = self.data_cfg["image_size"]

        def prep(X):
            return (X.astype("float32") / 255.0).reshape(-1, size, size, 1)

        return prep(d["X_train"]), d["y_train"], prep(d["X_test"]), d["y_test"]

    def _train_val_split(self, X, y):
        rng = np.random.default_rng(self.seed)
        idx = rng.permutation(len(X))
        n_val = int(len(X) * self.cfg["validation_split"])
        val_idx, train_idx = idx[:n_val], idx[n_val:]
        return X[train_idx], y[train_idx], X[val_idx], y[val_idx]

    def _build_model(self):
        from tensorflow.keras import Input
        from tensorflow.keras.layers import (
            Conv2D, Dense, Dropout, Flatten, MaxPooling2D,
        )
        from tensorflow.keras.models import Sequential

        m = self.cfg["model"]
        size = self.data_cfg["image_size"]
        layers = [Input(shape=(size, size, 1))]
        for filters in m["conv_filters"]:
            layers.append(Conv2D(filters, (m["kernel_size"],) * 2, activation="relu"))
            layers.append(MaxPooling2D((m["pool_size"],) * 2))
        layers += [
            Flatten(),
            Dense(m["dense_units"], activation="relu"),
            Dropout(m["dropout"]),
            Dense(self.data_cfg["num_classes"], activation="softmax"),
        ]
        return Sequential(layers)

    def _plot(self, history, path):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(history["accuracy"], label="Training Accuracy")
        plt.plot(history["val_accuracy"], label="Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("CNN Accuracy")
        plt.legend()
        plt.savefig(path, bbox_inches="tight")
        plt.close()

    # ------------------------------------------------------------------
    def run(self) -> dict:
        from tensorflow.keras.callbacks import EarlyStopping
        from tensorflow.keras.optimizers import Adam

        self._set_seeds()
        X, y, X_test, y_test = self._load_data()
        X_tr, y_tr, X_val, y_val = self._train_val_split(X, y)
        self.log.info("Train=%s Val=%s Test=%s", X_tr.shape, X_val.shape, X_test.shape)

        model = self._build_model()
        model.compile(
            optimizer=Adam(learning_rate=self.cfg["learning_rate"]),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        model.summary(print_fn=self.log.info)

        callbacks = []
        es = self.cfg["early_stopping"]
        if es["enabled"]:
            callbacks.append(EarlyStopping(
                monitor=es["monitor"], patience=es["patience"],
                restore_best_weights=True,
            ))

        hist = model.fit(
            X_tr, y_tr,
            validation_data=(X_val, y_val),
            epochs=self.cfg["epochs"],
            batch_size=self.cfg["batch_size"],
            callbacks=callbacks,
        )
        history = {k: [float(v) for v in vals] for k, vals in hist.history.items()}

        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        metrics = {"test_loss": float(test_loss), "test_accuracy": float(test_acc)}
        self.log.info("Test loss=%.4f accuracy=%.2f%%", test_loss, test_acc * 100)

        # Save artifacts
        out_dir = resolve(self.cfg["artifacts_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        model.save(resolve(self.cfg["model_path"]))
        with open(resolve(self.cfg["history_file"]), "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        with open(resolve(self.cfg["metrics_file"]), "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        self._plot(history, resolve(self.cfg["plot_file"]))

        self.log.info("Artifacts saved to %s", out_dir)
        return metrics
