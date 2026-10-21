"""Data ingestion pipeline.

Downloads EMNIST Letters from Hugging Face, converts images to uint8
grayscale arrays, normalises labels to 0-25, validates, and saves
everything to a single compressed .npz file.
"""
import json

import numpy as np
from datasets import load_dataset

from src.utils import get_logger, resolve


class DataIngestion:
    def __init__(self, config: dict):
        self.cfg = config["data_ingestion"]
        self.log = get_logger("DataIngestion", config["project"]["log_level"])
        self.out_file = resolve(self.cfg["processed_file"])
        self.meta_file = resolve(self.cfg["metadata_file"])

    # ------------------------------------------------------------------
    def _split_to_arrays(self, split):
        """Convert one HF split to (images uint8 [N,H,W], labels int64 [N])."""
        size = self.cfg["image_size"]
        chunks = []
        for batch in split.iter(batch_size=self.cfg["batch_size"]):
            arr = np.stack(
                [np.asarray(img.convert("L"), dtype=np.uint8) for img in batch["image"]]
            )
            chunks.append(arr)
        images = np.concatenate(chunks, axis=0)
        labels = np.asarray(split["label"], dtype=np.int64)

        if self.cfg["fix_orientation"]:
            # EMNIST raw images are transposed; swap H and W to correct it
            images = np.transpose(images, (0, 2, 1))

        # Labels: some copies are 1-26, we always want 0-25
        if labels.min() == 1:
            labels = labels - 1

        self._validate(images, labels, size)
        return images, labels

    def _validate(self, images, labels, size):
        n_classes = self.cfg["num_classes"]
        if images.shape[1:] != (size, size):
            raise ValueError(f"Expected {size}x{size} images, got {images.shape[1:]}")
        if len(images) != len(labels):
            raise ValueError("Image/label count mismatch")
        if labels.min() < 0 or labels.max() >= n_classes:
            raise ValueError(
                f"Labels out of range 0-{n_classes - 1}: "
                f"min={labels.min()}, max={labels.max()}"
            )

    # ------------------------------------------------------------------
    def run(self, force: bool = False) -> str:
        if self.out_file.exists() and self.cfg["skip_if_exists"] and not force:
            self.log.info("Processed data found at %s - skipping ingestion.", self.out_file)
            return str(self.out_file)

        self.log.info("Loading dataset '%s' ...", self.cfg["dataset_name"])
        dataset = load_dataset(self.cfg["dataset_name"])

        self.log.info("Converting train split ...")
        X_train, y_train = self._split_to_arrays(dataset["train"])
        self.log.info("Converting test split ...")
        X_test, y_test = self._split_to_arrays(dataset["test"])

        self.out_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            self.out_file,
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
        )

        metadata = {
            "dataset": self.cfg["dataset_name"],
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "image_shape": list(X_train.shape[1:]),
            "num_classes": self.cfg["num_classes"],
            "label_range": [int(y_train.min()), int(y_train.max())],
            "fix_orientation": self.cfg["fix_orientation"],
        }
        with open(self.meta_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        self.log.info("Saved %s | train=%s test=%s",
                      self.out_file, X_train.shape, X_test.shape)
        return str(self.out_file)
