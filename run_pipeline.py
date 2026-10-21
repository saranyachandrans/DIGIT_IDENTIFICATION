"""Entry point.

Usage:
    python run_pipeline.py --stage ingest
    python run_pipeline.py --stage train
    python run_pipeline.py --stage all [--force] [--config config.yaml]
"""
import argparse

from src.data_ingestion import DataIngestion
from src.model_training import ModelTraining
from src.utils import load_config


def main():
    p = argparse.ArgumentParser(description="EMNIST Letters pipeline")
    p.add_argument("--stage", choices=["ingest", "train", "all"], default="all")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--force", action="store_true",
                   help="re-run ingestion even if processed data exists")
    args = p.parse_args()

    config = load_config(args.config)

    if args.stage in ("ingest", "all"):
        DataIngestion(config).run(force=args.force)
    if args.stage in ("train", "all"):
        ModelTraining(config).run()


if __name__ == "__main__":
    main()
