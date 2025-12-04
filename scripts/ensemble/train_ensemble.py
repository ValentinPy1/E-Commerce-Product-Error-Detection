from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd

from ensemble import EnsembleConfig, EnsemblePipeline

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Train ensemble models on provided data."""
    parser = argparse.ArgumentParser(
        description="Train ensemble text classification models"
    )
    parser.add_argument(
        "--data", type=str, default="data/unique_products.csv"
    )
    parser.add_argument("--label_column", type=str, default="Libellé produit")
    parser.add_argument("--target_column", type=str, default="Nature")
    parser.add_argument("--tfidf", action="store_true")
    parser.add_argument("--sent", action="store_true")
    parser.add_argument("--camembert", action="store_true")
    parser.add_argument("--knn", action="store_true")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    # Set random seeds for reproducibility
    random.seed(args.random_state)
    np.random.seed(args.random_state)

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path)
    if args.label_column not in df.columns:
        raise ValueError(f"Label column '{args.label_column}' not found in data")
    if args.target_column not in df.columns:
        raise ValueError(
            f"Target column '{args.target_column}' not found in data"
        )

    df = df[[args.label_column, args.target_column]].dropna()
    df = df[
        (df[args.label_column].astype(str).str.strip() != "")
        & (df[args.target_column].astype(str).str.strip() != "")
    ]
    X = df[args.label_column].astype(str).tolist()
    y = df[args.target_column].astype(str).tolist()

    logger.info(f"Loaded {len(X)} samples from {data_path}")

    cfg = EnsembleConfig(
        use_tfidf=args.tfidf
        or (not args.sent and not args.camembert and not args.knn),
        use_sent_emb=args.sent,
        use_camembert=args.camembert,
        use_knn=args.knn,
        max_samples=args.max_samples,
        random_state=args.random_state,
    )
    pipeline = EnsemblePipeline(cfg)
    X_train, X_test, y_train, y_test = pipeline.split(X, y)
    logger.info(
        f"Split data: {len(X_train)} train, {len(X_test)} test samples"
    )
    pipeline.fit(X_train, y_train)
    metrics = pipeline.evaluate(X_test, y_test)
    logger.info("Evaluation metrics:")
    for key, value in metrics.items():
        logger.info(f"  {key}: {value:.4f}")


if __name__ == '__main__':
    main()


