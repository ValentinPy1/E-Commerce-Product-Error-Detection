from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from models import TfidfLogRegCleanlab

try:
    from models import SentenceEmbLogReg

    _HAS_SENT = True
except ImportError:
    _HAS_SENT = False

try:
    from models import CamembertLogReg

    _HAS_CAM = True
except ImportError:
    _HAS_CAM = False

try:
    from models import KNNConformity

    _HAS_KNN = True
except ImportError:
    _HAS_KNN = False

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_models(base_dir: str | Path) -> list[tuple[str, object]]:
    """
    Load all available models from artifacts directory.

    Args:
        base_dir: Directory containing saved models

    Returns:
        List of (model_name, model_instance) tuples
    """
    models = []
    path = Path(base_dir).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Artifacts directory not found: {path}")

    tfidf_path = path / "tfidf_logreg_cleanlab"
    if tfidf_path.exists() and tfidf_path.is_dir():
        models.append(
            ("tfidf_logreg_cleanlab", TfidfLogRegCleanlab.load(str(tfidf_path)))
        )

    sent_path = path / "sent_emb_logreg"
    if _HAS_SENT and sent_path.exists() and sent_path.is_dir():
        models.append(
            ("sent_emb_logreg", SentenceEmbLogReg.load(str(sent_path)))
        )

    cam_path = path / "camembert_logreg"
    if _HAS_CAM and cam_path.exists() and cam_path.is_dir():
        models.append(
            ("camembert_logreg", CamembertLogReg.load(str(cam_path)))
        )

    knn_path = path / "knn_conformity"
    if _HAS_KNN and knn_path.exists() and knn_path.is_dir():
        models.append(("knn_conformity", KNNConformity.load(str(knn_path))))

    return models


def main() -> None:
    """Generate ensemble predictions on input CSV file."""
    parser = argparse.ArgumentParser(
        description="Generate predictions using trained ensemble models"
    )
    parser.add_argument("--artifacts", type=str, default="./artifacts")
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--label_column", type=str, default="Libellé produit")
    parser.add_argument(
        "--output_csv", type=str, default="./artifacts/ensemble_predictions.csv"
    )
    args = parser.parse_args()

    models = load_models(args.artifacts)
    if not models:
        raise SystemExit("No saved models found in artifacts directory.")

    logger.info(f"Loaded {len(models)} models: {[name for name, _ in models]}")

    input_path = Path(args.input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    df = pd.read_csv(input_path)
    if args.label_column not in df.columns:
        raise ValueError(f"Label column '{args.label_column}' not found in CSV")

    texts = df[args.label_column].astype(str).tolist()
    logger.info(f"Processing {len(texts)} samples")

    # unify class set
    all_classes = []
    for _, m in models:
        all_classes.extend(m.classes_)
    classes = sorted(list(set(all_classes)))
    idx_map = {c: i for i, c in enumerate(classes)}

    probas_aligned = []
    # store per-model predictions and confidences
    per_model_preds = {}
    per_model_confs = {}
    for name, m in models:
        logger.info(f"Generating predictions with {name}...")
        p = m.predict_proba(texts)
        # per-model predicted class and confidence
        pm_idx = np.argmax(p, axis=1)
        pm_pred = [m.classes_[i] for i in pm_idx]
        pm_conf = np.max(p, axis=1)
        per_model_preds[name] = pm_pred
        per_model_confs[name] = pm_conf
        # align for ensemble averaging
        aligned = np.zeros((len(texts), len(classes)), dtype=np.float64)
        for j, c in enumerate(m.classes_):
            aligned[:, idx_map[c]] = p[:, j]
        probas_aligned.append(aligned)

    avg = np.mean(np.stack(probas_aligned, axis=0), axis=0)
    ensemble_idx = np.argmax(avg, axis=1)
    preds = [classes[i] for i in ensemble_idx]
    ensemble_conf = np.max(avg, axis=1)

    out = df.copy()
    out["ensemble_pred"] = preds
    out["ensemble_conf"] = ensemble_conf
    # append per-model columns
    for name in per_model_preds:
        out[f"{name}_pred"] = per_model_preds[name]
        out[f"{name}_conf"] = per_model_confs[name]

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    logger.info(f"Saved predictions to {output_path}")


if __name__ == '__main__':
    main()


