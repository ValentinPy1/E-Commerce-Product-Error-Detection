from __future__ import annotations

import argparse
import os
import json
import numpy as np
import pandas as pd
from models import TfidfLogRegCleanlab

try:
    from models import SentenceEmbLogReg
    _HAS_SENT = True
except Exception:
    _HAS_SENT = False
try:
    from models import CamembertLogReg
    _HAS_CAM = True
except Exception:
    _HAS_CAM = False
try:
    from models import KNNConformity
    _HAS_KNN = True
except Exception:
    _HAS_KNN = False


def load_models(base_dir: str):
    models = []
    path = os.path.abspath(base_dir)
    if os.path.isdir(os.path.join(path, 'tfidf_logreg_cleanlab')):
        models.append(('tfidf_logreg_cleanlab', TfidfLogRegCleanlab.load(os.path.join(path, 'tfidf_logreg_cleanlab'))))
    if _HAS_SENT and os.path.isdir(os.path.join(path, 'sent_emb_logreg')):
        models.append(('sent_emb_logreg', SentenceEmbLogReg.load(os.path.join(path, 'sent_emb_logreg'))))
    if _HAS_CAM and os.path.isdir(os.path.join(path, 'camembert_logreg')):
        models.append(('camembert_logreg', CamembertLogReg.load(os.path.join(path, 'camembert_logreg'))))
    if _HAS_KNN and os.path.isdir(os.path.join(path, 'knn_conformity')):
        models.append(('knn_conformity', KNNConformity.load(os.path.join(path, 'knn_conformity'))))
    return models


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--artifacts', type=str, default='./artifacts')
    parser.add_argument('--input_csv', type=str, required=True)
    parser.add_argument('--label_column', type=str, default='Libellé produit')
    parser.add_argument('--output_csv', type=str, default='./artifacts/ensemble_predictions.csv')
    args = parser.parse_args()

    models = load_models(args.artifacts)
    if not models:
        raise SystemExit('No saved models found in artifacts directory.')

    df = pd.read_csv(args.input_csv)
    texts = df[args.label_column].astype(str).tolist()

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
    out['ensemble_pred'] = preds
    out['ensemble_conf'] = ensemble_conf
    # append per-model columns
    for name in per_model_preds:
        out[f'{name}_pred'] = per_model_preds[name]
        out[f'{name}_conf'] = per_model_confs[name]
    out.to_csv(args.output_csv, index=False)
    print('Saved predictions to', args.output_csv)


if __name__ == '__main__':
    main()


