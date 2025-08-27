from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Sequence as Seq
import time
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import train_test_split
import os

from models import TfidfLogRegCleanlab
try:
    from models import SentenceEmbLogReg
    _HAS_SENT = True
except Exception:
    _HAS_SENT = False
try:
    from models import CamembertLogReg
    _HAS_CAMEMBERT = True
except Exception:
    _HAS_CAMEMBERT = False
try:
    from models import KNNConformity
    _HAS_KNN = True
except Exception:
    _HAS_KNN = False


@dataclass
class EnsembleConfig:
    test_size: float = 0.1
    random_state: int = 42
    use_tfidf: bool = True
    use_sent_emb: bool = False
    use_camembert: bool = False
    use_knn: bool = False
    max_samples: int | None = None


class EnsemblePipeline:
    def __init__(self, config: EnsembleConfig | None = None):
        self.config = config or EnsembleConfig()
        self.models = []
        if self.config.use_tfidf:
            # Use SGD backend by default for lower memory footprint
            self.models.append(("tfidf_logreg_cleanlab", TfidfLogRegCleanlab(backend='sgd', verbose=1)))
        if self.config.use_sent_emb and _HAS_SENT:
            self.models.append(("sent_emb_logreg", SentenceEmbLogReg()))
        if self.config.use_camembert and _HAS_CAMEMBERT:
            self.models.append(("camembert_logreg", CamembertLogReg()))
        if self.config.use_knn and _HAS_KNN:
            self.models.append(("knn_conformity", KNNConformity()))
        self.classes_: List[str] | None = None
        self.save_dir = os.path.join(os.getcwd(), 'artifacts')

    def split(self, texts: Seq[str], labels: Seq[str]):
        X = np.array(list(map(str, texts)))
        y = np.array(list(map(str, labels)))
        if self.config.max_samples is not None and self.config.max_samples < len(X):
            np.random.seed(self.config.random_state)
            idx = np.random.permutation(len(X))[: self.config.max_samples]
            X = X[idx]
            y = y[idx]
        # Stratify on classes with >=2 samples; put rare classes into train
        counts = pd.Series(y).value_counts()
        freq_classes = counts[counts >= 2].index
        frequent_mask = np.isin(y, freq_classes)
        idx_frequent = np.where(frequent_mask)[0]
        idx_rare = np.where(~frequent_mask)[0]

        if len(idx_frequent) >= 2:
            X_f, y_f = X[idx_frequent], y[idx_frequent]
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_f, y_f,
                    test_size=self.config.test_size,
                    random_state=self.config.random_state,
                    stratify=y_f,
                )
            except ValueError:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_f, y_f,
                    test_size=self.config.test_size,
                    random_state=self.config.random_state,
                    shuffle=True,
                    stratify=None,
                )
            # Append rare to training
            if len(idx_rare) > 0:
                X_train = np.concatenate([X_train, X[idx_rare]])
                y_train = np.concatenate([y_train, y[idx_rare]])
            return X_train.tolist(), X_test.tolist(), y_train.tolist(), y_test.tolist()
        # Fallback no stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            shuffle=True,
            stratify=None,
        )
        return X_train.tolist(), X_test.tolist(), y_train.tolist(), y_test.tolist()

    def fit(self, X_train: Seq[str], y_train: Seq[str]):
        for name, model in self.models:
            t0 = time.time()
            print(f"[Ensemble] Fitting {name} on {len(X_train)} samples...")
            try:
                model.fit(X_train, y_train)
            finally:
                dt = time.time()-t0
                eta = dt * (len(self.models))  # crude ETA for full suite
                print(f"[Ensemble] {name} done in {dt:.1f}s (est total ~{eta:.1f}s)")
            # Save after fit
            out_dir = os.path.join(self.save_dir, name)
            model.save(out_dir)
            print(f"[Ensemble] Saved {name} to {out_dir}")
        # unify classes across models by union
        all_classes = []
        for _, m in self.models:
            all_classes.extend(m.classes_)
        self.classes_ = sorted(list(set(all_classes)))

    def _align_proba(self, model, proba: np.ndarray) -> np.ndarray:
        assert self.classes_ is not None
        aligned = np.zeros((proba.shape[0], len(self.classes_)), dtype=np.float64)
        idx_map = {c: i for i, c in enumerate(self.classes_)}
        for j, c in enumerate(model.classes_):
            aligned[:, idx_map[c]] = proba[:, j]
        return aligned

    def predict_proba_all(self, X: Seq[str]) -> Dict[str, np.ndarray]:
        outputs: Dict[str, np.ndarray] = {}
        for name, model in self.models:
            t0 = time.time()
            print(f"[Ensemble] Predict proba {name} on {len(X)} samples...")
            try:
                p = model.predict_proba(X)
            finally:
                dt = time.time()-t0
                print(f"[Ensemble] {name} proba done in {dt:.1f}s")
            outputs[name] = self._align_proba(model, p)
        return outputs

    def majority_vote(self, X: Seq[str]) -> np.ndarray:
        assert self.classes_ is not None
        preds = []
        for _, model in self.models:
            preds.append(model.predict(X))
        preds = np.vstack([np.array(p) for p in preds])  # (M, N)
        # majority per column
        voted = []
        for col in preds.T:
            values, counts = np.unique(col, return_counts=True)
            voted.append(values[np.argmax(counts)])
        return np.array(voted)

    def avg_prob(self, X: Seq[str]) -> np.ndarray:
        aligned = self.predict_proba_all(X)
        stacked = np.stack(list(aligned.values()), axis=0)  # (M, N, C)
        return stacked.mean(axis=0)

    def evaluate(self, X_test: Seq[str], y_test: Seq[str]) -> Dict[str, float]:
        # majority vote
        y_pred_vote = self.majority_vote(X_test)
        acc_vote = accuracy_score(y_test, y_pred_vote)
        prec_vote = precision_score(y_test, y_pred_vote, average='weighted', zero_division=0)
        # avg prob
        proba = self.avg_prob(X_test)
        y_pred_avg = np.array([self.classes_[i] for i in np.argmax(proba, axis=1)])
        acc_avg = accuracy_score(y_test, y_pred_avg)
        prec_avg = precision_score(y_test, y_pred_avg, average='weighted', zero_division=0)
        return {
            'majority_vote_accuracy': float(acc_vote),
            'majority_vote_precision_weighted': float(prec_vote),
            'avg_prob_accuracy': float(acc_avg),
            'avg_prob_precision_weighted': float(prec_avg),
        }


