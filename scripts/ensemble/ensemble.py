from __future__ import annotations

import logging
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Sequence as Seq

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import train_test_split

from models import BaseModel, TfidfLogRegCleanlab

try:
    from models import SentenceEmbLogReg

    _HAS_SENT = True
except ImportError:
    _HAS_SENT = False

try:
    from models import CamembertLogReg

    _HAS_CAMEMBERT = True
except ImportError:
    _HAS_CAMEMBERT = False

try:
    from models import KNNConformity

    _HAS_KNN = True
except ImportError:
    _HAS_KNN = False

logger = logging.getLogger(__name__)


@dataclass
class EnsembleConfig:
    """
    Configuration for ensemble pipeline.

    Args:
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility
        use_tfidf: Whether to include TF-IDF model
        use_sent_emb: Whether to include sentence embeddings model
        use_camembert: Whether to include CamemBERT model
        use_knn: Whether to include KNN model
        max_samples: Maximum number of samples to use (None = all)
    """

    test_size: float = 0.1
    random_state: int = 42
    use_tfidf: bool = True
    use_sent_emb: bool = False
    use_camembert: bool = False
    use_knn: bool = False
    max_samples: int | None = None


class EnsemblePipeline:
    """
    Ensemble pipeline combining multiple text classification models.

    Supports TF-IDF, sentence embeddings, CamemBERT, and KNN models.
    Combines predictions using probability averaging or majority voting.
    """

    def __init__(
        self,
        config: EnsembleConfig | None = None,
        save_dir: str | Path | None = None,
    ):
        """
        Initialize ensemble pipeline.

        Args:
            config: Ensemble configuration
            save_dir: Directory to save trained models (default: ./artifacts)
        """
        self.config = config or EnsembleConfig()
        self.models = []
        if self.config.use_tfidf:
            # Use SGD backend by default for lower memory footprint
            self.models.append(
                (
                    "tfidf_logreg_cleanlab",
                    TfidfLogRegCleanlab(backend="sgd", verbose=1),
                )
            )
        if self.config.use_sent_emb and _HAS_SENT:
            self.models.append(("sent_emb_logreg", SentenceEmbLogReg()))
        if self.config.use_camembert and _HAS_CAMEMBERT:
            self.models.append(("camembert_logreg", CamembertLogReg()))
        if self.config.use_knn and _HAS_KNN:
            self.models.append(("knn_conformity", KNNConformity()))
        self.classes_: List[str] | None = None
        if save_dir is None:
            save_dir = Path.cwd() / "artifacts"
        self.save_dir = Path(save_dir)

    def split(
        self, texts: Seq[str], labels: Seq[str]
    ) -> tuple[list[str], list[str], list[str], list[str]]:
        """
        Split data into train and test sets with stratification.

        Handles rare classes (with <2 samples) by putting them in training set.

        Args:
            texts: Text samples
            labels: Corresponding labels

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        X = np.array(list(map(str, texts)))
        y = np.array(list(map(str, labels)))
        if self.config.max_samples is not None and self.config.max_samples < len(X):
            rng = np.random.RandomState(self.config.random_state)
            idx = rng.permutation(len(X))[: self.config.max_samples]
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
                    X_f,
                    y_f,
                    test_size=self.config.test_size,
                    random_state=self.config.random_state,
                    stratify=y_f,
                )
            except ValueError:
                logger.warning(
                    "Stratification failed, falling back to random split"
                )
                X_train, X_test, y_train, y_test = train_test_split(
                    X_f,
                    y_f,
                    test_size=self.config.test_size,
                    random_state=self.config.random_state,
                    shuffle=True,
                    stratify=None,
                )
            # Append rare to training
            if len(idx_rare) > 0:
                X_train = np.concatenate([X_train, X[idx_rare]])
                y_train = np.concatenate([y_train, y[idx_rare]])
            return (
                X_train.tolist(),
                X_test.tolist(),
                y_train.tolist(),
                y_test.tolist(),
            )
        # Fallback no stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            shuffle=True,
            stratify=None,
        )
        return (
            X_train.tolist(),
            X_test.tolist(),
            y_train.tolist(),
            y_test.tolist(),
        )

    def fit(self, X_train: Seq[str], y_train: Seq[str]) -> None:
        """
        Train all models in the ensemble.

        Args:
            X_train: Training text samples
            y_train: Corresponding labels
        """
        for name, model in self.models:
            t0 = time.time()
            logger.info(f"Fitting {name} on {len(X_train)} samples...")
            try:
                model.fit(X_train, y_train)
            finally:
                dt = time.time() - t0
                eta = dt * (len(self.models))  # crude ETA for full suite
                logger.info(
                    f"{name} done in {dt:.1f}s (est total ~{eta:.1f}s)"
                )
            # Save after fit
            out_dir = self.save_dir / name
            model.save(str(out_dir))
            logger.info(f"Saved {name} to {out_dir}")
        # unify classes across models by union
        all_classes = []
        for _, m in self.models:
            all_classes.extend(m.classes_)
        self.classes_ = sorted(list(set(all_classes)))

    def _align_proba(self, model: BaseModel, proba: np.ndarray) -> np.ndarray:
        """
        Align model probabilities to unified class space.

        Args:
            model: Model instance
            proba: Probability array from model

        Returns:
            Aligned probability array
        """
        if self.classes_ is None:
            raise ValueError("Classes not set. Call fit() first.")
        aligned = np.zeros(
            (proba.shape[0], len(self.classes_)), dtype=np.float64
        )
        idx_map = {c: i for i, c in enumerate(self.classes_)}
        for j, c in enumerate(model.classes_):
            aligned[:, idx_map[c]] = proba[:, j]
        return aligned

    def predict_proba_all(self, X: Seq[str]) -> Dict[str, np.ndarray]:
        """
        Get probability predictions from all models.

        Args:
            X: Text samples to predict

        Returns:
            Dictionary mapping model names to probability arrays
        """
        outputs: Dict[str, np.ndarray] = {}
        for name, model in self.models:
            t0 = time.time()
            logger.info(f"Predict proba {name} on {len(X)} samples...")
            try:
                p = model.predict_proba(X)
            finally:
                dt = time.time() - t0
                logger.info(f"{name} proba done in {dt:.1f}s")
            outputs[name] = self._align_proba(model, p)
        return outputs

    def majority_vote(self, X: Seq[str]) -> np.ndarray:
        """
        Predict using majority voting across models.

        Args:
            X: Text samples to predict

        Returns:
            Array of predicted labels
        """
        if self.classes_ is None:
            raise ValueError("Classes not set. Call fit() first.")
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
        """
        Predict using averaged probabilities across models.

        Args:
            X: Text samples to predict

        Returns:
            Array of averaged probabilities
        """
        aligned = self.predict_proba_all(X)
        stacked = np.stack(list(aligned.values()), axis=0)  # (M, N, C)
        return stacked.mean(axis=0)

    def evaluate(
        self, X_test: Seq[str], y_test: Seq[str]
    ) -> Dict[str, float]:
        """
        Evaluate ensemble performance on test set.

        Args:
            X_test: Test text samples
            y_test: True labels

        Returns:
            Dictionary with evaluation metrics
        """
        if self.classes_ is None:
            raise ValueError("Classes not set. Call fit() first.")
        # majority vote
        y_pred_vote = self.majority_vote(X_test)
        acc_vote = accuracy_score(y_test, y_pred_vote)
        prec_vote = precision_score(
            y_test, y_pred_vote, average="weighted", zero_division=0
        )
        # avg prob
        proba = self.avg_prob(X_test)
        y_pred_avg = np.array(
            [self.classes_[i] for i in np.argmax(proba, axis=1)]
        )
        acc_avg = accuracy_score(y_test, y_pred_avg)
        prec_avg = precision_score(
            y_test, y_pred_avg, average="weighted", zero_division=0
        )
        return {
            "majority_vote_accuracy": float(acc_vote),
            "majority_vote_precision_weighted": float(prec_vote),
            "avg_prob_accuracy": float(acc_avg),
            "avg_prob_precision_weighted": float(prec_avg),
        }


