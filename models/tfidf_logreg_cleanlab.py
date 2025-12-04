from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Sequence as Seq, Optional

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.preprocessing import LabelEncoder

try:
    from cleanlab.classification import CleanLearning

    CLEANLAB_AVAILABLE = True
except ImportError:
    CLEANLAB_AVAILABLE = False

from .base import BaseModel

logger = logging.getLogger(__name__)


class TfidfLogRegCleanlab(BaseModel):
    """
    TF-IDF vectorization with Logistic Regression classifier.

    Uses character-level n-grams for feature extraction. Supports both
    SGD and Logistic Regression backends, with optional CleanLab integration
    for handling label noise.

    Args:
        max_features: Maximum number of features for TF-IDF vectorizer
        c: Regularization strength (inverse of C)
        n_jobs: Number of parallel jobs (unused for SGD backend)
        use_cleanlab: Whether to use CleanLab for label noise handling
        solver: Solver for Logistic Regression backend
        verbose: Verbosity level (0 = silent, 1 = info)
        backend: Classifier backend ('sgd' or 'logreg')
    """

    def __init__(
        self,
        max_features: int = 100000,
        c: float = 4.0,
        n_jobs: int = -1,
        use_cleanlab: bool = False,
        solver: str = "lbfgs",
        verbose: int = 0,
        backend: str = "sgd",  # 'sgd' or 'logreg'
    ):
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 3), analyzer="char", max_features=max_features
        )
        self.backend = backend
        if backend == "logreg":
            # Use single-thread to avoid OOM due to parallel jobs across many classes
            self.clf = LogisticRegression(
                max_iter=200,
                C=c,
                n_jobs=1,
                multi_class="auto",
                solver=solver,
                verbose=verbose,
            )
        else:
            # Memory-lean, scalable classifier with probabilities
            self.clf = SGDClassifier(
                loss="log_loss",
                alpha=1e-5,
                max_iter=20,
                tol=1e-3,
                early_stopping=True,
                validation_fraction=0.1,
            )
        self.le = LabelEncoder()
        self.classes_: list[str] | None = None
        self.use_cleanlab = use_cleanlab and CLEANLAB_AVAILABLE
        self.verbose = verbose

    def fit(self, texts: Seq[str], labels: Seq[str]) -> None:
        """
        Train the TF-IDF vectorizer and classifier.

        Args:
            texts: Training text samples
            labels: Corresponding labels
        """
        t0 = time.time()
        texts = list(map(str, texts))
        labels = list(map(str, labels))
        if self.verbose:
            logger.info(f"Fitting vectorizer on {len(texts)} texts...")
        y = self.le.fit_transform(labels)
        X = self.vectorizer.fit_transform(texts)
        # Downcast to float32 to reduce memory usage during training
        X = X.astype(np.float32)
        if self.verbose:
            logger.info(
                f"Vectorizer done in {time.time()-t0:.1f}s (shape={X.shape})"
            )
        t1 = time.time()
        if self.use_cleanlab:
            if self.verbose:
                logger.info("Training classifier via Cleanlab...")
            cl = CleanLearning(self.clf)
            cl.fit(X, y)
            self.clf = cl.estimator
        else:
            if self.verbose:
                logger.info(f"Training {self.backend} classifier...")
            # If any class has only 1 sample, SGDClassifier early_stopping's internal CV will fail.
            # Disable early_stopping in that case.
            if self.backend == "sgd":
                class_counts = np.bincount(y)
                if (class_counts > 0).any() and (
                    class_counts[class_counts > 0].min() < 2
                ):
                    if self.verbose:
                        logger.warning(
                            "Detected classes with a single sample; disabling early_stopping."
                        )
                    try:
                        # Only disable early_stopping; do not change validation_fraction (must be in (0,1)).
                        self.clf.set_params(early_stopping=False)
                    except (AttributeError, ValueError) as e:
                        logger.debug(f"Could not disable early_stopping: {e}")
            self.clf.fit(X, y)
        if self.verbose:
            logger.info(
                f"Classifier done in {time.time()-t1:.1f}s; total {time.time()-t0:.1f}s"
            )
        self.classes_ = list(self.le.classes_)

    def predict_proba(self, texts: Seq[str]) -> np.ndarray:
        """
        Predict class probabilities for texts.

        Args:
            texts: Text samples to predict

        Returns:
            Array of shape (n_samples, n_classes) with class probabilities
        """
        Xq = self.vectorizer.transform(list(map(str, texts)))
        proba = self.clf.predict_proba(Xq)
        return proba

    def predict(self, texts: Seq[str]) -> np.ndarray:
        """
        Predict class labels for texts.

        Args:
            texts: Text samples to predict

        Returns:
            Array of predicted class labels
        """
        Xq = self.vectorizer.transform(list(map(str, texts)))
        yhat = self.clf.predict(Xq)
        return self.le.inverse_transform(yhat)

    def save(self, dir_path: str) -> None:
        """
        Save model to disk.

        Args:
            dir_path: Directory path to save model
        """
        path = Path(dir_path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "vectorizer": self.vectorizer,
                "classifier": self.clf,
                "label_encoder": self.le,
                "classes": self.classes_,
                "backend": self.backend,
            },
            path / "tfidf_model.joblib",
        )

    @classmethod
    def load(cls, dir_path: str) -> "TfidfLogRegCleanlab":
        """
        Load model from disk.

        Args:
            dir_path: Directory path containing saved model

        Returns:
            Loaded TfidfLogRegCleanlab instance
        """
        path = Path(dir_path)
        data = joblib.load(path / "tfidf_model.joblib")
        obj = cls()
        obj.vectorizer = data["vectorizer"]
        obj.clf = data["classifier"]
        obj.le = data["label_encoder"]
        obj.classes_ = data["classes"]
        obj.backend = data.get("backend", "sgd")
        return obj


