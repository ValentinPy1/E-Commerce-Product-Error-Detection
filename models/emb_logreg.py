from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence as Seq

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None  # type: ignore

from .base import BaseModel

logger = logging.getLogger(__name__)


class SentenceEmbLogReg(BaseModel):
    """
    Sentence embedding-based text classification model.

    Uses multilingual sentence transformers to generate semantic embeddings,
    then applies Logistic Regression for classification.

    Args:
        model_name: Sentence transformer model name
        c: Regularization strength (inverse of C)
        n_jobs: Number of parallel jobs for Logistic Regression
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        c: float = 4.0,
        n_jobs: int = -1,
    ):
        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers is required for SentenceEmbLogReg"
            )
        self.model_name = model_name
        self.encoder = SentenceTransformer(model_name)
        self.clf = LogisticRegression(
            max_iter=200, C=c, n_jobs=n_jobs, multi_class="auto"
        )
        self.le = LabelEncoder()
        self.classes_: list[str] | None = None

    def _embed(self, texts: Seq[str]) -> np.ndarray:
        """
        Generate embeddings for texts using sentence transformer.

        Args:
            texts: Text samples to embed

        Returns:
            Array of embeddings with shape (n_samples, embedding_dim)
        """
        return self.encoder.encode(
            list(map(str, texts)),
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    def fit(self, texts: Seq[str], labels: Seq[str]) -> None:
        """
        Train the classifier on embedded texts.

        Args:
            texts: Training text samples
            labels: Corresponding labels
        """
        y = self.le.fit_transform(list(map(str, labels)))
        X = self._embed(texts)
        self.clf.fit(X, y)
        self.classes_ = list(self.le.classes_)

    def predict_proba(self, texts: Seq[str]) -> np.ndarray:
        """
        Predict class probabilities for texts.

        Args:
            texts: Text samples to predict

        Returns:
            Array of shape (n_samples, n_classes) with class probabilities
        """
        Xq = self._embed(texts)
        return self.clf.predict_proba(Xq)

    def predict(self, texts: Seq[str]) -> np.ndarray:
        """
        Predict class labels for texts.

        Args:
            texts: Text samples to predict

        Returns:
            Array of predicted class labels
        """
        Xq = self._embed(texts)
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
        # Save only the classifier and label encoder; encoder is referenced by name
        encoder_name = self.model_name
        try:
            fm = (
                self.encoder._first_module()
                if hasattr(self.encoder, "_first_module")
                else None
            )
            if fm is not None:
                encoder_name = getattr(fm, "model_name", None)
                if encoder_name is None:
                    auto_model = getattr(fm, "auto_model", None)
                    if auto_model is not None:
                        encoder_name = getattr(auto_model, "name_or_path", None)
                        if encoder_name is None:
                            cfg = getattr(auto_model, "config", None)
                            if cfg is not None:
                                encoder_name = (
                                    getattr(cfg, "name_or_path", None)
                                    or getattr(cfg, "_name_or_path", None)
                                )
        except (AttributeError, TypeError) as e:
            logger.debug(f"Could not extract encoder name: {e}")
            encoder_name = self.model_name
        joblib.dump(
            {
                "classifier": self.clf,
                "label_encoder": self.le,
                "classes": self.classes_,
                "encoder_name": encoder_name,
            },
            path / "emb_logreg.joblib",
        )

    @classmethod
    def load(cls, dir_path: str) -> "SentenceEmbLogReg":
        """
        Load model from disk.

        Args:
            dir_path: Directory path containing saved model

        Returns:
            Loaded SentenceEmbLogReg instance
        """
        path = Path(dir_path)
        data = joblib.load(path / "emb_logreg.joblib")
        model_name = (
            data.get("encoder_name")
            or "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        obj = cls(model_name=model_name)
        obj.clf = data["classifier"]
        obj.le = data["label_encoder"]
        obj.classes_ = data["classes"]
        return obj


