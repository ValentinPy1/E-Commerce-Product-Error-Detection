from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence as Seq

import joblib
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None  # type: ignore

from .base import BaseModel

logger = logging.getLogger(__name__)


class KNNConformity(BaseModel):
    """
    Non-parametric model that computes conformity score.

    Computes fraction of k nearest neighbors sharing the predicted label.
    Uses sentence embeddings.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        k: int = 20,
    ):
        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers is required for KNNConformity"
            )
        self.model_name = model_name
        self.encoder = SentenceTransformer(model_name)
        self.k = k
        self.nn = NearestNeighbors(n_neighbors=k, metric="cosine")
        self.le = LabelEncoder()
        self.classes_: list[str] | None = None
        self._train_embeddings: np.ndarray | None = None

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
        Train the KNN model on embedded texts.

        Args:
            texts: Training text samples
            labels: Corresponding labels
        """
        y = self.le.fit_transform(list(map(str, labels)))
        emb = self._embed(texts)
        self.nn.fit(emb)
        self._train_embeddings = emb
        self.y_train = y
        self.classes_ = list(self.le.classes_)

    def predict_proba(self, texts: Seq[str]) -> np.ndarray:
        """
        Convert conformity to pseudo-probability distribution by neighbor label fractions.
        """
        emb = self._embed(texts)
        dists, idx = self.nn.kneighbors(emb, return_distance=True)
        y_neighbors = self.y_train[idx]
        num_classes = len(self.classes_)
        proba = np.zeros((len(texts), num_classes), dtype=np.float64)
        for i in range(len(texts)):
            counts = np.bincount(y_neighbors[i], minlength=num_classes)
            if counts.sum() > 0:
                proba[i] = counts / counts.sum()
            else:
                proba[i] = np.ones(num_classes) / num_classes
        return proba

    def predict(self, texts: Seq[str]) -> np.ndarray:
        """
        Predict class labels for texts.

        Args:
            texts: Text samples to predict

        Returns:
            Array of predicted class labels
        """
        proba = self.predict_proba(texts)
        yhat = np.argmax(proba, axis=1)
        return self.le.inverse_transform(yhat)

    def save(self, dir_path: str) -> None:
        """
        Save model to disk.

        Args:
            dir_path: Directory path to save model
        """
        path = Path(dir_path)
        path.mkdir(parents=True, exist_ok=True)
        # Try to recover a stable encoder model name
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
                        encoder_name = getattr(
                            auto_model, "name_or_path", None)
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
                "nn_params": self.nn.get_params(),
                "train_embeddings": self._train_embeddings,
                "y_train": self.y_train,
                "label_encoder": self.le,
                "classes": self.classes_,
                "encoder_name": encoder_name,
            },
            path / "knn_conformity.joblib",
        )

    @classmethod
    def load(cls, dir_path: str) -> "KNNConformity":
        """
        Load model from disk.

        Args:
            dir_path: Directory path containing saved model

        Returns:
            Loaded KNNConformity instance
        """
        path = Path(dir_path)
        data = joblib.load(path / "knn_conformity.joblib")
        model_name = (
            data.get("encoder_name")
            or "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        obj = cls(model_name=model_name)
        obj.nn.set_params(**data["nn_params"])
        obj._train_embeddings = data["train_embeddings"]
        obj.y_train = data["y_train"]
        obj.le = data["label_encoder"]
        obj.classes_ = data["classes"]
        # Refit NN index
        obj.nn.fit(obj._train_embeddings)
        return obj
