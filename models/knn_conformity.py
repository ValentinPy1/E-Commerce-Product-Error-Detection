from __future__ import annotations

from typing import Sequence as Seq
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
import os
import joblib

try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    SentenceTransformer = None  # type: ignore

from .base import BaseModel


class KNNConformity(BaseModel):
    """
    Non-parametric model that computes conformity score: fraction of k nearest neighbors sharing the predicted label.
    Uses sentence embeddings.
    """
    def __init__(self, model_name: str = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', k: int = 20):
        if SentenceTransformer is None:
            raise ImportError('sentence-transformers is required for KNNConformity')
        self.encoder = SentenceTransformer(model_name)
        self.k = k
        self.nn = NearestNeighbors(n_neighbors=k, metric='cosine')
        self.le = LabelEncoder()
        self.classes_ = None
        self._train_embeddings: np.ndarray | None = None

    def _embed(self, texts: Seq[str]) -> np.ndarray:
        return self.encoder.encode(
            list(map(str, texts)),
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    def fit(self, texts: Seq[str], labels: Seq[str]) -> None:
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
            proba[i] = counts / counts.sum() if counts.sum() > 0 else np.ones(num_classes) / num_classes
        return proba

    def predict(self, texts: Seq[str]) -> np.ndarray:
        proba = self.predict_proba(texts)
        yhat = np.argmax(proba, axis=1)
        return self.le.inverse_transform(yhat)

    def save(self, dir_path: str) -> None:
        os.makedirs(dir_path, exist_ok=True)
        # Try to recover a stable encoder model name
        encoder_name = None
        try:
            fm = self.encoder._first_module() if hasattr(self.encoder, '_first_module') else None
            if fm is not None:
                encoder_name = getattr(fm, 'model_name', None)
                if encoder_name is None:
                    auto_model = getattr(fm, 'auto_model', None)
                    if auto_model is not None:
                        encoder_name = getattr(auto_model, 'name_or_path', None)
                        if encoder_name is None:
                            cfg = getattr(auto_model, 'config', None)
                            if cfg is not None:
                                encoder_name = getattr(cfg, 'name_or_path', None) or getattr(cfg, '_name_or_path', None)
        except Exception:
            encoder_name = None
        joblib.dump({
            'nn_params': self.nn.get_params(),
            'train_embeddings': self._train_embeddings,
            'y_train': self.y_train,
            'label_encoder': self.le,
            'classes': self.classes_,
            'encoder_name': encoder_name,
        }, os.path.join(dir_path, 'knn_conformity.joblib'))

    @classmethod
    def load(cls, dir_path: str) -> "KNNConformity":
        data = joblib.load(os.path.join(dir_path, 'knn_conformity.joblib'))
        model_name = data.get('encoder_name') or 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
        obj = cls(model_name=model_name)
        obj.nn.set_params(**data['nn_params'])
        obj._train_embeddings = data['train_embeddings']
        obj.y_train = data['y_train']
        obj.le = data['label_encoder']
        obj.classes_ = data['classes']
        # Refit NN index
        obj.nn.fit(obj._train_embeddings)
        return obj


