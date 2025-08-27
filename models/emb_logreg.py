from __future__ import annotations

from typing import Sequence as Seq
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import os
import joblib

try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    SentenceTransformer = None  # type: ignore

from .base import BaseModel


class SentenceEmbLogReg(BaseModel):
    def __init__(self, model_name: str = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', c: float = 4.0, n_jobs: int = -1):
        if SentenceTransformer is None:
            raise ImportError('sentence-transformers is required for SentenceEmbLogReg')
        self.encoder = SentenceTransformer(model_name)
        self.clf = LogisticRegression(max_iter=200, C=c, n_jobs=n_jobs, multi_class='auto')
        self.le = LabelEncoder()
        self.classes_ = None

    def _embed(self, texts: Seq[str]) -> np.ndarray:
        return self.encoder.encode(
            list(map(str, texts)),
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    def fit(self, texts: Seq[str], labels: Seq[str]) -> None:
        y = self.le.fit_transform(list(map(str, labels)))
        X = self._embed(texts)
        self.clf.fit(X, y)
        self.classes_ = list(self.le.classes_)

    def predict_proba(self, texts: Seq[str]) -> np.ndarray:
        Xq = self._embed(texts)
        return self.clf.predict_proba(Xq)

    def predict(self, texts: Seq[str]) -> np.ndarray:
        Xq = self._embed(texts)
        yhat = self.clf.predict(Xq)
        return self.le.inverse_transform(yhat)

    def save(self, dir_path: str) -> None:
        os.makedirs(dir_path, exist_ok=True)
        # Save only the classifier and label encoder; encoder is referenced by name
        joblib.dump({
            'classifier': self.clf,
            'label_encoder': self.le,
            'classes': self.classes_,
            'encoder_name': self.encoder._first_module().name if hasattr(self.encoder, '_first_module') else None,
        }, os.path.join(dir_path, 'emb_logreg.joblib'))

    @classmethod
    def load(cls, dir_path: str) -> "SentenceEmbLogReg":
        data = joblib.load(os.path.join(dir_path, 'emb_logreg.joblib'))
        obj = cls()
        obj.clf = data['classifier']
        obj.le = data['label_encoder']
        obj.classes_ = data['classes']
        return obj


