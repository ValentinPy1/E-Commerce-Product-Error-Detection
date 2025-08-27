from __future__ import annotations

from typing import Sequence as Seq
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import os
import joblib

try:
    from transformers import AutoTokenizer, AutoModel
    import torch
except Exception:
    AutoTokenizer = None  # type: ignore
    AutoModel = None  # type: ignore
    torch = None  # type: ignore

from .base import BaseModel


class CamembertLogReg(BaseModel):
    def __init__(self, model_name: str = 'camembert-base', device: str | None = None, c: float = 4.0, n_jobs: int = -1):
        if AutoTokenizer is None or AutoModel is None or torch is None:
            raise ImportError('transformers[torch] is required for CamembertLogReg')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        if device is None:
            device = 'cuda' if (hasattr(torch, 'cuda') and torch.cuda.is_available()) else 'cpu'
        self.device = device
        self.backbone.to(self.device)
        self.backbone.eval()
        self.clf = LogisticRegression(max_iter=200, C=c, n_jobs=n_jobs, multi_class='auto')
        self.le = LabelEncoder()
        self.classes_ = None

    def _embed(self, texts: Seq[str]) -> np.ndarray:
        if torch is None:
            raise ImportError('transformers[torch] is required for CamembertLogReg')
        batch_size = 64
        all_vecs: list[np.ndarray] = []
        start = 0
        total = len(texts)
        for i in range(0, total, batch_size):
            batch = list(map(str, texts[i:i+batch_size]))
            toks = self.tokenizer(batch, padding=True, truncation=True, return_tensors='pt', max_length=64)
            toks = {k: v.to(self.device) for k, v in toks.items()}
            with torch.no_grad():
                outputs = self.backbone(**toks)
                # Use mean pooling over token embeddings (excluding padding)
                last_hidden = outputs.last_hidden_state  # (B, T, H)
                attn_mask = toks['attention_mask'].unsqueeze(-1)  # (B, T, 1)
                masked = last_hidden * attn_mask
                sum_vec = masked.sum(dim=1)
                lengths = attn_mask.sum(dim=1).clamp(min=1)
                vec = (sum_vec / lengths).cpu().numpy()
            # L2 normalize
            vec = vec / (np.linalg.norm(vec, axis=1, keepdims=True) + 1e-12)
            all_vecs.append(vec)
            done = min(i + batch_size, total)
            if (i // batch_size) % 10 == 0 or done == total:
                print(f"[Camembert] Embedded {done}/{total}")
        return np.vstack(all_vecs)

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
        joblib.dump({
            'classifier': self.clf,
            'label_encoder': self.le,
            'classes': self.classes_,
            'backbone_name': self.tokenizer.name_or_path,
        }, os.path.join(dir_path, 'camembert_logreg.joblib'))

    @classmethod
    def load(cls, dir_path: str) -> "CamembertLogReg":
        data = joblib.load(os.path.join(dir_path, 'camembert_logreg.joblib'))
        obj = cls()
        obj.clf = data['classifier']
        obj.le = data['label_encoder']
        obj.classes_ = data['classes']
        return obj


