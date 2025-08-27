from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Sequence as Seq, Type, TypeVar
import numpy as np
import pandas as pd


class BaseModel(ABC):
    @abstractmethod
    def fit(self, texts: Seq[str], labels: Seq[str]) -> None: ...

    @abstractmethod
    def predict_proba(self, texts: Seq[str]) -> np.ndarray: ...

    @abstractmethod
    def predict(self, texts: Seq[str]) -> np.ndarray: ...

    @abstractmethod
    def save(self, dir_path: str) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, dir_path: str) -> "BaseModel": ...

    def score_mislabeled(self, texts: Seq[str], labels: Seq[str], prob_threshold: float = 0.2):
        probs = self.predict_proba(texts)
        preds = self.predict(texts)
        classes = self.classes_
        class_index = {c: i for i, c in enumerate(classes)}
        rows = []
        for i, (t, y) in enumerate(zip(texts, labels)):
            pr = probs[i]
            yi = class_index.get(str(y))
            p_true = float(pr[yi]) if yi is not None else np.nan
            top_idx = np.argsort(pr)[::-1][:5]
            top = [(classes[j], float(pr[j])) for j in top_idx]
            rows.append({
                'label': t,
                'true_nature': str(y),
                'true_prob': p_true,
                'pred_nature': str(preds[i]),
                'pred_prob': float(np.max(pr)),
                'top_suggestions': top,
                'candidate_mislabel': bool(not np.isnan(p_true) and p_true < prob_threshold),
            })
        return pd.DataFrame(rows)


