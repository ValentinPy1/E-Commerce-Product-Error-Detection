from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Sequence as Seq

import numpy as np
import pandas as pd


class BaseModel(ABC):
    """
    Abstract base class for all text classification models.

    All models must implement fit, predict, predict_proba, save, and load methods.
    Models should set self.classes_ after fitting.
    """

    @abstractmethod
    def fit(self, texts: Seq[str], labels: Seq[str]) -> None:
        """
        Train the model on texts and labels.

        Args:
            texts: Training text samples
            labels: Corresponding labels
        """
        ...

    @abstractmethod
    def predict_proba(self, texts: Seq[str]) -> np.ndarray:
        """
        Predict class probabilities for texts.

        Args:
            texts: Text samples to predict

        Returns:
            Array of shape (n_samples, n_classes) with class probabilities
        """
        ...

    @abstractmethod
    def predict(self, texts: Seq[str]) -> np.ndarray:
        """
        Predict class labels for texts.

        Args:
            texts: Text samples to predict

        Returns:
            Array of predicted class labels
        """
        ...

    @abstractmethod
    def save(self, dir_path: str) -> None:
        """
        Save model to disk.

        Args:
            dir_path: Directory path to save model
        """
        ...

    @classmethod
    @abstractmethod
    def load(cls, dir_path: str) -> "BaseModel":
        """
        Load model from disk.

        Args:
            dir_path: Directory path containing saved model

        Returns:
            Loaded model instance
        """
        ...

    def score_mislabeled(
        self, texts: Seq[str], labels: Seq[str], prob_threshold: float = 0.2
    ) -> pd.DataFrame:
        """
        Score texts for potential mislabeling.

        Identifies samples where the true label has low probability,
        suggesting potential labeling errors.

        Args:
            texts: Text samples to score
            labels: True labels for texts
            prob_threshold: Probability threshold below which a label is
                considered a candidate mislabel

        Returns:
            DataFrame with columns:
                - label: Original text
                - true_nature: True label
                - true_prob: Probability assigned to true label
                - pred_nature: Predicted label
                - pred_prob: Probability of predicted label
                - top_suggestions: Top 5 predicted classes with probabilities
                - candidate_mislabel: Boolean indicating if true_prob < threshold
        """
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
            rows.append(
                {
                    "label": t,
                    "true_nature": str(y),
                    "true_prob": p_true,
                    "pred_nature": str(preds[i]),
                    "pred_prob": float(np.max(pr)),
                    "top_suggestions": top,
                    "candidate_mislabel": bool(
                        not np.isnan(p_true) and p_true < prob_threshold
                    ),
                }
            )
        return pd.DataFrame(rows)


