"""Tests for BaseModel abstract class."""

import pytest
import pandas as pd

from models.base import BaseModel


class ConcreteModel(BaseModel):
    """Concrete implementation for testing."""

    def __init__(self):
        self.classes_ = ["A", "B", "C"]

    def fit(self, texts, labels):
        pass

    def predict_proba(self, texts):
        import numpy as np
        n = len(texts)
        return np.ones((n, 3)) / 3

    def predict(self, texts):
        return ["A"] * len(texts)

    def save(self, dir_path):
        pass

    @classmethod
    def load(cls, dir_path):
        return cls()


def test_base_model_cannot_instantiate():
    """Test that BaseModel cannot be instantiated directly."""
    with pytest.raises(TypeError):
        BaseModel()


def test_concrete_model_implements_all_methods(sample_texts, sample_labels):
    """Test that concrete model implements all required methods."""
    model = ConcreteModel()
    model.fit(sample_texts, sample_labels)
    preds = model.predict(sample_texts)
    proba = model.predict_proba(sample_texts)
    assert len(preds) == len(sample_texts)
    assert proba.shape == (len(sample_texts), 3)


def test_score_mislabeled(sample_texts, sample_labels):
    """Test score_mislabeled method."""
    model = ConcreteModel()
    result = model.score_mislabeled(sample_texts, sample_labels)
    assert isinstance(result, pd.DataFrame)
    assert "label" in result.columns
    assert "true_nature" in result.columns
    assert "pred_nature" in result.columns
    assert "candidate_mislabel" in result.columns
    assert len(result) == len(sample_texts)

