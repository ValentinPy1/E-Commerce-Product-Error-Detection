"""Tests for TfidfLogRegCleanlab model."""

import tempfile
from pathlib import Path

import pytest

from models import TfidfLogRegCleanlab


def test_tfidf_initialization():
    """Test TF-IDF model initialization."""
    model = TfidfLogRegCleanlab()
    assert model.backend == "sgd"
    assert model.classes_ is None


def test_tfidf_fit_predict(sample_texts, sample_labels):
    """Test TF-IDF model fit and predict."""
    model = TfidfLogRegCleanlab(backend="sgd", verbose=0)
    model.fit(sample_texts, sample_labels)
    assert model.classes_ is not None
    assert len(model.classes_) > 0

    predictions = model.predict(sample_texts)
    assert len(predictions) == len(sample_texts)

    proba = model.predict_proba(sample_texts)
    assert proba.shape[0] == len(sample_texts)
    assert proba.shape[1] == len(model.classes_)


def test_tfidf_save_load(sample_texts, sample_labels):
    """Test TF-IDF model save and load."""
    model = TfidfLogRegCleanlab(backend="sgd", verbose=0)
    model.fit(sample_texts, sample_labels)

    with tempfile.TemporaryDirectory() as tmpdir:
        model.save(tmpdir)
        loaded_model = TfidfLogRegCleanlab.load(tmpdir)
        assert loaded_model.classes_ == model.classes_
        assert loaded_model.backend == model.backend

        # Test predictions match
        orig_preds = model.predict(sample_texts)
        loaded_preds = loaded_model.predict(sample_texts)
        assert list(orig_preds) == list(loaded_preds)


def test_tfidf_logreg_backend(sample_texts, sample_labels):
    """Test TF-IDF with Logistic Regression backend."""
    model = TfidfLogRegCleanlab(backend="logreg", verbose=0)
    model.fit(sample_texts, sample_labels)
    predictions = model.predict(sample_texts)
    assert len(predictions) == len(sample_texts)

