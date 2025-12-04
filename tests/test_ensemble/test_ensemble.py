"""Tests for ensemble pipeline."""

import tempfile
from pathlib import Path

import pytest

from scripts.ensemble.ensemble import EnsembleConfig, EnsemblePipeline


def test_ensemble_config_defaults():
    """Test EnsembleConfig default values."""
    config = EnsembleConfig()
    assert config.test_size == 0.1
    assert config.random_state == 42
    assert config.use_tfidf is True
    assert config.use_sent_emb is False


def test_ensemble_pipeline_initialization():
    """Test EnsemblePipeline initialization."""
    config = EnsembleConfig(use_tfidf=True, use_sent_emb=False)
    pipeline = EnsemblePipeline(config)
    assert len(pipeline.models) == 1
    assert pipeline.classes_ is None


def test_ensemble_split(sample_texts_multi_class, sample_labels_multi_class):
    """Test ensemble data splitting."""
    config = EnsembleConfig(test_size=0.2, random_state=42)
    pipeline = EnsemblePipeline(config)
    X_train, X_test, y_train, y_test = pipeline.split(
        sample_texts_multi_class, sample_labels_multi_class
    )
    assert len(X_train) + len(X_test) == len(sample_texts_multi_class)
    assert len(y_train) + len(y_test) == len(sample_labels_multi_class)


@pytest.mark.slow
def test_ensemble_fit_evaluate(
    sample_texts_multi_class, sample_labels_multi_class
):
    """Test ensemble fitting and evaluation."""
    config = EnsembleConfig(
        use_tfidf=True, use_sent_emb=False, test_size=0.2, random_state=42
    )
    pipeline = EnsemblePipeline(config)
    X_train, X_test, y_train, y_test = pipeline.split(
        sample_texts_multi_class, sample_labels_multi_class
    )
    pipeline.fit(X_train, y_train)
    assert pipeline.classes_ is not None

    metrics = pipeline.evaluate(X_test, y_test)
    assert "majority_vote_accuracy" in metrics
    assert "avg_prob_accuracy" in metrics
    assert 0 <= metrics["majority_vote_accuracy"] <= 1

