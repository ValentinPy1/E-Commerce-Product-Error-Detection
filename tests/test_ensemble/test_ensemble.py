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
    # Use more samples for better split
    texts = sample_texts_multi_class * 2
    labels = sample_labels_multi_class * 2
    
    config = EnsembleConfig(test_size=0.2, random_state=42)
    pipeline = EnsemblePipeline(config)
    X_train, X_test, y_train, y_test = pipeline.split(texts, labels)
    assert len(X_train) + len(X_test) == len(texts)
    assert len(y_train) + len(y_test) == len(labels)


@pytest.mark.slow
def test_ensemble_fit_evaluate(
    sample_texts_multi_class, sample_labels_multi_class
):
    """Test ensemble fitting and evaluation."""
    # Use more samples to avoid validation split issues
    texts = sample_texts_multi_class * 3  # Repeat to get more samples
    labels = sample_labels_multi_class * 3
    
    config = EnsembleConfig(
        use_tfidf=True, use_sent_emb=False, test_size=0.2, random_state=42
    )
    pipeline = EnsemblePipeline(config)
    X_train, X_test, y_train, y_test = pipeline.split(texts, labels)
    
    # Ensure we have enough samples for training
    assert len(X_train) >= 3, "Need at least 3 training samples"
    assert len(X_test) >= 1, "Need at least 1 test sample"
    
    pipeline.fit(X_train, y_train)
    assert pipeline.classes_ is not None

    metrics = pipeline.evaluate(X_test, y_test)
    assert "majority_vote_accuracy" in metrics
    assert "avg_prob_accuracy" in metrics
    assert 0 <= metrics["majority_vote_accuracy"] <= 1

