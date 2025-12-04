"""Tests for KNNConformity model."""

import pytest

try:
    from models import KNNConformity

    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

pytestmark = pytest.mark.skipif(
    not HAS_SENTENCE_TRANSFORMERS,
    reason="sentence-transformers not available",
)


def test_knn_initialization():
    """Test KNN model initialization."""
    model = KNNConformity(k=5)
    assert model.k == 5
    assert model.classes_ is None


@pytest.mark.slow
def test_knn_fit_predict(sample_texts, sample_labels):
    """Test KNN model fit and predict."""
    model = KNNConformity(k=3)
    model.fit(sample_texts, sample_labels)
    assert model.classes_ is not None

    predictions = model.predict(sample_texts)
    assert len(predictions) == len(sample_texts)

    proba = model.predict_proba(sample_texts)
    assert proba.shape[0] == len(sample_texts)
    assert proba.shape[1] == len(model.classes_)

