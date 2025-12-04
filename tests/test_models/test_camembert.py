"""Tests for CamembertLogReg model."""

import pytest

try:
    from models import CamembertLogReg

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

pytestmark = pytest.mark.skipif(
    not HAS_TRANSFORMERS, reason="transformers not available"
)


def test_camembert_initialization():
    """Test CamemBERT model initialization."""
    model = CamembertLogReg()
    assert model.classes_ is None
    assert model.device in ["cpu", "cuda"]


@pytest.mark.slow
def test_camembert_fit_predict(sample_texts, sample_labels):
    """Test CamemBERT model fit and predict."""
    model = CamembertLogReg()
    model.fit(sample_texts, sample_labels)
    assert model.classes_ is not None

    predictions = model.predict(sample_texts)
    assert len(predictions) == len(sample_texts)

    proba = model.predict_proba(sample_texts)
    assert proba.shape[0] == len(sample_texts)
    assert proba.shape[1] == len(model.classes_)

