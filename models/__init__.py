"""Model implementations for text classification."""

from .base import BaseModel
from .camembert_logreg import CamembertLogReg
from .emb_logreg import SentenceEmbLogReg
from .knn_conformity import KNNConformity
from .tfidf_logreg_cleanlab import TfidfLogRegCleanlab

__version__ = "0.1.0"

__all__ = [
    "BaseModel",
    "TfidfLogRegCleanlab",
    "SentenceEmbLogReg",
    "CamembertLogReg",
    "KNNConformity",
]
