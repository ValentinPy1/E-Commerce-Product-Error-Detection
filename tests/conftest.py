"""Pytest configuration and fixtures."""

import numpy as np
import pytest


@pytest.fixture
def sample_texts():
    """Sample text data for testing."""
    return [
        "Chaise en bois",
        "Table ronde",
        "Canapé gris",
        "Lampe de bureau",
        "Étagère blanche",
    ]


@pytest.fixture
def sample_labels():
    """Sample labels for testing."""
    return ["Mobilier", "Mobilier", "Mobilier", "Éclairage", "Mobilier"]


@pytest.fixture
def sample_texts_multi_class():
    """Sample texts with multiple classes."""
    # Provide more samples to avoid validation split issues in tests
    return [
        "Chaise en bois",
        "Table ronde",
        "Lampe LED",
        "Étagère blanche",
        "Ampoule 60W",
        "Fauteuil confortable",
        "Plafonnier moderne",
        "Bureau en chêne",
        "Lampe de chevet",
        "Tabouret design",
    ]


@pytest.fixture
def sample_labels_multi_class():
    """Sample labels with multiple classes."""
    # Provide more samples to avoid validation split issues in tests
    return [
        "Mobilier",
        "Mobilier",
        "Éclairage",
        "Mobilier",
        "Éclairage",
        "Mobilier",
        "Éclairage",
        "Mobilier",
        "Éclairage",
        "Mobilier",
    ]


@pytest.fixture
def random_seed():
    """Random seed for reproducibility."""
    return 42


@pytest.fixture(autouse=True)
def set_random_seed(random_seed):
    """Set random seed before each test."""
    np.random.seed(random_seed)
    yield
    np.random.seed(None)

