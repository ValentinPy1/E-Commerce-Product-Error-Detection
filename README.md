# nricher

Ensemble text classification models for French product categorization using multiple complementary approaches.

## Overview

This project implements an ensemble of four different text classification models, each with complementary strengths:

1. **TF-IDF + Logistic Regression with CleanLab** - Character-level n-gram features for handling spelling variations
2. **Sentence Embeddings + Logistic Regression** - Semantic similarity using multilingual transformers
3. **CamemBERT + Logistic Regression** - French language-specific BERT embeddings
4. **KNN Conformity** - Non-parametric similarity-based classification

The ensemble combines predictions from all available models to provide robust and accurate classifications.

## Installation

### Prerequisites

- Python 3.9 or higher
- pip

### Basic Installation

```bash
pip install -r requirements.txt
```

### Development Installation

```bash
# Install with development dependencies
pip install -e ".[dev]"

# Or install separately
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### Optional Dependencies

Some models require additional dependencies:

- **CamemBERT**: Requires `transformers` and `torch` (already in requirements)
- **Sentence Embeddings**: Requires `sentence-transformers` (already in requirements)
- **CleanLab**: Requires `cleanlab` (already in requirements)

## Quick Start

### Training an Ensemble

```bash
# Train with default TF-IDF model only
python -m scripts.ensemble.train_ensemble --data data/unique_products.csv

# Train with multiple models
python -m scripts.ensemble.train_ensemble \
    --data data/unique_products.csv \
    --tfidf \
    --sent \
    --camembert \
    --knn

# Limit training samples
python -m scripts.ensemble.train_ensemble \
    --data data/unique_products.csv \
    --tfidf \
    --max_samples 10000
```

### Making Predictions

```bash
python -m scripts.ensemble.predict_ensemble \
    --artifacts ./artifacts \
    --input_csv data/test_products.csv \
    --output_csv predictions.csv
```

### LLM-Assisted Judging

```bash
# Set your OpenAI API key in .env file
echo "OPENAI_API_KEY=your_key_here" > .env

# Run LLM judging on model predictions
python -m scripts.llm_judge \
    --predictions ./artifacts/ensemble_predictions.csv \
    --samples_per_model 100
```

## Project Structure

```
nricher/
├── models/              # Model implementations
│   ├── base.py         # Base model interface
│   ├── tfidf_logreg_cleanlab.py
│   ├── emb_logreg.py
│   ├── camembert_logreg.py
│   └── knn_conformity.py
├── scripts/            # Training and prediction scripts
│   ├── ensemble/       # Ensemble pipeline
│   ├── llm_judge.py    # LLM-assisted evaluation
│   └── ...
├── tests/              # Test suite
├── artifacts/          # Saved models and predictions
├── data/               # Training and test data
└── requirements.txt    # Python dependencies
```

## Usage Examples

### Python API

```python
from scripts.ensemble.ensemble import EnsemblePipeline, EnsembleConfig
import pandas as pd

# Load data
df = pd.read_csv('data/products.csv')
texts = df['Libellé produit'].tolist()
labels = df['Nature'].tolist()

# Configure ensemble
config = EnsembleConfig(
    use_tfidf=True,
    use_sent_emb=True,
    use_camembert=False,  # Slower but more accurate
    use_knn=False,
    test_size=0.1,
    random_state=42
)

# Create and train pipeline
pipeline = EnsemblePipeline(config)
X_train, X_test, y_train, y_test = pipeline.split(texts, labels)
pipeline.fit(X_train, y_train)

# Evaluate
metrics = pipeline.evaluate(X_test, y_test)
print(metrics)

# Make predictions
predictions = pipeline.avg_prob(X_test)
```

### Individual Models

```python
from models import TfidfLogRegCleanlab

# Initialize model
model = TfidfLogRegCleanlab(backend='sgd', use_cleanlab=True)

# Train
model.fit(texts, labels)

# Predict
predictions = model.predict(new_texts)
probabilities = model.predict_proba(new_texts)

# Save
model.save('artifacts/my_model')

# Load
loaded_model = TfidfLogRegCleanlab.load('artifacts/my_model')
```

## Configuration

### Environment Variables

Create a `.env` file (see `.env.example`) with:

```bash
# API Keys (for LLM judging)
OPENAI_API_KEY=your_key_here

# Paths (optional, defaults shown)
ARTIFACTS_DIR=./artifacts
DATA_DIR=./data
```

### Model Configuration

Each model can be configured with different parameters:

- **TF-IDF**: `max_features`, `c` (regularization), `backend` ('sgd' or 'logreg')
- **Sentence Embeddings**: `model_name`, `c` (regularization)
- **CamemBERT**: `model_name`, `device` ('cuda' or 'cpu'), `c` (regularization)
- **KNN**: `model_name`, `k` (number of neighbors)

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov

# Run specific test file
pytest tests/test_models/test_tfidf.py
```

### Code Quality

```bash
# Format code
black .

# Lint code
ruff check .

# Type checking
mypy models scripts

# Run all checks
make lint
```

### Pre-commit Hooks

```bash
# Install pre-commit hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

## Model Details

For detailed information about each model's architecture and training process, see:
- [Model Details](model_details.md)
- [Ensemble Workflow](ensemble_workflow.md)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass and code is formatted
6. Submit a pull request

## License

MIT License

## Acknowledgments

- CamemBERT by Hugging Face
- Sentence Transformers library
- CleanLab for label noise handling

