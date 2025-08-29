# Ensemble Model Details

This document provides a comprehensive explanation of how each model in the ensemble works, including their architecture, training process, and prediction mechanisms.

## Overview

The ensemble consists of four different model types, each with complementary strengths:

1. **TF-IDF + Logistic Regression with CleanLab** - Character-level n-gram features
2. **Sentence Embeddings + Logistic Regression** - Semantic similarity using transformers
3. **CamemBERT + Logistic Regression** - French language-specific BERT embeddings
4. **KNN Conformity** - Non-parametric similarity-based classification

---

## 1. TF-IDF + Logistic Regression with CleanLab

### Architecture
```
Text Input → TF-IDF Vectorization → Logistic Regression → Class Prediction
```

### Key Components

#### **TF-IDF Vectorizer**
- **N-gram Range**: `(1, 3)` - Uses character-level unigrams, bigrams, and trigrams
- **Analyzer**: `'char'` - Character-level tokenization instead of word-level
- **Max Features**: `100,000` - Limits vocabulary size to prevent memory issues
- **Example**: "hello" → ['h', 'e', 'l', 'l', 'o', 'he', 'el', 'll', 'lo', 'hel', 'ell', 'llo']

#### **Classifier Backend Options**
1. **SGD Classifier** (default):
   - Loss: `'log_loss'` (logistic regression)
   - Alpha: `1e-5` (L2 regularization)
   - Max Iterations: `20`
   - Early Stopping: `True` with 10% validation split
   - Memory-efficient for large datasets

2. **Logistic Regression**:
   - Solver: `'lbfgs'` (default) or configurable
   - Max Iterations: `200`
   - C: `4.0` (inverse regularization strength)
   - Single-threaded to avoid memory issues

#### **CleanLab Integration**
- **Purpose**: Handles label noise and data quality issues
- **Method**: Uses confident learning to identify and correct mislabeled samples
- **Process**: 
  1. Estimates label noise
  2. Identifies potentially mislabeled samples
  3. Trains classifier on cleaned data

### Training Process
```python
# 1. Text preprocessing
texts = list(map(str, texts))
labels = list(map(str, labels))

# 2. Label encoding
y = self.le.fit_transform(labels)

# 3. TF-IDF vectorization
X = self.vectorizer.fit_transform(texts)
X = X.astype(np.float32)  # Memory optimization

# 4. Classifier training
if self.use_cleanlab:
    cl = CleanLearning(self.clf)
    cl.fit(X, y)
    self.clf = cl.estimator
else:
    self.clf.fit(X, y)
```

### Prediction Process
```python
# 1. Vectorize input text
Xq = self.vectorizer.transform(texts)

# 2. Get probability predictions
proba = self.clf.predict_proba(Xq)
```

### Strengths
- **Character-level features**: Captures spelling variations and typos
- **Memory efficient**: Sparse matrix representation
- **Fast inference**: Simple linear model
- **Noise handling**: CleanLab integration for robust training

### Use Cases
- Product names with spelling variations
- Short text classification
- When computational resources are limited

---

## 2. Sentence Embeddings + Logistic Regression

### Architecture
```
Text Input → Sentence Transformer → Dense Embeddings → Logistic Regression → Class Prediction
```

### Key Components

#### **Sentence Transformer**
- **Model**: `'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'`
- **Type**: Multilingual BERT-based encoder
- **Output Dimension**: 384-dimensional embeddings
- **Normalization**: L2 normalization applied
- **Language Support**: 50+ languages including French

#### **Logistic Regression**
- **Max Iterations**: `200`
- **C**: `4.0` (regularization strength)
- **Multi-class**: `'auto'` (handles multiple classes automatically)

### Training Process
```python
# 1. Label encoding
y = self.le.fit_transform(labels)

# 2. Generate embeddings
X = self._embed(texts)  # Returns 384-dim vectors

# 3. Train classifier
self.clf.fit(X, y)
```

### Embedding Generation
```python
def _embed(self, texts: Seq[str]) -> np.ndarray:
    return self.encoder.encode(
        list(map(str, texts)),
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # L2 normalization
    )
```

### Prediction Process
```python
# 1. Generate embeddings for new texts
Xq = self._embed(texts)

# 2. Get probability predictions
proba = self.clf.predict_proba(Xq)
```

### Strengths
- **Semantic understanding**: Captures meaning beyond exact word matches
- **Multilingual**: Works across different languages
- **Transfer learning**: Leverages pre-trained language models
- **Dense representations**: Rich feature space

### Use Cases
- Semantic similarity classification
- Cross-language text classification
- When semantic understanding is important

---

## 3. CamemBERT + Logistic Regression

### Architecture
```
Text Input → CamemBERT Tokenizer → BERT Encoder → Mean Pooling → Logistic Regression → Class Prediction
```

### Key Components

#### **CamemBERT Model**
- **Base Model**: `'camembert-base'`
- **Type**: French-specific BERT model
- **Architecture**: 12-layer transformer with 768 hidden dimensions
- **Vocabulary**: French-optimized tokenizer
- **Device**: Automatic CUDA/CPU selection

#### **Tokenization Process**
- **Max Length**: `64` tokens
- **Padding**: Dynamic padding within batches
- **Truncation**: Automatic truncation for long texts
- **Return Format**: PyTorch tensors

#### **Embedding Extraction**
```python
# 1. Tokenize input
toks = self.tokenizer(batch, padding=True, truncation=True, 
                     return_tensors='pt', max_length=64)

# 2. Get BERT outputs
outputs = self.backbone(**toks)
last_hidden = outputs.last_hidden_state  # (B, T, H)

# 3. Mean pooling (excluding padding)
attn_mask = toks['attention_mask'].unsqueeze(-1)
masked = last_hidden * attn_mask
sum_vec = masked.sum(dim=1)
lengths = attn_mask.sum(dim=1).clamp(min=1)
vec = (sum_vec / lengths).cpu().numpy()

# 4. L2 normalization
vec = vec / (np.linalg.norm(vec, axis=1, keepdims=True) + 1e-12)
```

#### **Logistic Regression**
- **Max Iterations**: `200`
- **C**: `4.0`
- **Multi-class**: `'auto'`

### Training Process
```python
# 1. Label encoding
y = self.le.fit_transform(labels)

# 2. Generate BERT embeddings (batched)
X = self._embed(texts)  # Returns 768-dim vectors

# 3. Train classifier
self.clf.fit(X, y)
```

### Prediction Process
```python
# 1. Generate BERT embeddings
Xq = self._embed(texts)

# 2. Get probability predictions
proba = self.clf.predict_proba(Xq)
```

### Strengths
- **French language expertise**: Specifically trained on French text
- **Contextual understanding**: Captures word context and relationships
- **Rich representations**: 768-dimensional feature space
- **State-of-the-art**: Leverages latest transformer technology

### Use Cases
- French text classification
- When contextual understanding is crucial
- High-accuracy requirements

---

## 4. KNN Conformity

### Architecture
```
Text Input → Sentence Embeddings → K-Nearest Neighbors → Conformity Scoring → Class Prediction
```

### Key Components

#### **Sentence Transformer**
- **Model**: `'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'`
- **Same as**: Sentence Embeddings model
- **Purpose**: Generate semantic embeddings for similarity computation

#### **K-Nearest Neighbors**
- **K**: `20` (default, configurable)
- **Metric**: `'cosine'` similarity
- **Algorithm**: Ball tree or KD tree (automatic selection)

#### **Conformity Scoring**
The model computes a "conformity score" based on the fraction of nearest neighbors sharing the same label:

```python
def predict_proba(self, texts: Seq[str]) -> np.ndarray:
    # 1. Generate embeddings
    emb = self._embed(texts)
    
    # 2. Find k nearest neighbors
    dists, idx = self.nn.kneighbors(emb, return_distance=True)
    y_neighbors = self.y_train[idx]
    
    # 3. Compute class fractions
    num_classes = len(self.classes_)
    proba = np.zeros((len(texts), num_classes), dtype=np.float64)
    
    for i in range(len(texts)):
        # Count neighbors per class
        counts = np.bincount(y_neighbors[i], minlength=num_classes)
        # Convert to probabilities
        proba[i] = counts / counts.sum() if counts.sum() > 0 else np.ones(num_classes) / num_classes
    
    return proba
```

### Training Process
```python
# 1. Label encoding
y = self.le.fit_transform(labels)

# 2. Generate embeddings
emb = self._embed(texts)

# 3. Build nearest neighbors index
self.nn.fit(emb)
self._train_embeddings = emb
self.y_train = y
```

### Prediction Process
```python
# 1. Generate embeddings for query texts
emb = self._embed(texts)

# 2. Find k nearest neighbors
dists, idx = self.nn.kneighbors(emb)

# 3. Compute conformity scores
y_neighbors = self.y_train[idx]
# ... compute class fractions as probabilities
```

### Strengths
- **Non-parametric**: No assumptions about data distribution
- **Interpretable**: Predictions based on similar examples
- **Robust**: Less sensitive to outliers
- **Adaptive**: Naturally handles new classes

### Use Cases
- When interpretability is important
- Small to medium datasets
- When you want predictions based on similar examples
- Novel class detection

---

## Model Comparison Summary

| Model | Feature Type | Strengths | Weaknesses | Best For |
|-------|-------------|-----------|------------|----------|
| **TF-IDF** | Character n-grams | Fast, memory efficient, handles typos | Limited semantic understanding | Large datasets, spelling variations |
| **Sentence Embeddings** | Semantic vectors | Multilingual, semantic understanding | Slower inference | Cross-language, semantic tasks |
| **CamemBERT** | Contextual embeddings | French expertise, state-of-the-art | Resource intensive | High-accuracy French text |
| **KNN** | Similarity-based | Interpretable, non-parametric | Slow for large datasets | Small datasets, interpretability |

## Ensemble Benefits

The combination of these models provides:

1. **Complementary Strengths**: Each model excels at different aspects
2. **Robustness**: Reduces overfitting to any single approach
3. **Coverage**: Handles various text characteristics
4. **Confidence Estimation**: Ensemble confidence provides reliability measure

## Performance Considerations

- **TF-IDF**: Fastest, lowest memory usage
- **Sentence Embeddings**: Moderate speed and memory
- **CamemBERT**: Slowest, highest memory usage
- **KNN**: Speed depends on dataset size

The ensemble automatically balances these trade-offs by averaging predictions, providing the best of all approaches.
