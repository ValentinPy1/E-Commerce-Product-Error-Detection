# Ensemble Prediction Workflow

```mermaid
flowchart TD
    A[Start: Parse Arguments] --> B[Load Models from Artifacts]
    
    B --> C{Check Model Availability}
    C --> D[TF-IDF + Logistic Regression]
    C --> E[Sentence Embeddings + Logistic Regression]
    C --> F[CamemBERT + Logistic Regression]
    C --> G[KNN Conformity]
    
    D --> H[Load TF-IDF Model]
    E --> I[Load Sentence Embeddings Model]
    F --> J[Load CamemBERT Model]
    G --> K[Load KNN Model]
    
    H --> L[Read Input CSV]
    I --> L
    J --> L
    K --> L
    
    L --> M[Extract Text Column]
    M --> N[Create Unified Class Mapping]
    
    N --> O[Initialize Arrays]
    O --> P["probas_aligned = []"]
    O --> Q["per_model_preds = {}"]
    O --> R["per_model_confs = {}"]
    
    P --> S[For Each Model]
    Q --> S
    R --> S
    
    S --> T[Get predict_proba from Model]
    T --> U[Extract Predicted Class & Confidence]
    U --> V[Align Probabilities to Unified Classes]
    
    V --> W[Store Individual Results]
    W --> X[Add to probas_aligned Array]
    
    X --> Y{More Models?}
    Y -->|Yes| S
    Y -->|No| Z[Stack All Probability Matrices]
    
    Z --> AA[Compute Mean Across Models]
    AA --> BB[Find Class with Max Average Probability]
    BB --> CC[Extract Ensemble Predictions & Confidence]
    
    CC --> DD[Create Output DataFrame]
    DD --> EE[Add Ensemble Results]
    EE --> FF[Add Individual Model Results]
    FF --> GG[Save to CSV]
    GG --> HH[End]
    
    class A,HH startEnd
    class B,L,M,N,O,P,Q,R,S,T,U,V,W,X,Z,AA,BB,CC,DD,EE,FF,GG process
    class C,Y decision
    class D,E,F,G,H,I,J,K data
    class D,E,F,G model
```

## Key Components Explained

### Model Types
- **TF-IDF + Logistic Regression**: Character n-gram based classification with optional CleanLab
- **Sentence Embeddings + Logistic Regression**: Uses multilingual sentence transformers
- **CamemBERT + Logistic Regression**: French BERT model for text classification  
- **KNN Conformity**: Non-parametric model using nearest neighbors

### Core Algorithm
1. **Model Loading**: Dynamically loads available trained models
2. **Class Alignment**: Creates unified class space across all models
3. **Individual Prediction**: Each model predicts probabilities for all classes
4. **Probability Alignment**: Maps each model's classes to unified class space
5. **Ensemble Averaging**: Computes mean probabilities across all models
6. **Final Prediction**: Selects class with highest average probability

### Output
- **ensemble_pred**: Final ensemble prediction
- **ensemble_conf**: Ensemble confidence score
- **{model}_pred**: Individual model predictions
- **{model}_conf**: Individual model confidence scores
