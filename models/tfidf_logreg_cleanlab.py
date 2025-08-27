from __future__ import annotations

from typing import Sequence as Seq, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.preprocessing import LabelEncoder
import time
import os
import joblib

try:
    from cleanlab.classification import CleanLearning
    CLEANLAB_AVAILABLE = True
except Exception:
    CLEANLAB_AVAILABLE = False

from .base import BaseModel


class TfidfLogRegCleanlab(BaseModel):
    def __init__(
        self,
        max_features: int = 100000,
        c: float = 4.0,
        n_jobs: int = -1,
        use_cleanlab: bool = False,
        solver: str = 'lbfgs',
        verbose: int = 0,
        backend: str = 'sgd',  # 'sgd' or 'logreg'
    ):
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 3), analyzer='char', max_features=max_features)
        self.backend = backend
        if backend == 'logreg':
            # Use single-thread to avoid OOM due to parallel jobs across many classes
            self.clf = LogisticRegression(max_iter=200, C=c, n_jobs=1, multi_class='auto', solver=solver, verbose=verbose)
        else:
            # Memory-lean, scalable classifier with probabilities
            self.clf = SGDClassifier(loss='log_loss', alpha=1e-5, max_iter=20, tol=1e-3, early_stopping=True, validation_fraction=0.1)
        self.le = LabelEncoder()
        self.classes_ = None
        self.use_cleanlab = use_cleanlab and CLEANLAB_AVAILABLE
        self.verbose = verbose

    def fit(self, texts: Seq[str], labels: Seq[str]) -> None:
        t0 = time.time()
        texts = list(map(str, texts))
        labels = list(map(str, labels))
        if self.verbose:
            print(f"[TFIDF] Fitting vectorizer on {len(texts)} texts...")
        y = self.le.fit_transform(labels)
        X = self.vectorizer.fit_transform(texts)
        # Downcast to float32 to reduce memory usage during training
        X = X.astype(np.float32)
        if self.verbose:
            print(f"[TFIDF] Vectorizer done in {time.time()-t0:.1f}s (shape={X.shape})")
        t1 = time.time()
        if self.use_cleanlab:
            if self.verbose:
                print("[TFIDF] Training classifier via Cleanlab...")
            cl = CleanLearning(self.clf)
            cl.fit(X, y)
            self.clf = cl.estimator
        else:
            if self.verbose:
                print(f"[TFIDF] Training {self.backend} classifier...")
            # If any class has only 1 sample, SGDClassifier early_stopping's internal CV will fail.
            # Disable early_stopping in that case.
            if self.backend == 'sgd':
                class_counts = np.bincount(y)
                if (class_counts > 0).any() and (class_counts[class_counts > 0].min() < 2):
                    if self.verbose:
                        print("[TFIDF] Detected classes with a single sample; disabling early_stopping.")
                    try:
                        # Only disable early_stopping; do not change validation_fraction (must be in (0,1)).
                        self.clf.set_params(early_stopping=False)
                    except Exception:
                        pass
            self.clf.fit(X, y)
        if self.verbose:
            print(f"[TFIDF] Classifier done in {time.time()-t1:.1f}s; total {time.time()-t0:.1f}s")
        self.classes_ = list(self.le.classes_)

    def predict_proba(self, texts: Seq[str]) -> np.ndarray:
        Xq = self.vectorizer.transform(list(map(str, texts)))
        proba = self.clf.predict_proba(Xq)
        return proba

    def predict(self, texts: Seq[str]) -> np.ndarray:
        Xq = self.vectorizer.transform(list(map(str, texts)))
        yhat = self.clf.predict(Xq)
        return self.le.inverse_transform(yhat)

    def save(self, dir_path: str) -> None:
        os.makedirs(dir_path, exist_ok=True)
        joblib.dump({
            'vectorizer': self.vectorizer,
            'classifier': self.clf,
            'label_encoder': self.le,
            'classes': self.classes_,
            'backend': self.backend,
        }, os.path.join(dir_path, 'tfidf_model.joblib'))

    @classmethod
    def load(cls, dir_path: str) -> "TfidfLogRegCleanlab":
        data = joblib.load(os.path.join(dir_path, 'tfidf_model.joblib'))
        obj = cls()
        obj.vectorizer = data['vectorizer']
        obj.clf = data['classifier']
        obj.le = data['label_encoder']
        obj.classes_ = data['classes']
        obj.backend = data.get('backend', 'sgd')
        return obj


