from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence as Seq, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, vstack
from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import NFD, Lowercase, StripAccents, Sequence as TkSequence
from tokenizers.processors import TemplateProcessing


@dataclass
class BPENBConfig:
    vocab_size: int = 20000
    min_frequency: int = 2
    max_features: int = 50000
    alpha: float = 0.1
    random_state: int = 42
    test_size: float = 0.1


class BPENaiveBayesModel:
    def __init__(self, config: Optional[BPENBConfig] = None) -> None:
        self.config: BPENBConfig = config or BPENBConfig()
        self.tokenizer: Optional[Tokenizer] = None
        self.vocab: Dict[str, int] = {}
        self.special_token_ids: set[int] = set()
        self.feature_index: Dict[str, int] = {}
        self.classifier: Optional[MultinomialNB] = None
        self.class_names: List[str] = []

    def train_tokenizer(self, texts: Iterable[str]) -> None:
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        tokenizer.normalizer = TkSequence([NFD(), Lowercase(), StripAccents()])
        tokenizer.pre_tokenizer = Whitespace()
        special_tokens = ["[PAD]", "[UNK]"]
        trainer = BpeTrainer(
            vocab_size=self.config.vocab_size,
            min_frequency=self.config.min_frequency,
            special_tokens=special_tokens,
        )
        tokenizer.train_from_iterator((str(t) for t in texts), trainer=trainer)
        tokenizer.post_processor = TemplateProcessing(
            single="$0",
            pair="$A $B",
            special_tokens=[("[PAD]", 0), ("[UNK]", 1)],
        )
        self.tokenizer = tokenizer
        self.vocab = tokenizer.get_vocab()
        self.special_token_ids = {self.vocab.get("[PAD]"), self.vocab.get("[UNK]")}

    def tokenize(self, text: str) -> List[str]:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not trained. Call train_tokenizer first.")
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        encoding = self.tokenizer.encode(text)
        return [t for t in encoding.tokens if self.vocab.get(t) not in self.special_token_ids]

    def build_feature_index(self, texts: Iterable[str]) -> None:
        token_counts: Dict[str, int] = {}
        for t in texts:
            for tok in self.tokenize(t):
                token_counts[tok] = token_counts.get(tok, 0) + 1
        sorted_tokens = sorted(token_counts.items(), key=lambda kv: kv[1], reverse=True)
        limited_tokens = sorted_tokens[: self.config.max_features]
        self.feature_index = {tok: i for i, (tok, _) in enumerate(limited_tokens)}

    def vectorize(self, texts: Seq[str]) -> csr_matrix:
        if not self.feature_index:
            raise RuntimeError("Feature index not built. Call build_feature_index first.")
        rows: List[int] = []
        cols: List[int] = []
        vals: List[float] = []
        for row_idx, text in enumerate(texts):
            local_counts: Dict[str, int] = {}
            for tok in self.tokenize(text):
                if tok in self.feature_index:
                    local_counts[tok] = local_counts.get(tok, 0) + 1
            for tok, c in local_counts.items():
                rows.append(row_idx)
                cols.append(self.feature_index[tok])
                vals.append(float(c))
        return csr_matrix((vals, (rows, cols)), shape=(len(texts), len(self.feature_index)), dtype=np.float64)

    def fit(self, labels: Seq[str], targets: Seq[str]) -> None:
        self.train_tokenizer(labels)
        self.build_feature_index(labels)
        X = self.vectorize(labels)
        y = np.asarray(targets, dtype=str)
        clf = MultinomialNB(alpha=self.config.alpha)
        clf.fit(X, y)
        self.classifier = clf
        self.class_names = clf.classes_.tolist()

    def predict_proba(self, labels: Seq[str]) -> np.ndarray:
        if self.classifier is None:
            raise RuntimeError("Model not fitted. Call fit first.")
        Xq = self.vectorize(labels)
        return self.classifier.predict_proba(Xq)

    def predict(self, labels: Seq[str]) -> np.ndarray:
        if self.classifier is None:
            raise RuntimeError("Model not fitted. Call fit first.")
        Xq = self.vectorize(labels)
        return self.classifier.predict(Xq)

    def evaluate_holdout(self, labels: Seq[str], targets: Seq[str]) -> Dict[str, float]:
        labels_arr = np.asarray(labels, dtype=str)
        y = np.asarray(targets, dtype=str)
        class_counts = pd.Series(y).value_counts()
        frequent_classes = class_counts[class_counts >= 2].index
        frequent_mask = np.isin(y, frequent_classes)
        idx_frequent = np.where(frequent_mask)[0]
        idx_rare = np.where(~frequent_mask)[0]

        if len(idx_frequent) >= 2:
            labels_f = labels_arr[idx_frequent]
            y_f = y[idx_frequent]
            X_train_labels, X_test_labels, y_train, y_test = train_test_split(
                labels_f,
                y_f,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
                stratify=y_f,
            )
        else:
            X_train_labels, X_test_labels, y_train, y_test = train_test_split(
                labels_arr,
                y,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
                shuffle=True,
                stratify=None,
            )

        self.train_tokenizer(X_train_labels)
        self.build_feature_index(X_train_labels)
        X_train = self.vectorize(X_train_labels)
        X_test = self.vectorize(X_test_labels)
        clf = MultinomialNB(alpha=self.config.alpha)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
        prec_weighted = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        return {
            "accuracy": float(acc),
            "precision_macro": float(prec_macro),
            "precision_weighted": float(prec_weighted),
        }

    def cross_validate(self, labels: Seq[str], targets: Seq[str], n_splits: int = 5) -> Dict[str, Tuple[float, float]]:
        labels_arr = np.asarray(labels, dtype=str)
        y = np.asarray(targets, dtype=str)
        class_counts = pd.Series(y).value_counts()
        frequent_classes = class_counts[class_counts >= 2].index
        mask = np.isin(y, frequent_classes)
        labels_cv = labels_arr[mask]
        y_cv = y[mask]

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.config.random_state)
        accs: List[float] = []
        p_macros: List[float] = []
        p_weighteds: List[float] = []
        for tr_idx, va_idx in skf.split(labels_cv, y_cv):
            lab_tr, lab_va = labels_cv[tr_idx], labels_cv[va_idx]
            y_tr, y_va = y_cv[tr_idx], y_cv[va_idx]
            self.train_tokenizer(lab_tr)
            self.build_feature_index(lab_tr)
            X_tr = self.vectorize(lab_tr)
            X_va = self.vectorize(lab_va)
            clf = MultinomialNB(alpha=self.config.alpha)
            clf.fit(X_tr, y_tr)
            y_va_pred = clf.predict(X_va)
            accs.append(accuracy_score(y_va, y_va_pred))
            p_macros.append(precision_score(y_va, y_va_pred, average="macro", zero_division=0))
            p_weighteds.append(precision_score(y_va, y_va_pred, average="weighted", zero_division=0))
        return {
            "accuracy_mean_std": (float(np.mean(accs)), float(np.std(accs))),
            "precision_macro_mean_std": (float(np.mean(p_macros)), float(np.std(p_macros))),
            "precision_weighted_mean_std": (float(np.mean(p_weighteds)), float(np.std(p_weighteds))),
        }

    def score_mislabeled(
        self,
        labels: Seq[str],
        targets: Seq[str],
        prob_threshold: float = 0.2,
        topk: int = 5,
    ) -> pd.DataFrame:
        if self.classifier is None:
            self.fit(labels, targets)
        proba = self.predict_proba(labels)
        preds = self.predict(labels)
        class_index = {c: i for i, c in enumerate(self.classifier.classes_)}
        rows: List[Dict[str, object]] = []
        for i, (text, true_cls) in enumerate(zip(labels, targets)):
            probs = proba[i]
            true_idx = class_index.get(str(true_cls))
            true_prob = float(probs[true_idx]) if true_idx is not None else np.nan
            top_idx = np.argsort(probs)[::-1][:topk]
            top_suggestions = [(self.classifier.classes_[j], float(probs[j])) for j in top_idx]
            rows.append(
                {
                    "label": text,
                    "true_nature": str(true_cls),
                    "true_prob": true_prob,
                    "pred_nature": str(preds[i]),
                    "pred_prob": float(np.max(probs)),
                    "top_suggestions": top_suggestions,
                    "candidate_mislabel": bool(not np.isnan(true_prob) and true_prob < prob_threshold),
                }
            )
        result = pd.DataFrame(rows)
        return result.sort_values(["candidate_mislabel", "true_prob"], ascending=[False, True]).reset_index(drop=True)


def train_bpe_nb_from_dataframe(
    df: pd.DataFrame,
    label_column: str = "Libellé produit",
    target_column: str = "Nature",
    config: Optional[BPENBConfig] = None,
) -> BPENaiveBayesModel:
    if label_column not in df.columns or target_column not in df.columns:
        raise ValueError("DataFrame must contain the specified label and target columns")
    data = df[[label_column, target_column]].dropna()
    data = data[(data[label_column].astype(str).str.strip() != "") & (data[target_column].astype(str).str.strip() != "")]
    model = BPENaiveBayesModel(config)
    model.fit(data[label_column].astype(str).tolist(), data[target_column].astype(str).tolist())
    return model


