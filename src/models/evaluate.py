"""
Module: src/models/evaluate.py
Role: Temporal Out-of-Time Metric Evaluation & Baseline Comparison.
Concepts: Chronological Splitting, Macro-F1, Confusion Matrix, Majority-Class Baseline.
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

from src.models.stacking import OutOfFoldStackingClassifier


def temporal_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    as_of_dates: pd.Series,
    train_ratio: float = 0.8,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Splits dataset strictly by chronological cutoff timestamp."""
    sorted_indices = as_of_dates.sort_values().index
    split_idx = int(len(sorted_indices) * train_ratio)

    train_idx = sorted_indices[:split_idx]
    test_idx = sorted_indices[split_idx:]

    return X.loc[train_idx], X.loc[test_idx], y.loc[train_idx], y.loc[test_idx]


def evaluate_classification_performance(
    model: OutOfFoldStackingClassifier,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, Any]:
    """
    Fits baseline and stacked ensemble, computing classification metrics on
    an unseen future test partition.
    """
    # 1. Evaluate Simple Baseline (Majority Class Predictor)
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)
    baseline_preds = baseline.predict(X_test)
    baseline_acc = float(accuracy_score(y_test, baseline_preds))
    baseline_f1 = float(f1_score(y_test, baseline_preds, average="macro", zero_division=0))

    # 2. Fit and Evaluate Out-of-Fold Stacking Model
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = float(accuracy_score(y_test, preds))
    macro_f1 = float(f1_score(y_test, preds, average="macro", zero_division=0))
    prec_per_class = precision_score(y_test, preds, average=None, zero_division=0).tolist()
    rec_per_class = recall_score(y_test, preds, average=None, zero_division=0).tolist()
    cm = confusion_matrix(y_test, preds).tolist()

    return {
        "sample_counts": {
            "train": int(len(X_train)),
            "test": int(len(X_test)),
        },
        "baseline_metrics": {
            "strategy": "most_frequent",
            "accuracy": round(baseline_acc, 4),
            "macro_f1": round(baseline_f1, 4),
        },
        "stacked_model_metrics": {
            "accuracy": round(acc, 4),
            "macro_f1": round(macro_f1, 4),
            "precision_per_class": [round(x, 4) for x in prec_per_class],
            "recall_per_class": [round(x, 4) for x in rec_per_class],
            "confusion_matrix": cm,
        },
    }