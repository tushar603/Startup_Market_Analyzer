"""
Module: src/models/stacking.py
Role: Production Stacking Ensemble with Out-Of-Fold (OOF) Prediction Harness.
Concepts: Stacking Generalization, Out-of-Fold Cross-Validation, Base vs. Meta Learner.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import joblib


class OutOfFoldStackingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        n_splits: int = 5,
        random_state: int = 42,
    ):
        self.n_splits = n_splits
        self.random_state = random_state

        # Base Estimators
        self.rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            random_state=self.random_state,
        )
        self.xgb = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            eval_metric="mlogloss",
            random_state=self.random_state,
        )

        # Meta-Learner & Feature Scaler
        self.meta_learner = LogisticRegression(C=1.0, max_iter=1000)
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "OutOfFoldStackingClassifier":
        X_mat = self.scaler.fit_transform(X.values)
        y_vec = y.values.astype(int)

        n_samples = len(X_mat)
        n_classes = len(np.unique(y_vec))

        # Out-of-fold prediction matrices for meta-learner features
        oof_rf = np.zeros((n_samples, n_classes))
        oof_xgb = np.zeros((n_samples, n_classes))

        skf = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state,
        )

        for train_idx, val_idx in skf.split(X_mat, y_vec):
            X_tr, y_tr = X_mat[train_idx], y_vec[train_idx]
            X_va = X_mat[val_idx]

            # Clone and fit fold-specific base learners
            fold_rf = RandomForestClassifier(
                n_estimators=100,
                max_depth=6,
                random_state=self.random_state,
            )
            fold_xgb = XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.05,
                eval_metric="mlogloss",
                random_state=self.random_state,
            )

            fold_rf.fit(X_tr, y_tr)
            fold_xgb.fit(X_tr, y_tr)

            oof_rf[val_idx] = fold_rf.predict_proba(X_va)
            oof_xgb[val_idx] = fold_xgb.predict_proba(X_va)

        # Meta-feature matrix composed strictly of out-of-fold probability vectors
        meta_features = np.hstack([oof_rf, oof_xgb])
        self.meta_learner.fit(meta_features, y_vec)

        # Refit final base learners on complete training dataset for production inference
        self.rf.fit(X_mat, y_vec)
        self.xgb.fit(X_mat, y_vec)
        self.is_fitted = True

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted.")
        X_mat = self.scaler.transform(X.values)
        p_rf = self.rf.predict_proba(X_mat)
        p_xgb = self.xgb.predict_proba(X_mat)
        meta_features = np.hstack([p_rf, p_xgb])
        return self.meta_learner.predict_proba(meta_features)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        prob = self.predict_proba(X)
        return np.argmax(prob, axis=1)

    def save_artifacts(self, filepath: str) -> None:
        if not self.is_fitted:
            raise RuntimeError("Cannot serialize unfitted ensemble.")
        joblib.dump(
            {
                "scaler": self.scaler,
                "rf": self.rf,
                "xgb": self.xgb,
                "meta": self.meta_learner,
            },
            filepath,
        )

    def load_artifacts(self, filepath: str) -> None:
        artifacts = joblib.load(filepath)
        self.scaler = artifacts["scaler"]
        self.rf = artifacts["rf"]
        self.xgb = artifacts["xgb"]
        self.meta_learner = artifacts["meta"]
        self.is_fitted = True