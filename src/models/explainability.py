"""
Module: src/models/explainability.py
Role: Tree-based feature attribution and prediction explainability via SHAP.
Concepts: Shapley Values, Game-Theoretic Attribution, Local vs Global Explainability.
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd
import shap

from src.models.stacking import OutOfFoldStackingClassifier


class ModelExplainer:
    def __init__(self, model: OutOfFoldStackingClassifier, feature_names: List[str]):
        if not model.is_fitted:
            raise RuntimeError("Model must be fitted before initializing ModelExplainer.")
        self.model = model
        self.feature_names = feature_names

        # Use XGBoost tree model for fast TreeExplainer computation
        self.tree_explainer = shap.TreeExplainer(self.model.xgb)

    def explain_global(self, X: pd.DataFrame) -> Dict[str, float]:
        """
        Computes mean absolute SHAP values across all features to produce 
        global feature importance rankings.
        """
        X_scaled = self.model.scaler.transform(X.values)
        shap_values = self.tree_explainer.shap_values(X_scaled)

        # For multiclass, aggregate mean absolute attribution across classes
        if isinstance(shap_values, list):
            mean_abs = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
        elif len(shap_values.shape) == 3:  # (samples, features, classes)
            mean_abs = np.abs(shap_values).mean(axis=(0, 2))
        else:
            mean_abs = np.abs(shap_values).mean(axis=0)

        importance_dict = {
            feat: float(round(score, 5))
            for feat, score in zip(self.feature_names, mean_abs)
        }
        # Sort descending by importance
        return dict(sorted(importance_dict.items(), key=lambda item: item[1], reverse=True))

    def explain_local_instance(self, sample_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Computes local feature contribution (waterfall breakdown) for a single startup.
        """
        if len(sample_df) != 1:
            raise ValueError("explain_local_instance expects a single-row DataFrame.")

        X_scaled = self.model.scaler.transform(sample_df.values)
        shap_values = self.tree_explainer.shap_values(X_scaled)
        probs = self.model.predict_proba(sample_df)[0]
        predicted_class = int(np.argmax(probs))

        # Class-specific SHAP attribution vector
        if isinstance(shap_values, list):
            class_shap = shap_values[predicted_class][0]
        elif len(shap_values.shape) == 3:
            class_shap = shap_values[0, :, predicted_class]
        else:
            class_shap = shap_values[0]

        attributions = {
            feat: float(round(val, 5))
            for feat, val in zip(self.feature_names, class_shap)
        }

        return {
            "predicted_class": predicted_class,
            "class_probabilities": [round(float(p), 4) for p in probs],
            "top_positive_drivers": dict(
                sorted(
                    {k: v for k, v in attributions.items() if v > 0}.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )[:3]
            ),
            "top_negative_drags": dict(
                sorted(
                    {k: v for k, v in attributions.items() if v < 0}.items(),
                    key=lambda x: x[1],
                )[:3]
            ),
        }