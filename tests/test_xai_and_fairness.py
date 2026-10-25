"""
Suite: tests/test_xai_and_fairness.py
Role: Unit tests for SHAP TreeExplainer and AIF360 fairness auditing metrics.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.build_dataset import DatasetBuilder
from src.data.synthetic_generator import generate_synthetic_smoke_dataset
from src.features.fusion import FeatureFusionPipeline
from src.models.stacking import OutOfFoldStackingClassifier
from src.models.explainability import ModelExplainer
from src.fairness.audit import FairnessAuditor
from src.nlp.sentiment_engine import HybridNLPEngine


@pytest.fixture(scope="module")
def trained_pipeline_artifacts():
    nlp = HybridNLPEngine(use_transformer=False)
    pipeline = FeatureFusionPipeline(nlp_engine=nlp)
    builder = DatasetBuilder(pipeline=pipeline)

    cohort = generate_synthetic_smoke_dataset(n_companies=50)
    X, y, _ = builder.compile_tabular_dataset(cohort, drop_unlabeled=True)
    
    # Create a protected attribute series (e.g., 1 = Tier-1 Tech Hub, 0 = Non-hub)
    np.random.seed(42)
    protected_attr = pd.Series(np.random.binomial(1, 0.5, size=len(X)), name="is_tier1_hub")

    model = OutOfFoldStackingClassifier(n_splits=3, random_state=42)
    model.fit(X, y["growth_category"])

    return model, X, y["growth_category"], protected_attr


def test_shap_global_and_local_explanations(trained_pipeline_artifacts):
    model, X, _, _ = trained_pipeline_artifacts
    explainer = ModelExplainer(model=model, feature_names=list(X.columns))

    # Test global importance
    global_importance = explainer.explain_global(X)
    assert len(global_importance) == len(X.columns)
    assert all(isinstance(v, float) for v in global_importance.values())
    assert list(global_importance.values()) == sorted(global_importance.values(), reverse=True)

    # Test single-row local explanation
    sample = X.iloc[[0]]
    local_exp = explainer.explain_local_instance(sample)
    assert "predicted_class" in local_exp
    assert "class_probabilities" in local_exp
    assert len(local_exp["class_probabilities"]) == 3
    assert "top_positive_drivers" in local_exp
    assert "top_negative_drags" in local_exp


def test_aif360_fairness_audit(trained_pipeline_artifacts):
    model, X, y, protected_attr = trained_pipeline_artifacts
    preds = model.predict(X)

    auditor = FairnessAuditor(protected_attribute="is_tier1_hub", privileged_value=1, unprivileged_value=0)
    audit_results = auditor.audit_predictions(
        X_df=X,
        y_true=y,
        y_pred=preds,
        protected_series=protected_attr,
    )

    assert "demographic_parity_gap" in audit_results
    assert "equal_opportunity_difference" in audit_results
    assert "disparate_impact_ratio" in audit_results
    assert isinstance(audit_results["fairness_threshold_met_dpd_05"], bool)