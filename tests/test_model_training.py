"""
Suite: tests/test_model_training.py
Role: Unit tests for OutOfFoldStackingClassifier and temporal evaluation harness.
"""

import os
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import pytest

from src.data.build_dataset import DatasetBuilder
from src.data.synthetic_generator import generate_synthetic_smoke_dataset
from src.features.fusion import FeatureFusionPipeline
from src.models.evaluate import temporal_train_test_split, evaluate_classification_performance
from src.models.stacking import OutOfFoldStackingClassifier
from src.nlp.sentiment_engine import HybridNLPEngine


@pytest.fixture(scope="module")
def prepared_data():
    nlp = HybridNLPEngine(use_transformer=False)
    pipeline = FeatureFusionPipeline(nlp_engine=nlp)
    builder = DatasetBuilder(pipeline=pipeline)

    cohort = generate_synthetic_smoke_dataset(n_companies=40)
    X, y, report = builder.compile_tabular_dataset(cohort, drop_unlabeled=True)
    dates = pd.Series([item[2] for item in cohort])
    return X, y["growth_category"], dates


def test_stacking_classifier_fit_predict(prepared_data):
    X, y, _ = prepared_data
    model = OutOfFoldStackingClassifier(n_splits=3, random_state=42)
    model.fit(X, y)

    preds = model.predict(X)
    probs = model.predict_proba(X)

    assert len(preds) == len(X)
    assert probs.shape == (len(X), 3)
    assert set(np.unique(preds)).issubset({0, 1, 2})


def test_temporal_evaluation_pipeline(prepared_data, tmp_path):
    X, y, dates = prepared_data
    X_tr, X_te, y_tr, y_te = temporal_train_test_split(X, y, dates, train_ratio=0.75)

    assert len(X_tr) == 30
    assert len(X_te) == 10

    model = OutOfFoldStackingClassifier(n_splits=3, random_state=42)
    report = evaluate_classification_performance(model, X_tr, y_tr, X_te, y_te)

    assert "stacked_model_metrics" in report
    assert "baseline_metrics" in report
    assert report["sample_counts"]["train"] == 30
    assert report["sample_counts"]["test"] == 10

    # Test artifact serialization
    artifact_path = os.path.join(tmp_path, "test_stack.pkl")
    model.save_artifacts(artifact_path)
    assert os.path.exists(artifact_path)

    loaded_model = OutOfFoldStackingClassifier()
    loaded_model.load_artifacts(artifact_path)
    loaded_preds = loaded_model.predict(X_te)
    assert len(loaded_preds) == len(X_te)