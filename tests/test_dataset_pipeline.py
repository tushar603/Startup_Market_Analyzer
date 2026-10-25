"""
Suite: tests/test_dataset_pipeline.py
Role: Tests for dataset builder, missing-target drop rules, and feature-target alignment.
"""

from datetime import datetime, timezone, timedelta
import pandas as pd
import pytest

from src.data.build_dataset import DatasetBuilder
from src.data.synthetic_generator import generate_synthetic_smoke_dataset
from src.domain.models import CompanyProfile, ArticleRecord
from src.features.fusion import FeatureFusionPipeline
from src.nlp.sentiment_engine import HybridNLPEngine


@pytest.fixture(scope="module")
def builder():
    nlp = HybridNLPEngine(use_transformer=False)
    pipeline = FeatureFusionPipeline(nlp_engine=nlp)
    return DatasetBuilder(pipeline=pipeline)


def test_dataset_compilation_and_isolation(builder):
    now = datetime.now(timezone.utc)
    profile = CompanyProfile(
        company_id="acme",
        company_name="Acme",
        snapshot_date=now,
        growth_category=2,
        observed_valuation_usd_m=50.0,
    )
    samples = [(profile, [], now)]

    X, y, report = builder.compile_tabular_dataset(samples, drop_unlabeled=True)

    assert len(X) == 1
    assert len(y) == 1
    assert "growth_category" not in X.columns
    assert "valuation_usd_m" not in X.columns
    assert y["growth_category"].iloc[0] == 2
    assert report["total_rows_ingested"] == 1
    assert report["labeled_growth_rows"] == 1


def test_unlabeled_records_dropped(builder):
    now = datetime.now(timezone.utc)
    profile_unlabeled = CompanyProfile(
        company_id="ghost",
        company_name="Ghost",
        snapshot_date=now,
        growth_category=None,
    )
    samples = [(profile_unlabeled, [], now)]

    X, y, report = builder.compile_tabular_dataset(samples, drop_unlabeled=True)

    assert len(X) == 0
    assert len(y) == 0
    assert report["total_rows_ingested"] == 1
    assert report["labeled_growth_rows"] == 0


def test_synthetic_smoke_cohort_generation(builder):
    smoke_samples = generate_synthetic_smoke_dataset(n_companies=10)
    X, y, report = builder.compile_tabular_dataset(smoke_samples, drop_unlabeled=True)

    assert len(X) == 10
    assert len(y) == 10
    assert report["unique_startups"] == 10
    assert not X.isna().any().any()