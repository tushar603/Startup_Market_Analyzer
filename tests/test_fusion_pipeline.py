"""
Suite: tests/test_fusion_pipeline.py
Role: Automated verification of anti-leakage, timezone safety, and feature isolation.
"""

import pytest
from datetime import datetime, timezone, timedelta
from src.domain.models import CompanyProfile, ArticleRecord
from src.features.fusion import FeatureFusionPipeline
from src.nlp.sentiment_engine import HybridNLPEngine


@pytest.fixture(scope="module")
def pipeline():
    # Use lightweight mode for deterministic unit test execution
    nlp = HybridNLPEngine(use_transformer=False)
    return FeatureFusionPipeline(nlp_engine=nlp)


def test_1_future_articles_never_leak(pipeline):
    now = datetime.now(timezone.utc)
    profile = CompanyProfile(company_id="spacex", company_name="SpaceX", snapshot_date=now)
    
    past_art = ArticleRecord("a1", "spacex", "news", "T1", "C1", now - timedelta(days=2))
    future_art = ArticleRecord("a2", "spacex", "news", "T2", "C2", now + timedelta(days=2))
    
    vector = pipeline.fuse(profile, [past_art, future_art], as_of_date=now)
    assert vector.article_count == 1


def test_2_foreign_company_articles_excluded(pipeline):
    now = datetime.now(timezone.utc)
    profile = CompanyProfile(company_id="spacex", company_name="SpaceX", snapshot_date=now)
    
    spacex_art = ArticleRecord("a1", "spacex", "news", "T1", "C1", now - timedelta(days=1))
    stripe_art = ArticleRecord("a2", "stripe", "news", "T2", "C2", now - timedelta(days=1))
    
    vector = pipeline.fuse(profile, [spacex_art, stripe_art], as_of_date=now)
    assert vector.article_count == 1


def test_3_exact_and_syndicated_duplicates_deduplicated(pipeline):
    now = datetime.now(timezone.utc)
    profile = CompanyProfile(company_id="spacex", company_name="SpaceX", snapshot_date=now)
    
    art1 = ArticleRecord("a1", "spacex", "news", "Contract signed", "SpaceX secures deal.", now - timedelta(days=1))
    art2 = ArticleRecord("a2", "spacex", "wire", "Contract  signed ", " SpaceX secures deal. ", now - timedelta(days=1))
    
    vector = pipeline.fuse(profile, [art1, art2], as_of_date=now)
    assert vector.article_count == 1


def test_4_naive_timestamps_rejected():
    with pytest.raises(ValueError, match="Naive datetimes are forbidden"):
        ArticleRecord("a1", "spacex", "news", "T", "C", datetime(2025, 1, 1))  # Naive


def test_5_missing_growth_labels_remain_none(pipeline):
    now = datetime.now(timezone.utc)
    unlabeled_profile = CompanyProfile(
        company_id="spacex", company_name="SpaceX", snapshot_date=now, growth_category=None
    )
    vector = pipeline.fuse(unlabeled_profile, [], as_of_date=now)
    assert vector.target_future_growth_category is None


def test_6_sentiment_trend_none_on_missing_window(pipeline):
    now = datetime.now(timezone.utc)
    profile = CompanyProfile(company_id="spacex", company_name="SpaceX", snapshot_date=now)
    
    # Only 1 article in the 30-day window, none in the 30-60 day window
    art = ArticleRecord("a1", "spacex", "news", "T", "C", now - timedelta(days=10))
    vector = pipeline.fuse(profile, [art], as_of_date=now)
    
    assert vector.has_recent_coverage_30d == 1
    assert vector.has_previous_30d_coverage == 0
    assert vector.sentiment_trend is None


def test_7_target_columns_never_enter_features_dict(pipeline):
    now = datetime.now(timezone.utc)
    profile = CompanyProfile(
        company_id="spacex",
        company_name="SpaceX",
        snapshot_date=now,
        growth_category=2,
        observed_valuation_usd_m=150000.0,
    )
    vector = pipeline.fuse(profile, [], as_of_date=now)
    features = vector.to_features_dict()
    targets = vector.to_targets_dict()

    assert "target_future_growth_category" not in features
    assert "observed_future_valuation_usd_m" not in features
    assert targets["growth_category"] == 2
    assert targets["valuation_usd_m"] == 150000.0