"""
Suite: tests/test_ingestion_and_reports.py
Role: Unit tests for live RSS fallback news ingestion and markdown memo generation.
"""

from datetime import datetime, timezone
import pytest
from src.ingestion.news_client import LiveNewsIngestor
from src.reports.generator import DueDiligenceReportGenerator


def test_live_news_rss_fallback():
    ingestor = LiveNewsIngestor()
    # Test public fallback using an established entity
    articles = ingestor.fetch_rss_fallback(company_name="Stripe", company_id="stripe", max_articles=3)

    assert isinstance(articles, list)
    if articles:
        first = articles[0]
        assert first.company_id == "stripe"
        assert first.published_at.tzinfo == timezone.utc
        assert len(first.title) > 0


def test_due_diligence_report_assembly():
    profile = {
        "company_name": "SpaceX",
        "company_id": "spacex",
        "industry_sector": "Aerospace",
        "total_funding_usd_m": 9800.0,
        "employee_count": 13000,
        "company_age_years": 22,
    }
    prediction = {
        "predicted_label": "High Growth",
        "probabilities": {"high_growth": 0.85, "stable": 0.12, "distressed": 0.03},
        "fused_features": {"funding_intensity": 445.45, "article_count": 12},
    }
    shap_exp = {
        "top_positive_drivers": {"funding_intensity": 0.42},
        "top_negative_drags": {"company_age_years": -0.05},
    }
    fairness = {
        "protected_attribute": "is_tier1_hub",
        "demographic_parity_gap": 0.02,
        "equal_opportunity_difference": 0.01,
        "fairness_threshold_met_dpd_05": True,
    }

    memo = DueDiligenceReportGenerator.generate_markdown_memo(profile, prediction, shap_exp, fairness)

    assert "Startup Market Analyzer (SMA) — Due Diligence Memo" in memo
    assert "High Growth" in memo
    assert "funding_intensity" in memo
    assert "APPROVED" in memo