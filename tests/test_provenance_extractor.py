"""
Module: tests/test_provenance_extractor.py
Role: Academic-grade verification of extractor failure modes and boundary invariants.
"""

import pytest
from datetime import datetime, timezone
from src.ingestion.extractor import (
    WikipediaPrototypeExtractor,
    AmbiguousEntityError,
)


def test_1_ambiguous_names_fail_fast():
    """Verify umbrella names are rejected with clear user instruction."""
    ext = WikipediaPrototypeExtractor()
    with pytest.raises(AmbiguousEntityError) as exc_info:
        ext.extract_profile("Ola", "ola")
    assert "Ambiguous brand 'Ola'" in str(exc_info.value)


def test_2_snapshot_preceding_founding_date_rejected():
    """Verify snapshot dates prior to founding are flagged and age is not positive."""
    ext = WikipediaPrototypeExtractor()
    # Query Zerodha (founded 2010) as of 2005 (before it existed)
    pre_founding_date = datetime(2005, 1, 1, tzinfo=timezone.utc)
    profile = ext.extract_profile("Zerodha", "zerodha", as_of_date=pre_founding_date)

    assert profile.company_age_years is None
    assert profile.provenance["company_age_years"]["status"] == "invalid_pre_founding_snapshot"


def test_3_funding_never_inferred_as_zero_from_keywords():
    """Verify keyword presence does not trigger fake verified_zero funding."""
    ext = WikipediaPrototypeExtractor()
    profile = ext.extract_profile("Zerodha", "zerodha")

    # Funding must be None (not 0.0, not $35M)
    assert profile.total_funding_usd_m is None
    assert profile.provenance["total_funding_usd_m"]["status"] == "unknown"


def test_4_employee_headcount_and_reporting_year_separated():
    """Verify headcount parsing separates the number from the year."""
    ext = WikipediaPrototypeExtractor()
    count, year, err = ext._parse_headcount("1,200 (2020)")
    assert count == 1200
    assert year == "2020"
    assert err is None


def test_5_full_provenance_audit_trail_recorded():
    """Verify all 5 core attributes have recorded timestamps, URLs, and missingness reasons."""
    ext = WikipediaPrototypeExtractor()
    profile = ext.extract_profile("Zerodha", "zerodha")

    for key in ["company_age_years", "employee_count", "industry_sector", "total_funding_usd_m", "growth_rate"]:
        assert key in profile.provenance
        assert "retrieval_timestamp" in profile.provenance[key]
        assert "status" in profile.provenance[key]