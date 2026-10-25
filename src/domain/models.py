"""
Module: src/domain/models.py
Role: Core Domain Entities with Provenance Tracking, Invariant Validation, and UTC Standardization.
"""

from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from typing import Optional, Dict, Any


def ensure_utc(dt: Any) -> datetime:
    """Enforces timezone-aware UTC datetime across all domain entity boundaries."""
    if isinstance(dt, str):
        dt = datetime.fromisoformat(dt.replace("Z", "+00:00"))
    if not isinstance(dt, datetime):
        raise TypeError(f"Expected datetime or ISO string, got {type(dt).__name__}")
    if dt.tzinfo is None:
        raise ValueError("Naive datetimes are forbidden. All timestamps must be timezone-aware (UTC).")
    return dt.astimezone(timezone.utc)


@dataclass
class AttributeProvenance:
    """Audit trail documenting the evidence source for a financial or firmographic attribute."""
    value: Any
    source_url: Optional[str] = None
    retrieval_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    as_of_date: Optional[str] = None
    status: str = "unknown"  # "verified_zero", "reported", "unknown"
    confidence: float = 0.0

    def to_dict(self) -> dict:
        return {
            "value": self.value,
            "source_url": self.source_url,
            "retrieval_timestamp": self.retrieval_timestamp.isoformat(),
            "as_of_date": self.as_of_date,
            "status": self.status,
            "confidence": self.confidence,
        }


@dataclass
class CompanyProfile:
    company_id: str
    company_name: str
    snapshot_date: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Core Attributes (Nullable if unverified)
    total_funding_usd_m: Optional[float] = None
    employee_count: Optional[int] = None
    growth_rate: Optional[float] = None
    company_age_years: Optional[float] = None  # Exact floating years (e.g. 16.2 yrs)
    industry_sector: Optional[str] = None
    headquarters: Optional[str] = None
    is_bootstrapped: Optional[bool] = None

    # Lineage & Provenance Metadata
    provenance: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    growth_category: Optional[int] = None
    observed_valuation_usd_m: Optional[float] = None

    def __post_init__(self):
        if not self.company_id or not isinstance(self.company_id, str):
            raise ValueError("company_id must be a non-empty string.")
        self.company_id = self.company_id.strip().lower()
        self.company_name = self.company_name.strip()
        self.snapshot_date = ensure_utc(self.snapshot_date)

        if self.growth_category is not None and self.growth_category not in (0, 1, 2):
            raise ValueError(f"growth_category must be 0, 1, 2, or None. Got: {self.growth_category}")
        if self.total_funding_usd_m is not None and self.total_funding_usd_m < 0:
            raise ValueError("Funding cannot be negative.")
        if self.employee_count is not None and self.employee_count < 0:
            raise ValueError("Employee count cannot be negative.")

    def to_dict(self) -> dict:
        data = asdict(self)
        data["snapshot_date"] = self.snapshot_date.isoformat()
        return data


@dataclass
class ArticleRecord:
    article_id: str
    company_id: str
    source: str
    title: str
    content: str
    published_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.article_id or not self.company_id:
            raise ValueError("article_id and company_id must be non-empty strings.")
        self.company_id = self.company_id.strip().lower()
        self.article_id = self.article_id.strip()
        self.published_at = ensure_utc(self.published_at)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["published_at"] = self.published_at.isoformat()
        return data