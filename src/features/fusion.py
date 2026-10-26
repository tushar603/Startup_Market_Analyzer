"""
Module: src/features/fusion.py
Role: Point-in-Time Multimodal Feature Synthesis with Explicit Target Isolation.
"""

import hashlib
import re
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Set, Tuple, Dict, Any
import numpy as np

from src.domain.models import CompanyProfile, ArticleRecord, ensure_utc
from src.nlp.sentiment_engine import HybridNLPEngine


# Canonical event taxonomy mapping
EVENT_TAXONOMY = {
    "funding": "FINANCING_EVENT",
    "series a": "FINANCING_EVENT",
    "series b": "FINANCING_EVENT",
    "series c": "FINANCING_EVENT",
    "series d": "FINANCING_EVENT",
    "contract": "COMMERCIAL_CONTRACT",
    "launch": "PRODUCT_OR_SERVICE_LAUNCH",
    "acquire": "MA_ACTIVITY",
    "acquisition": "MA_ACTIVITY",
    "partnership": "COMMERCIAL_PARTNERSHIP",
}

EVENT_REGEX = re.compile(
    r"\b(launch|contract|funding|series [a-d]|acquire|acquisition|partnership)\b",
    re.IGNORECASE,
)


@dataclass
class FusedFeatureVector:
    # 1. Identifiers & Lineage Metadata (Never used as ML features)
    company_id: str
    as_of_date: str

    # 2. Historical Structured Signals (Available <= as_of_date)
    funding_intensity: float
    historical_growth_rate: float
    employee_count: int
    company_age_years: int
    is_funding_missing: int
    is_growth_missing: int
    is_employees_missing: int

    # 3. Unstructured Media & Sentiment Signals (Available <= as_of_date)
    article_count: int
    has_media_coverage: int
    has_recent_coverage_30d: int
    has_previous_30d_coverage: int
    sentiment_mean_all: float
    recent_sentiment_30d: Optional[float]
    sentiment_trend: Optional[float]
    sentiment_volatility: float
    positive_ratio: float
    negative_ratio: float
    regex_event_signals_90d: int

    # 4. Independent Future Targets (Isolated for loss computation)
    target_future_growth_category: Optional[int]
    observed_future_valuation_usd_m: Optional[float]

    @property
    def mean_sentiment(self) -> float:
        """Convenience alias returning the overall computed mean sentiment."""
        return self.sentiment_mean_all

    def to_features_dict(self) -> Dict[str, Any]:
        """Extracts strictly historical features for Model input X."""
        excluded = {
            "company_id",
            "as_of_date",
            "target_future_growth_category",
            "observed_future_valuation_usd_m",
        }
        raw = asdict(self)
        features = {k: v for k, v in raw.items() if k not in excluded}

        # Guard: If no coverage exists in the recent 30-day window,
        # recent_sentiment_30d is numerically set to 0.0 for tree input,
        # but has_recent_coverage_30d = 0 informs the model that coverage was absent.
        if features["recent_sentiment_30d"] is None:
            features["recent_sentiment_30d"] = 0.0
        if features["sentiment_trend"] is None:
            features["sentiment_trend"] = 0.0

        return features

    def to_targets_dict(self) -> Dict[str, Any]:
        """Extracts strictly independent outcomes for Model Target y."""
        return {
            "growth_category": self.target_future_growth_category,
            "valuation_usd_m": self.observed_future_valuation_usd_m,
        }


class FeatureFusionPipeline:
    def __init__(self, nlp_engine: Optional[HybridNLPEngine] = None):
        self.nlp = nlp_engine or HybridNLPEngine(use_transformer=False)

    def _canonical_hash(self, art: ArticleRecord) -> str:
        """
        Deduplicates syndicated press releases by normalizing whitespace, 
        casing, and punctuation across the full title and body.
        """
        raw_text = f"{art.title} {art.content}".lower()
        cleaned_text = re.sub(r"\W+", " ", raw_text).strip()
        payload = f"{art.company_id}:{cleaned_text}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def fuse(
        self,
        profile: CompanyProfile,
        articles: List[ArticleRecord],
        as_of_date: Optional[datetime] = None,
    ) -> FusedFeatureVector:
        cutoff = ensure_utc(as_of_date) if as_of_date else datetime.now(timezone.utc)

        if profile.snapshot_date > cutoff:
            raise ValueError(
                f"Look-ahead violation: Profile snapshot ({profile.snapshot_date.isoformat()}) "
                f"is strictly newer than feature cutoff ({cutoff.isoformat()})."
            )

        # 1. Foreign Key Verification, Point-in-Time Pruning & Canonical Deduplication
        seen_hashes: Set[str] = set()
        valid_articles: List[ArticleRecord] = []

        for a in articles:
            if a.company_id != profile.company_id:
                continue
            if a.published_at <= cutoff:
                art_hash = self._canonical_hash(a)
                if art_hash not in seen_hashes:
                    seen_hashes.add(art_hash)
                    valid_articles.append(a)

        valid_articles.sort(key=lambda x: x.published_at)

        # 2. Structured Metrics with Explicit Missingness Indicators
        is_funding_missing = 1 if profile.total_funding_usd_m is None else 0
        is_growth_missing = 1 if profile.growth_rate is None else 0
        is_employees_missing = 1 if profile.employee_count is None else 0

        age = profile.company_age_years if profile.company_age_years and profile.company_age_years > 0 else 1
        funding = profile.total_funding_usd_m if profile.total_funding_usd_m is not None else 0.0
        funding_intensity = funding / age
        
        # Historical growth: default to 0.0 for tree input while preserving missingness flag
        growth_rate = profile.growth_rate if profile.growth_rate is not None else 0.0
        employee_count = profile.employee_count if profile.employee_count is not None else 0

        # 3. Batch NLP Sentiment Scoring
        sentiment_scores = self.nlp.score_articles(valid_articles)

        # 4. Temporal Aggregation Windows
        d30_cutoff = cutoff - timedelta(days=30)
        d60_cutoff = cutoff - timedelta(days=60)
        d90_cutoff = cutoff - timedelta(days=90)

        scores_30d: List[float] = []
        scores_prev_30d: List[float] = []
        unique_canonical_events: Set[Tuple[str, str, Any]] = set()
        pos_count = 0
        neg_count = 0

        for art, score in zip(valid_articles, sentiment_scores):
            if score > 0.05:
                pos_count += 1
            elif score < -0.05:
                neg_count += 1

            if art.published_at >= d30_cutoff:
                scores_30d.append(score)
            elif art.published_at >= d60_cutoff:
                scores_prev_30d.append(score)

            # Normalized Canonical Event Extraction
            if art.published_at >= d90_cutoff:
                matches = EVENT_REGEX.findall(f"{art.title}. {art.content}")
                for raw_match in matches:
                    canonical_type = EVENT_TAXONOMY.get(raw_match.lower(), "OTHER_EVENT")
                    event_tuple = (art.company_id, canonical_type, art.published_at.date())
                    unique_canonical_events.add(event_tuple)

        # 5. Missingness Flags & Aggregation
        total_arts = len(valid_articles)
        has_coverage = 1 if total_arts > 0 else 0
        has_recent = 1 if len(scores_30d) > 0 else 0
        has_prev = 1 if len(scores_prev_30d) > 0 else 0

        mean_sentiment = float(np.mean(sentiment_scores)) if total_arts > 0 else 0.0
        volatility = float(np.std(sentiment_scores)) if total_arts > 1 else 0.0

        recent_30d = float(np.mean(scores_30d)) if has_recent else None
        prev_30d = float(np.mean(scores_prev_30d)) if has_prev else None

        # Momentum trend: strictly defined ONLY when both time windows exist
        sentiment_trend: Optional[float] = None
        if recent_30d is not None and prev_30d is not None:
            sentiment_trend = round(recent_30d - prev_30d, 4)

        pos_ratio = (pos_count / total_arts) if total_arts > 0 else 0.0
        neg_ratio = (neg_count / total_arts) if total_arts > 0 else 0.0

        return FusedFeatureVector(
            company_id=profile.company_id,
            as_of_date=cutoff.isoformat(),
            funding_intensity=round(funding_intensity, 4),
            historical_growth_rate=round(growth_rate, 4),
            employee_count=employee_count,
            company_age_years=age,
            is_funding_missing=is_funding_missing,
            is_growth_missing=is_growth_missing,
            is_employees_missing=is_employees_missing,
            article_count=total_arts,
            has_media_coverage=has_coverage,
            has_recent_coverage_30d=has_recent,
            has_previous_30d_coverage=has_prev,
            sentiment_mean_all=round(mean_sentiment, 4),
            recent_sentiment_30d=round(recent_30d, 4) if recent_30d is not None else None,
            sentiment_trend=sentiment_trend,
            sentiment_volatility=round(volatility, 4),
            positive_ratio=round(pos_ratio, 4),
            negative_ratio=round(neg_ratio, 4),
            regex_event_signals_90d=len(unique_canonical_events),
            target_future_growth_category=profile.growth_category,
            observed_future_valuation_usd_m=profile.observed_valuation_usd_m,
        )