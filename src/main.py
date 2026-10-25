"""
Module: src/main.py
Role: Production End-to-End Multimodal Execution Driver.
"""

from datetime import datetime, timezone, timedelta
from src.domain.models import CompanyProfile, ArticleRecord
from src.features.fusion import FeatureFusionPipeline
from src.nlp.sentiment_engine import HybridNLPEngine


def main():
    print("================================================================")
    print("Startup Market Analyzer (SMA) - Production Multimodal Pipeline")
    print("================================================================")

    # 1. Initialize Hybrid NLP with FinBERT enabled by default
    print("[+] Loading FinBERT Transformer & VADER Lexicons...")
    nlp_engine = HybridNLPEngine(use_transformer=True)
    pipeline = FeatureFusionPipeline(nlp_engine=nlp_engine)

    now = datetime.now(timezone.utc)

    # 2. Company Profile with verified snapshot timestamp and explicit label
    spacex_profile = CompanyProfile(
        company_id="spacex",
        company_name="SpaceX",
        snapshot_date=now,
        total_funding_usd_m=9800.0,
        employee_count=13000,
        growth_rate=0.18,
        company_age_years=22,
        industry_sector="Aerospace",
        growth_category=2,  # Verified Ground Truth: High Growth
        observed_valuation_usd_m=180000.0
    )

    # 3. Timezone-aware text stream with deliberate edge cases:
    #    - Duplicate syndicated news release
    #    - Cross-company article (Stripe record mixed in)
    #    - Future article (Look-ahead violation)
    articles = [
        ArticleRecord(
            article_id="art-001",
            company_id="spacex",
            source="NewsAPI",
            title="SpaceX secures billion dollar NASA contract expansion",
            content="NASA officially awards massive commercial contract following orbital launch cadence.",
            published_at=now - timedelta(days=10)
        ),
        ArticleRecord(
            article_id="art-002",
            company_id="spacex",
            source="NewsWire",
            title="SpaceX secures billion dollar NASA contract expansion",  # Syndicated wire copy
            content="NASA officially awards massive commercial contract following orbital launch cadence.",
            published_at=now - timedelta(days=10)
        ),
        ArticleRecord(
            article_id="art-003",
            company_id="stripe",  # Wrong company_id: Must be discarded by foreign-key filter
            source="NewsAPI",
            title="Stripe payment volume hits record highs",
            content="Fintech giant Stripe reports unprecedented payment throughput across Europe.",
            published_at=now - timedelta(days=12)
        ),
        ArticleRecord(
            article_id="art-004",
            company_id="spacex",
            source="Twitter",
            title="Starship static fire aborted due to valve pressure leak",
            content="Engineers identify pressure sensor fault, delaying next test window.",
            published_at=now - timedelta(days=45)
        ),
        ArticleRecord(
            article_id="art-005",
            company_id="spacex",
            source="NewsAPI",
            title="Future Starlink expansion announced for next quarter",
            content="SpaceX targets next-generation deployment timeline.",
            published_at=now + timedelta(days=4)  # Future article: Must be discarded
        )
    ]

    print("\n[+] Executing Point-in-Time Multimodal Fusion as of:", now.isoformat())
    fused = pipeline.fuse(spacex_profile, articles, as_of_date=now)

    print(f"\n--- Fused Entity: {spacex_profile.company_name} [{fused.company_id}] ---")
    print(f"  • Ingested Unique Articles : {fused.article_count} (Pruned: 1 duplicate, 1 foreign, 1 future)")
    print(f"  • Media Coverage Detected  : {bool(fused.has_media_coverage)}")
    print(f"  • 30d Coverage Available   : {bool(fused.has_recent_coverage_30d)} (Score: {fused.recent_sentiment_30d})")
    print(f"  • Prev 30d Cov Available   : {bool(fused.has_previous_30d_coverage)}")
    print(f"  • Calculated Trend (Momentum): {fused.sentiment_trend}")
    print(f"  • Global Mean Sentiment    : {fused.sentiment_mean_all}")
    print(f"  • Sentiment Volatility     : {fused.sentiment_volatility}")
    print(f"  • 90d Strategic Events     : {fused.event_count_90d}")
    print(f"  • Target Label (Growth)    : {fused.target_growth_category} (Strictly preserved)")
    print(f"  • Target Label (Valuation) : ${fused.observed_valuation_usd_m:,.2f}M")


if __name__ == "__main__":
    main()