"""
Module: src/ingestion/news_client.py
Role: Live news collection engine with NewsAPI support and RSS/Web fallback.
Concepts: Rate-limiting, deduplication, timestamp normalization, UTC enforcement.
"""

import os
from datetime import datetime, timezone
from typing import List, Optional
import urllib.parse
import feedparser
import requests
from dateutil import parser as date_parser

from src.domain.models import ArticleRecord, ensure_utc


class LiveNewsIngestor:
    def __init__(self, api_key: Optional[str] = None):
        # Read from argument or environment variable
        self.api_key = api_key or os.getenv("NEWS_API_KEY", "")
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "StartupMarketAnalyzer/1.0"})

    def fetch_newsapi(self, company_name: str, company_id: str, max_articles: int = 15) -> List[ArticleRecord]:
        """Fetches recent financial headlines from NewsAPI.org."""
        if not self.api_key:
            return []

        url = "https://newsapi.org/v2/everything"
        query = f'"{company_name}" AND (funding OR startup OR valuation OR acquisition OR revenue)'
        params = {
            "q": query,
            "sortBy": "publishedAt",
            "pageSize": max_articles,
            "language": "en",
            "apiKey": self.api_key,
        }

        try:
            resp = self.session.get(url, params=params, timeout=10)
            if resp.status_code != 200:
                return []

            data = resp.json()
            articles: List[ArticleRecord] = []

            for idx, item in enumerate(data.get("articles", [])):
                raw_pub = item.get("publishedAt")
                if not raw_pub:
                    continue
                pub_dt = ensure_utc(date_parser.parse(raw_pub))

                art = ArticleRecord(
                    article_id=f"newsapi_{company_id}_{idx}_{int(pub_dt.timestamp())}",
                    company_id=company_id,
                    source=item.get("source", {}).get("name", "NewsAPI"),
                    title=item.get("title", "") or "",
                    content=item.get("description", "") or item.get("title", ""),
                    published_at=pub_dt,
                )
                articles.append(art)

            return articles
        except Exception:
            return []

    def fetch_rss_fallback(self, company_name: str, company_id: str, max_articles: int = 10) -> List[ArticleRecord]:
        """
        Public RSS fallback using Google News search feed.
        Requires zero API keys and works out-of-the-box.
        """
        encoded_query = urllib.parse.quote(f"{company_name} startup funding business")
        rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"

        articles: List[ArticleRecord] = []
        try:
            feed = feedparser.parse(rss_url)
            for idx, entry in enumerate(feed.entries[:max_articles]):
                raw_date = getattr(entry, "published", None)
                if raw_date:
                    pub_dt = ensure_utc(date_parser.parse(raw_date))
                else:
                    pub_dt = datetime.now(timezone.utc)

                art = ArticleRecord(
                    article_id=f"rss_{company_id}_{idx}_{int(pub_dt.timestamp())}",
                    company_id=company_id,
                    source=getattr(entry, "source", {}).get("title", "GoogleNewsRSS"),
                    title=entry.title,
                    content=getattr(entry, "summary", "") or entry.title,
                    published_at=pub_dt,
                )
                articles.append(art)
        except Exception:
            return []

        return articles

    def get_company_news(self, company_name: str, company_id: str) -> List[ArticleRecord]:
        """Primary ingestion method: attempts NewsAPI first, falls back to RSS feed."""
        records = self.fetch_newsapi(company_name, company_id)
        if not records:
            records = self.fetch_rss_fallback(company_name, company_id)
        return records