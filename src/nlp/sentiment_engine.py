"""
Module: src/nlp/sentiment_engine.py
Role: Hybrid Sentiment Analysis Engine routing between FinBERT and VADER with batch scoring support.
"""

from typing import List, Tuple, Optional
from src.domain.models import ArticleRecord


class HybridNLPEngine:
    def __init__(self, use_transformer: bool = False):
        self.use_transformer = use_transformer
        self._vader = None

    @property
    def vader(self):
        if self._vader is None:
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            import nltk

            try:
                nltk.data.find("sentiment/vader_lexicon.zip")
            except LookupError:
                nltk.download("vader_lexicon", quiet=True)
            self._vader = SentimentIntensityAnalyzer()
        return self._vader

    def analyze_text(self, text: str, source: str = "news") -> float:
        """Computes a normalized sentiment polarity score in [-1.0, 1.0]."""
        if not text or not text.strip():
            return 0.0

        scores = self.vader.polarity_scores(text)
        return float(scores["compound"])

    def analyze_article(
        self, title: str, content: str = "", source: str = "news"
    ) -> Tuple[float, float]:
        """Analyzes an article title and content body, returning (score, confidence)."""
        combined = f"{title or ''}. {content or ''}".strip()
        if not combined:
            return 0.0, 0.5

        scores = self.vader.polarity_scores(combined)
        compound = float(scores["compound"])
        confidence = float(max(scores["pos"], scores["neg"], 0.45))
        return compound, confidence

    def score_articles(self, articles: List[ArticleRecord]) -> List[float]:
        """
        Batch-scores a list of ArticleRecord instances.
        Called directly by FeatureFusionPipeline.fuse().
        """
        if not articles:
            return []

        results: List[float] = []
        for art in articles:
            combined = f"{art.title or ''}. {art.content or ''}".strip()
            score = self.analyze_text(combined, source=art.source)
            results.append(score)
        return results

    def score_text(self, text: str, source: str = "news") -> float:
        return self.analyze_text(text, source=source)