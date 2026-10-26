"""Startup Market Analyzer sentiment engine.

FinBERT (ProsusAI/finbert) analyzes financial/news text when
``use_transformer=True``. NLTK VADER handles social posts and remains
available for legacy ``use_transformer=False`` callers.

The public ``analyze_text(text, source=...) -> float`` interface preserves
scores in [-1, 1], so existing feature-fusion callers need not change.

FinBERT scores: P(positive) - P(negative).
VADER scores: the NLTK compound sentiment value.

If FinBERT cannot be loaded, an explicit exception is raised; financial
news is NEVER silently scored with VADER in FinBERT mode.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Optional, Sequence

from nltk.sentiment import SentimentIntensityAnalyzer

logger = logging.getLogger(__name__)


class HybridNLPEngine:
    """Route financial news to FinBERT and social text to VADER.

    Args:
        use_transformer: Enable real FinBERT inference for financial/news text.
        model_name: Hugging Face model ID or directory with model weights.
        batch_size: Number of texts per FinBERT forward pass.
        max_length: Maximum tokens, capped at 512 for BERT.
        device: Optional torch device ("cpu", "cuda", etc.). Auto-selected if None.

    Note:
        FinBERT is loaded lazily on the first financial-text inference.
    """

    FINBERT_MODEL = "ProsusAI/finbert"
    SOCIAL_SOURCES = frozenset({
        "social", "social_media", "social post", "social_post",
        "twitter", "x", "x/twitter", "tweet", "reddit", "forum",
        "stocktwits", "facebook", "instagram", "comment", "comments",
    })

    def __init__(
        self,
        use_transformer: bool = False,
        model_name: str = FINBERT_MODEL,
        batch_size: int = 8,
        max_length: int = 512,
        device: Optional[str] = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if max_length < 1:
            raise ValueError("max_length must be >= 1")

        self.use_transformer = bool(use_transformer)
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = min(max_length, 512)
        self.requested_device = device
        # Only initialize VADER if/when a VADER-routed text is analyzed.
        # FinBERT does not require downloading the NLTK lexicon.
        self.vader = None

        self._tokenizer = None
        self._model = None
        self._device = None
        self._torch = None
        self._label_index: Dict[str, int] = {}
        self._load_lock = threading.Lock()

    def _vader_score(self, text: str) -> float:
        if self.vader is None:
            try:
                self.vader = SentimentIntensityAnalyzer()
            except LookupError as exc:
                raise RuntimeError(
                    "VADER lexicon not found. Install it with: "
                    "python -c \"import nltk; nltk.download('vader_lexicon')\""
                ) from exc
        return float(self.vader.polarity_scores(text)["compound"])

    def _backend_for(self, source: Optional[str]) -> str:
        if not self.use_transformer:
            return "vader"
        normalized = (source or "").strip().lower()
        return "vader" if normalized in self.SOCIAL_SOURCES else "finbert"

    def _ensure_finbert(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            try:
                import torch
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
            except ImportError as exc:
                raise RuntimeError(
                    "FinBERT requires torch and transformers. Install them using "
                    "python -m pip install torch transformers"
                ) from exc

            try:
                tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
                device = self.requested_device or (
                    "cuda" if torch.cuda.is_available() else "cpu"
                )
                model.to(device)
                model.eval()
            except Exception as exc:
                raise RuntimeError(
                    f"Could not load FinBERT '{self.model_name}'. "
                    "Check model access, connectivity and local disk space. "
                    "FinBERT has NOT been replaced silently with VADER."
                ) from exc

            labels = {
                str(label).strip().lower(): int(index)
                for index, label in model.config.id2label.items()
            }
            # Some saved model configs use generic labels. The officially
            # published ProsusAI checkpoint declares 0=positive, 1=negative,
            # 2=neutral. Do NOT assume this mapping for other checkpoints.
            if set(labels) != {"positive", "negative", "neutral"}:
                if self.model_name == self.FINBERT_MODEL and set(labels) == {
                    "label_0", "label_1", "label_2"
                }:
                    labels = {"positive": 0, "negative": 1, "neutral": 2}
                else:
                    raise RuntimeError(
                        f"Unexpected sentiment label mapping for {self.model_name}: "
                        f"{model.config.id2label!r}"
                    )

            self._torch = torch
            self._tokenizer = tokenizer
            self._model = model
            self._device = device
            self._label_index = labels
            logger.info("FinBERT loaded: %s on %s", self.model_name, device)

    def _finbert_details_batch(self, texts: Sequence[str]) -> List[Dict[str, Any]]:
        """Perform real batched transformer inference (no mock probabilities)."""
        if not texts:
            return []
        self._ensure_finbert()
        details: List[Dict[str, Any]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            encoded = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: tensor.to(self._device) for key, tensor in encoded.items()}
            with self._torch.inference_mode():
                logits = self._model(**encoded).logits
                probabilities = self._torch.softmax(logits, dim=-1).cpu().tolist()

            for row in probabilities:
                p = {name: float(row[idx]) for name, idx in self._label_index.items()}
                score = p["positive"] - p["negative"]
                details.append({
                    "score": float(max(-1.0, min(1.0, score))),
                    "label": max(p, key=p.get),
                    "probabilities": p,
                    "backend": "finbert",
                    "model": self.model_name,
                })
        return details

    def analyze_text_details(
        self, text: str, source: Optional[str] = None
    ) -> Dict[str, Any]:
        """Return polarity, backend, and FinBERT probabilities (when available)."""
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        backend = self._backend_for(source)
        if not text.strip():
            return {"score": 0.0, "label": "neutral", "backend": "no_text", "model": None}
        if backend == "finbert":
            return self._finbert_details_batch([text])[0]

        score = self._vader_score(text)
        label = "positive" if score >= 0.05 else "negative" if score <= -0.05 else "neutral"
        return {"score": score, "label": label, "backend": "vader", "model": "nltk-vader"}

    def analyze_text(self, text: str, source: Optional[str] = None) -> float:
        """Existing project interface: float sentiment polarity in [-1, 1]."""
        return self.analyze_text_details(text, source=source)["score"]

    def analyze_batch(
        self,
        texts: Sequence[str],
        sources: Optional[Sequence[Optional[str]]] = None,
    ) -> List[float]:
        """Score many texts efficiently, preserving input order."""
        texts = list(texts)
        if sources is None:
            sources = [None] * len(texts)
        else:
            sources = list(sources)
        if len(sources) != len(texts):
            raise ValueError("sources must have the same length as texts")
        if any(not isinstance(text, str) for text in texts):
            raise TypeError("all texts must be strings")

        scores = [0.0] * len(texts)
        finbert_indices = []
        finbert_texts = []
        for i, (text, source) in enumerate(zip(texts, sources)):
            if not text.strip():
                continue
            if self._backend_for(source) == "finbert":
                finbert_indices.append(i)
                finbert_texts.append(text)
            else:
                scores[i] = self._vader_score(text)

        for index, result in zip(finbert_indices, self._finbert_details_batch(finbert_texts)):
            scores[index] = result["score"]
        return scores

    def analyze_articles(self, articles: Sequence[Any]) -> List[float]:
        """Convenience method for ArticleRecord objects (title, content, source)."""
        texts = [f"{a.title}. {a.content}".strip() for a in articles]
        sources = [a.source for a in articles]
        return self.analyze_batch(texts, sources)
