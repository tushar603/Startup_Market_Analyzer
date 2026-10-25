"""
Module: src/nlp/preprocessor.py
Role: Text sanitization, tokenization, and lemmatization using spaCy.
Concepts: NLP Pipeline, Regular Expressions, Morphological Analysis.
"""

import re
import spacy


class TextSanitizer:
    def __init__(self, spacy_model: str = "en_core_web_sm"):
        # Load lightweight spaCy model; disable heavy parser and NER to optimize OS memory and CPU
        self._nlp = spacy.load(spacy_model, disable=["parser", "ner"])

    def clean_text(self, text: str) -> str:
        """
        Strips noise (URLs, user mentions, hashtags, punctuation) and applies lemmatization.
        """
        if not text or not isinstance(text, str):
            return ""

        # Remove HTTP/HTTPS URLs
        text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
        # Remove social handles and hashtags
        text = re.sub(r"[@#][A-Za-z0-9_]+", "", text)
        # Remove non-alphanumeric symbols except spaces
        text = re.sub(r"[^A-Za-z0-9\s]", " ", text)

        doc = self._nlp(text)
        lemmatized_tokens = [
            token.lemma_.lower().strip()
            for token in doc
            if not token.is_stop and len(token.lemma_.strip()) > 1
        ]
        return " ".join(lemmatized_tokens)