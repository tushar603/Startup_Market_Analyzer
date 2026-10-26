"""Wikipedia infobox retrieval/parsing helpers for Startup Market Analyzer.

Compatible with the method signatures in the user's newer extract_profile snippet.
Uses the MediaWiki Action API and Python stdlib + requests only.
No changes to model, dashboard, or persistence layers.
"""
from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import quote

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

LOG = logging.getLogger(__name__)
API = "https://en.wikipedia.org/w/api.php"

class WikipediaInfoboxClient:
    def __init__(self, session: Optional[requests.Session] = None):
        self.session = session or requests.Session()
        # Wikimedia prefers client/version + real contact information.
        contact = os.getenv("WIKIMEDIA_CONTACT", "https://github.com/tushar603/Startup-Market-Analyzer")
        self.session.headers.update({
            "User-Agent": f"StartupMarketAnalyzer/1.0 ({contact}) requests/{requests.__version__}",
            "Accept": "application/json",
        })
        # Avoid silently hammering the API. Only retry transient failures.
        retry = Retry(total=2, backoff_factor=0.5,
                      status_forcelist=(429, 500, 502, 503, 504),
                      allowed_methods=frozenset(["GET"]),
                      respect_retry_after_header=True)
        self.session.mount("https://", HTTPAdapter(max_retries=retry))

    def _json(self, **params):
        params.update({"format": "json", "formatversion": 2, "maxlag": 5})
        try:
            res = self.session.get(API, params=params, timeout=(5, 18))
            res.raise_for_status()
            payload = res.json()
        except (requests.RequestException, ValueError) as exc:
            raise RuntimeError(f"Wikipedia request failed: {exc}") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("Wikipedia API returned an unexpected JSON structure")
        if "error" in payload:
            e = payload["error"]
            raise RuntimeError(f"Wikipedia API error: {e.get('code', 'unknown')}: {e.get('info', '')}")
        return payload

    @staticmethod
    def _name_key(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", value.casefold())

    @classmethod
    def _plausible_match(cls, query: str, title: str) -> bool:
        # Conservative: similar article title, rather than selecting arbitrary search result.
        q, t = cls._name_key(query), cls._name_key(title)
        return bool(q and (q == t or (len(q) >= 5 and (t.startswith(q) or q.startswith(t)))))

    def _retrieve_page(self, title: str):
        data = self._json(action="query", prop="revisions|info|pageprops",
                          inprop="url", ppprop="disambiguation", titles=title,
                          redirects=1, rvprop="content", rvslots="main")
        pages = data.get("query", {}).get("pages", [])
        if not pages or pages[0].get("missing") or pages[0].get("invalid"):
            return None
        page = pages[0]
        if "disambiguation" in page.get("pageprops", {}):
            return None
        revisions = page.get("revisions", [])
        if not revisions:
            return None
        content = revisions[0].get("slots", {}).get("main", {}).get("content", "")
        if not isinstance(content, str) or not content:
            return None
        title = page.get("title", title)
        url = page.get("fullurl") or "https://en.wikipedia.org/wiki/" + quote(title.replace(" ", "_"))
        return title, url, content

    @staticmethod
    def _template_span(text: str, start: int):
        """Return balanced {{...}} text, honoring nested double-brace templates."""
        depth = 0
        i = start
        while i < len(text) - 1:
            pair = text[i:i + 2]
            if pair == "{{":
                depth += 1
                i += 2
            elif pair == "}}":
                depth -= 1
                i += 2
                if depth == 0:
                    return text[start:i]
            else:
                i += 1
        return None

    @classmethod
    def _get_infobox(cls, content: str):
        for match in re.finditer(r"\{\{\s*Infobox[ _]+([\w -]+)", content, flags=re.I):
            kind = match.group(1).strip().casefold()
            # Avoid mistaking non-company pages for a company.
            if not any(k in kind for k in ("company", "organization", "business", "website")):
                continue
            template = cls._template_span(content, match.start())
            if template:
                return template
        return None

    def _fetch_infobox_wikitext(self, query: str):
        """Return (matched_title, page_url, raw_infobox, error_message)."""
        query = (query or "").strip()
        if not query:
            return None, None, None, "Empty company query"
        try:
            candidate = self._retrieve_page(query)
            if candidate and self._plausible_match(query, candidate[0]):
                title, url, content = candidate
                infobox = self._get_infobox(content)
                if infobox:
                    return title, url, infobox, None

            result = self._json(action="query", list="search", srsearch=query + " company",
                                srnamespace=0, srlimit=6)
            found = result.get("query", {}).get("search", [])
            for item in found:
                title = item.get("title", "")
                if not self._plausible_match(query, title):
                    continue
                candidate = self._retrieve_page(title)
                if candidate:
                    canonical, url, content = candidate
                    if self._plausible_match(query, canonical):
                        infobox = self._get_infobox(content)
                        if infobox:
                            return canonical, url, infobox, None
            return None, None, None, (
                f"No matching English Wikipedia company infobox found for {query!r}. "
                "The page may be absent, ambiguous, or use an unsupported infobox."
            )
        except RuntimeError as exc:
            LOG.warning("Wikipedia extraction failed for %r: %s", query, exc)
            return None, None, None, str(exc)

    @staticmethod
    def _top_level_parts(template: str):
        """Split only on | outside nested {{templates}} and [[links]]."""
        parts = []
        start = 2
        i = 2
        tpl_depth = 0
        link_depth = 0
        end = len(template) - 2
        while i < end:
            pair = template[i:i + 2]
            if pair == "{{":
                tpl_depth += 1; i += 2; continue
            if pair == "}}" and tpl_depth:
                tpl_depth -= 1; i += 2; continue
            if pair == "[[":
                link_depth += 1; i += 2; continue
            if pair == "]]" and link_depth:
                link_depth -= 1; i += 2; continue
            if template[i] == "|" and tpl_depth == 0 and link_depth == 0:
                parts.append(template[start:i]); start = i + 1
            i += 1
        parts.append(template[start:end])
        return parts

    @staticmethod
    def _clean_markup(raw: str) -> str:
        s = re.sub(r"<!--.*?-->", "", raw, flags=re.S)
        s = re.sub(r"<ref\b[^>]*>.*?</ref>|<ref\b[^>]*/>", "", s, flags=re.S | re.I)
        s = re.sub(r"\[\[([^\]]+)\]\]", lambda m: m.group(1).split("|")[-1], s)
        s = re.sub(r"\[https?://\S+\s+([^\]]+)\]", r"\1", s)
        s = re.sub(r"\{\{\s*formatnum\s*:\s*([\d,]+)\s*}}", r"\1", s, flags=re.I)
        s = re.sub(r"\{\{\s*(?:as of|asof)\s*\|\s*(\d{4})(?:\|[^{}]*)?}}", r"(\1)", s, flags=re.I)
        # Retain date template, as the founding parser needs its exact components.
        s = re.sub(r"\{\{(?!\s*(?:start[ _]date|birth[ _]date))[^{}]*}}", " ", s, flags=re.I)
        s = re.sub(r"<[^>]+>", " ", s)
        s = s.replace("'''", "").replace("''", "")
        return re.sub(r"[ \t]+", " ", s).strip()

    @classmethod
    def _parse_infobox_field(cls, infobox: str, field_names):
        """Return (raw, clean) or None. Handles nested pipes inside Wiki templates."""
        if not infobox:
            return None
        keys = {k.casefold().strip() for k in field_names}
        for piece in cls._top_level_parts(infobox):
            if "=" not in piece:
                continue
            key, value = piece.split("=", 1)
            if key.strip().casefold() in keys:
                raw = value.strip()
                clean = cls._clean_markup(raw)
                if clean:
                    return raw, clean
        return None

    @staticmethod
    def _parse_exact_founding_date(clean_found: str):
        """Return (UTC datetime | None, source_date | None, precision)."""
        s = clean_found or ""
        # Date and age template, including named/unnamed padding or extra args.
        m = re.search(r"\{\{\s*(?:start[ _]date(?:[ _]and[ _]age)?|birth[ _]date(?:[ _]and[ _]age)?)\s*\|\s*(\d{4})\s*\|\s*(\d{1,2})\s*\|\s*(\d{1,2})(?:\||\s*}})", s, re.I)
        if m:
            try:
                dt = datetime(int(m[1]), int(m[2]), int(m[3]), tzinfo=timezone.utc)
                return dt, dt.strftime("%Y-%m-%d"), "exact"
            except ValueError:
                return None, None, "invalid"
        for pattern, fmt in [
            (r"\b\d{4}-\d{2}-\d{2}\b", "%Y-%m-%d"),
            (r"\b\d{1,2} [A-Za-z]+ \d{4}\b", "%d %B %Y"),
            (r"\b[A-Za-z]+ \d{1,2},? \d{4}\b", "%B %d %Y"),
        ]:
            for match in re.finditer(pattern, s):
                try:
                    dt = datetime.strptime(match.group(0), fmt).replace(tzinfo=timezone.utc)
                    return dt, dt.strftime("%Y-%m-%d"), "exact"
                except ValueError:
                    continue
        # Only a year is known; Jan 1 is a proxy rather than a precise founding date.
        m = re.search(r"\b(18\d{2}|19\d{2}|20\d{2})\b", s)
        if m:
            year = int(m[1]); dt = datetime(year, 1, 1, tzinfo=timezone.utc)
            return dt, str(year), "year"
        return None, None, "unparseable"

    @staticmethod
    def _parse_headcount(clean_emp: str):
        """Return (employee_count | None, reporting_year | None, error | None)."""
        s = (clean_emp or "").strip()
        if not s:
            return None, None, "Empty employee field"
        year_match = re.search(r"\((?:as of\s*)?((?:19|20)\d{2})\)", s, re.I)
        year = year_match.group(1) if year_match else None
        # Remove explicitly parenthesized reporting year before numeric parsing.
        s = re.sub(r"\((?:as of\s*)?(?:19|20)\d{2}\)", " ", s, flags=re.I)
        s = re.sub(r"^\s*(?:approx\.?|approximately|about|over|more than|~|≈)\s*", "", s, flags=re.I)
        m = re.search(r"\b([\d,]+(?:\.\d+)?)\s*(million|thousand|m|k)?\b", s, flags=re.I)
        if not m:
            return None, year, "No valid numeric headcount"
        try:
            num = float(m.group(1).replace(",", ""))
            unit = (m.group(2) or "").casefold()
            multiplier = {"million": 1_000_000, "m": 1_000_000, "thousand": 1_000, "k": 1_000}.get(unit, 1)
            count = num * multiplier
            if not count.is_integer() or count <= 0 or count > 100_000_000:
                return None, year, "Implausible or noninteger employee count"
            # An isolated YYYY could be a reference year rather than headcount.
            if not year and re.fullmatch(r"(?:19|20)\d{2}", s.strip()):
                return None, None, "Only a year-like number appears in headcount field"
            return int(count), year, None
        except ValueError:
            return None, year, "Unable to parse employee count"
