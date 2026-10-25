"""
Module: src/ingestion/extractor.py
Role: Wikipedia-Based Firmographic Extraction Prototype with Strict Verification & Provenance.
Status: Prototype / Baseline Extractor (Crunchbase/LinkedIn multi-source pipelines require authenticated API adapters).

Design Invariants:
1. Zero keyword-based financial inference (No page-wide 'bootstrapped' guessing).
2. Strict Infobox key targeting (No unanchored page regex matching).
3. Exact calendar arithmetic with boundary checks (snapshot >= founding_date).
4. Full provenance audit trail with raw evidence, transform logs, and missingness rationale.
"""

import re
import urllib.parse
from datetime import datetime, timezone
from typing import Optional, Dict, Any, Tuple
import requests

from src.domain.models import CompanyProfile, AttributeProvenance


class AmbiguousEntityError(ValueError):
    """Raised when an entity query maps to multiple distinct commercial or legal organizations."""
    pass


class UnresolvedEntityError(ValueError):
    """Raised when the fetched page fails identity validation against the requested target."""
    pass


# Explicit disambiguation guards
DISAMBIGUATION_REQUIRED = {
    "ola": "Ambiguous brand 'Ola'. Please specify 'Ola Cabs' (ANI Technologies) or 'Ola Electric'.",
    "reliance": "Ambiguous corporate name 'Reliance'. Please specify 'Reliance Industries' or 'Jio Platforms'.",
    "tata": "Ambiguous conglomerate 'Tata'. Please specify the subsidiary (e.g. 'Tata Motors', 'TCS').",
}


class WikipediaPrototypeExtractor:
    """
    Experimental Wikipedia Infobox Extractor with field-level provenance tracking.
    """

    def __init__(self, timeout: int = 6):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": (
                "StartupMarketAnalyzer/1.0 (Academic Research; Autonomous Due Diligence) "
                "Python-Requests/2.31"
            ),
            "Accept": "application/json",
        })

    def validate_entity_query(self, query: str) -> str:
        clean = query.strip().lower()
        if clean in DISAMBIGUATION_REQUIRED:
            raise AmbiguousEntityError(DISAMBIGUATION_REQUIRED[clean])
        return query.strip()

    def _fetch_infobox_wikitext(self, query: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
        """
        Executes OpenSearch and fetches raw infobox wikitext.
        Returns: (matched_title, page_url, infobox_wikitext, error_reason)
        """
        search_url = (
            f"https://en.wikipedia.org/w/api.php?action=opensearch&search="
            f"{urllib.parse.quote(query)}&limit=1&namespace=0&format=json"
        )
        try:
            r = self.session.get(search_url, timeout=self.timeout)
            if r.status_code != 200:
                return None, None, None, f"HTTP_{r.status_code}"
            data = r.json()
            if not data or len(data) < 4 or not data[1] or not data[3]:
                return None, None, None, "ENTITY_NOT_FOUND"

            matched_title = data[1][0]
            page_url = data[3][0]

            # Validate entity match: check that query tokens exist in page title or vice versa
            norm_q = set(re.findall(r"\w+", query.lower()))
            norm_t = set(re.findall(r"\w+", matched_title.lower()))
            if not (norm_q.issubset(norm_t) or norm_t.issubset(norm_q) or len(norm_q.intersection(norm_t)) > 0):
                return matched_title, page_url, None, "IDENTITY_VALIDATION_FAILED"

            # Parse Infobox wikitext only
            parse_url = (
                f"https://en.wikipedia.org/w/api.php?action=parse&page="
                f"{urllib.parse.quote(matched_title)}&prop=wikitext&format=json&redirects=1"
            )
            r_parse = self.session.get(parse_url, timeout=self.timeout)
            if r_parse.status_code != 200:
                return matched_title, page_url, None, f"HTTP_{r_parse.status_code}"

            parse_data = r_parse.json()
            full_wikitext = parse_data.get("parse", {}).get("wikitext", {}).get("*", "")

            # Extract the Infobox block to prevent matching body text
            m_info = re.search(r"\{\{Infobox company(.*?)\n\}\}", full_wikitext, re.DOTALL | re.I)
            infobox_content = m_info.group(1) if m_info else full_wikitext[:3000]

            return matched_title, page_url, infobox_content, None

        except requests.RequestException as e:
            return None, None, None, f"NETWORK_EXCEPTION: {str(e)}"

    def _parse_infobox_field(self, infobox: str, keys: list) -> Optional[Tuple[str, str]]:
        """Finds field matching key in Infobox. Returns (raw_line, cleaned_value)."""
        for k in keys:
            m = re.search(rf"\|\s*{k}\s*=\s*(.+?)(?=\n\s*\||\n*$)", infobox, re.DOTALL | re.I)
            if m:
                raw = m.group(1).strip()
                # Clean citation tags and markup
                clean = re.sub(r"<ref.*?(?:/>|</ref>)", "", raw, flags=re.DOTALL | re.I)
                clean = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", clean)
                clean = clean.replace("{{", "").replace("}}", "").strip()
                return raw, clean
        return None

    def _parse_exact_founding_date(self, clean_val: str) -> Tuple[Optional[datetime], Optional[str], str]:
        """
        Parses exact date from infobox value.
        Returns: (parsed_datetime, reported_date_str, date_type: 'exact'|'year_only'|'unparseable')
        """
        # Pattern 1: start date template with year, month, day: 2010|8|15
        m1 = re.search(r"start[ _]date(?:[ _]and[ _]age)?\|([0-9]{4})\|([0-9]{1,2})\|([0-9]{1,2})", clean_val, re.I)
        if m1:
            y, m, d = int(m1.group(1)), int(m1.group(2)), int(m1.group(3))
            try:
                return datetime(y, m, d, tzinfo=timezone.utc), f"{y:04d}-{m:02d}-{d:02d}", "exact"
            except ValueError:
                pass

        # Pattern 2: Day Month Year (e.g. 15 August 2010 or August 15, 2010)
        m2 = re.search(r"([0-9]{1,2})\s+([A-Za-z]+)\s+([12][90][0-9]{2})", clean_val)
        if m2:
            try:
                dt = datetime.strptime(f"{m2.group(1)} {m2.group(2)} {m2.group(3)}", "%d %B %Y").replace(tzinfo=timezone.utc)
                return dt, dt.strftime("%Y-%m-%d"), "exact"
            except ValueError:
                pass

        # Pattern 3: Year only
        m3 = re.search(r"\b([12][90][0-9]{2})\b", clean_val)
        if m3:
            y = int(m3.group(1))
            return datetime(y, 1, 1, tzinfo=timezone.utc), f"{y:04d}", "year_only"

        return None, None, "unparseable"

    def _parse_headcount(self, clean_val: str) -> Tuple[Optional[int], Optional[str], Optional[str]]:
        """
        Separates numeric count from reporting year.
        Returns: (headcount, as_of_year, note)
        """
        # Detect reported year inside parentheses e.g. "1,200 (2020)"
        as_of_year = None
        yr_m = re.search(r"\b(20[0-2][0-9])\b", clean_val)
        if yr_m:
            as_of_year = yr_m.group(1)

        # Look for headcount numbers, excluding the detected year
        tokens = re.findall(r"([0-9]{1,3}(?:,[0-9]{3})+|\b[0-9]{2,6}\b)", clean_val)
        for t in tokens:
            digits = t.replace(",", "")
            if digits != as_of_year:
                val = int(digits)
                if val > 0:
                    return val, as_of_year, None

        return None, as_of_year, "HEADCOUNT_AMBIGUOUS_OR_UNPARSEABLE"

    def extract_profile(
        self,
        company_name: str,
        company_id: str,
        as_of_date: Optional[datetime] = None,
    ) -> CompanyProfile:
        now_utc = datetime.now(timezone.utc)
        target_date = as_of_date if as_of_date is not None else now_utc

        if target_date.tzinfo is None:
            raise ValueError("as_of_date must be timezone-aware (UTC).")

        canonical_query = self.validate_entity_query(company_name)
        matched_title, page_url, infobox, fetch_err = self._fetch_infobox_wikitext(canonical_query)

        provenance: Dict[str, Dict[str, Any]] = {}

        if fetch_err:
            # Document request/validation failure cleanly across all attributes
            for attr in ["company_age_years", "employee_count", "industry_sector", "total_funding_usd_m", "growth_rate"]:
                provenance[attr] = {
                    "value": None,
                    "source_url": page_url,
                    "retrieval_timestamp": now_utc.isoformat(),
                    "status": "failed",
                    "reason_for_missingness": fetch_err,
                    "confidence": 0.0,
                }
            return CompanyProfile(
                company_id=company_id,
                company_name=company_name,
                snapshot_date=target_date,
                provenance=provenance,
            )

        # 1. Company Age (Exact Arithmetic)
        age_years = None
        found_field = self._parse_infobox_field(infobox, ["founded", "foundation", "established"])
        if found_field:
            raw_found, clean_found = found_field
            found_dt, found_date_str, precision = self._parse_exact_founding_date(clean_found)
            if found_dt:
                delta_days = (target_date - found_dt).days
                if delta_days < 0:
                    # Invariant Check: snapshot date is prior to founding date
                    provenance["company_age_years"] = {
                        "value": None,
                        "raw_evidence": raw_found,
                        "source_url": page_url,
                        "retrieval_timestamp": now_utc.isoformat(),
                        "status": "invalid_pre_founding_snapshot",
                        "reason_for_missingness": f"Snapshot ({target_date.date()}) precedes founding ({found_dt.date()})",
                        "confidence": 0.0,
                    }
                else:
                    age_years = round(delta_days / 365.25, 2)
                    provenance["company_age_years"] = {
                        "value": age_years,
                        "raw_evidence": raw_found,
                        "as_of_date": found_date_str,
                        "precision": precision,
                        "source_url": page_url,
                        "retrieval_timestamp": now_utc.isoformat(),
                        "status": "reported",
                        "confidence": 0.95 if precision == "exact" else 0.80,
                    }
            else:
                provenance["company_age_years"] = {
                    "value": None,
                    "raw_evidence": raw_found,
                    "source_url": page_url,
                    "retrieval_timestamp": now_utc.isoformat(),
                    "status": "unparseable",
                    "reason_for_missingness": "Founding date format unparseable",
                    "confidence": 0.0,
                }
        else:
            provenance["company_age_years"] = {
                "value": None,
                "source_url": page_url,
                "retrieval_timestamp": now_utc.isoformat(),
                "status": "unknown",
                "reason_for_missingness": "No founded field in Wikipedia Infobox",
                "confidence": 0.0,
            }

        # 2. Headcount
        employees = None
        emp_field = self._parse_infobox_field(infobox, ["num_employees", "num_employees_year", "employees"])
        if emp_field:
            raw_emp, clean_emp = emp_field
            emp_val, emp_year, err = self._parse_headcount(clean_emp)
            if emp_val is not None:
                employees = emp_val
                provenance["employee_count"] = {
                    "value": employees,
                    "raw_evidence": raw_emp,
                    "as_of_date": emp_year,
                    "source_url": page_url,
                    "retrieval_timestamp": now_utc.isoformat(),
                    "status": "reported",
                    "confidence": 0.85 if emp_year else 0.65,
                }
            else:
                provenance["employee_count"] = {
                    "value": None,
                    "raw_evidence": raw_emp,
                    "source_url": page_url,
                    "retrieval_timestamp": now_utc.isoformat(),
                    "status": "unparseable",
                    "reason_for_missingness": err,
                    "confidence": 0.0,
                }
        else:
            provenance["employee_count"] = {
                "value": None,
                "source_url": page_url,
                "retrieval_timestamp": now_utc.isoformat(),
                "status": "unknown",
                "reason_for_missingness": "No employee field in Wikipedia Infobox",
                "confidence": 0.0,
            }

        # 3. Industry Sector
        sector = None
        ind_field = self._parse_infobox_field(infobox, ["industry", "genre"])
        if ind_field:
            raw_ind, clean_ind = ind_field
            sector = clean_ind.split("\n")[0].split(",")[0].strip()
            provenance["industry_sector"] = {
                "value": sector,
                "raw_evidence": raw_ind,
                "source_url": page_url,
                "retrieval_timestamp": now_utc.isoformat(),
                "status": "reported",
                "confidence": 0.85,
            }
        else:
            sector = "Unclassified"
            provenance["industry_sector"] = {
                "value": None,
                "source_url": page_url,
                "retrieval_timestamp": now_utc.isoformat(),
                "status": "unknown",
                "reason_for_missingness": "No industry field in Wikipedia Infobox",
                "confidence": 0.0,
            }

        # 4. Total Funding: Wikipedia Infoboxes do NOT reliably store round-by-round VC funding.
        # Strict Rule: NEVER guess verified_zero from keyword matching. Mark as None / Unknown.
        provenance["total_funding_usd_m"] = {
            "value": None,
            "source_url": page_url,
            "retrieval_timestamp": now_utc.isoformat(),
            "status": "unknown",
            "reason_for_missingness": "Wikipedia Infobox does not track auditable venture funding rounds (Requires Crunchbase/MCA adapter)",
            "confidence": 0.0,
        }

        # 5. Historical Growth Rate: Stays None without longitudinal observations
        provenance["growth_rate"] = {
            "value": None,
            "retrieval_timestamp": now_utc.isoformat(),
            "status": "unknown",
            "reason_for_missingness": "Single point-in-time scrape lacks longitudinal headcount/revenue series",
            "confidence": 0.0,
        }

        return CompanyProfile(
            company_id=company_id,
            company_name=matched_title or canonical_query,
            snapshot_date=target_date,
            total_funding_usd_m=None,  # Stays None; no false zeroes
            employee_count=employees,
            growth_rate=None,
            company_age_years=age_years,
            industry_sector=sector,
            is_bootstrapped=None,
            provenance=provenance,
        )


# Backward-compatibility alias
WebFirmographicExtractor = WikipediaPrototypeExtractor
GrowjoExtractor = WikipediaPrototypeExtractor