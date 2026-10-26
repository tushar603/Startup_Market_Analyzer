"""
Module: src/ingestion/extractor.py
Role: Wikipedia-based firmographic extraction with attribute-level provenance.
Status: Prototype / baseline extractor; not a historical funding-data source.

Design invariants:
- Never infer funding or bootstrapping from page-wide keywords.
- Parse named Wikipedia infobox fields, not arbitrary article text.
- Preserve missing values and source evidence.
- Preserve the existing extract_profile() and legacy extractor class interfaces.
"""

import re
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

import requests

from src.domain.models import CompanyProfile
from src.ingestion.wikipedia_helpers import WikipediaInfoboxClient


class AmbiguousEntityError(ValueError):
    """A company name needs a more precise legal entity or brand."""


class UnresolvedEntityError(ValueError):
    """A candidate Wikipedia article could not be verified as the target."""


DISAMBIGUATION_REQUIRED = {
    "ola": "Ambiguous brand 'Ola'. Please specify 'Ola Cabs' (ANI Technologies) or 'Ola Electric'.",
    "reliance": "Ambiguous corporate name 'Reliance'. Please specify 'Reliance Industries' or 'Jio Platforms'.",
    "tata": "Ambiguous conglomerate 'Tata'. Please specify the subsidiary (e.g. 'Tata Motors', 'TCS').",
}


class WikipediaPrototypeExtractor:
    """Extracts available company infobox fields without inventing missing facts."""

    def __init__(self, timeout: int = 6):
        # Retained for constructor compatibility. The helper manages request timeouts.
        self.timeout = timeout
        self.session = requests.Session()
        self._wikipedia = WikipediaInfoboxClient(self.session)

    def validate_entity_query(self, query: str) -> str:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Company name must be a nonempty string.")
        clean = query.strip()
        if clean.casefold() in DISAMBIGUATION_REQUIRED:
            raise AmbiguousEntityError(DISAMBIGUATION_REQUIRED[clean.casefold()])
        return clean

    # Compatibility for existing service code using the older validator name.
    def validate_and_disambiguate(self, company_name: str) -> str:
        return self.validate_entity_query(company_name)

    # Delegate *all* Wikipedia retrieval/parsing to the tested helper.
    # Do not define these methods again later in this class.
    def _fetch_infobox_wikitext(
        self, query: str
    ) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
        return self._wikipedia._fetch_infobox_wikitext(query)

    def _parse_infobox_field(self, infobox: str, field_names: list):
        return self._wikipedia._parse_infobox_field(infobox, field_names)

    def _parse_exact_founding_date(self, clean_found: str):
        """Return (date, source_date, precision); recognize month/year templates too."""
        # Some infoboxes use {{start date and age |August 2010}}, not YYYY|MM|DD.
        month_match = re.search(
            r"\{\{\s*start[ _]date(?:[ _]and[ _]age)?\s*\|\s*"
            r"([A-Za-z]+)\s+((?:18|19|20)\d{2})(?=\s*(?:\||\}\}))",
            clean_found or "",
            flags=re.I,
        )
        if month_match:
            for fmt in ("%B %Y", "%b %Y"):
                try:
                    month_start = datetime.strptime(
                        f"{month_match.group(1)} {month_match.group(2)}", fmt
                    ).replace(tzinfo=timezone.utc)
                    return month_start, month_start.strftime("%Y-%m"), "month"
                except ValueError:
                    continue
        return self._wikipedia._parse_exact_founding_date(clean_found)

    def _parse_headcount(self, clean_emp: str):
        count, report_year, error = self._wikipedia._parse_headcount(clean_emp)
        # The helper detects '(2026)'; Wikipedia also uses '(April 2026)'.
        if report_year is None:
            year_match = re.search(
                r"\(\s*(?:as of\s+)?[A-Za-z]+\s+((?:19|20)\d{2})\s*\)",
                clean_emp or "",
                flags=re.I,
            )
            if year_match:
                report_year = year_match.group(1)
        return count, report_year, error

    def extract_profile(
        self,
        company_name: str,
        company_id: str,
        as_of_date: Optional[datetime] = None,
    ) -> CompanyProfile:
        now_utc = datetime.now(timezone.utc)
        target_date = as_of_date if as_of_date is not None else now_utc

        if target_date.tzinfo is None or target_date.utcoffset() is None:
            raise ValueError("as_of_date must be timezone-aware.")
        target_date = target_date.astimezone(timezone.utc)

        canonical_query = self.validate_entity_query(company_name)
        matched_title, page_url, infobox, fetch_err = self._fetch_infobox_wikitext(
            canonical_query
        )
        provenance: Dict[str, Dict[str, Any]] = {}
        retrieved_at = now_utc.isoformat()

        if fetch_err or not infobox:
            reason = fetch_err or "Wikipedia returned no usable company infobox"
            for attr in (
                "company_age_years",
                "employee_count",
                "industry_sector",
                "total_funding_usd_m",
                "growth_rate",
            ):
                provenance[attr] = {
                    "value": None,
                    "source_url": page_url,
                    "retrieval_timestamp": retrieved_at,
                    "status": "failed",
                    "reason_for_missingness": reason,
                    "confidence": 0.0,
                }
            return CompanyProfile(
                company_id=company_id,
                company_name=company_name,
                snapshot_date=target_date,
                provenance=provenance,
            )

        # 1. Company age, using the infobox founding field only.
        age_years = None
        found_field = self._parse_infobox_field(
            infobox, ["founded", "foundation", "established"]
        )
        if found_field:
            raw_found, clean_found = found_field
            found_dt, found_date_str, precision = self._parse_exact_founding_date(
                clean_found
            )
            if found_dt:
                delta_days = (target_date - found_dt).days
                if delta_days < 0:
                    provenance["company_age_years"] = {
                        "value": None,
                        "raw_evidence": raw_found,
                        "source_url": page_url,
                        "retrieval_timestamp": retrieved_at,
                        "status": "invalid_pre_founding_snapshot",
                        "reason_for_missingness": (
                            f"Snapshot ({target_date.date()}) precedes "
                            f"reported founding date ({found_dt.date()})"
                        ),
                        "confidence": 0.0,
                    }
                else:
                    # For 'month' or 'year' precision, the first day is a proxy.
                    age_years = round(delta_days / 365.25, 2)
                    provenance["company_age_years"] = {
                        "value": age_years,
                        "raw_evidence": raw_found,
                        "as_of_date": found_date_str,
                        "precision": precision,
                        "source_url": page_url,
                        "retrieval_timestamp": retrieved_at,
                        "status": "reported" if precision == "exact" else "approximate",
                        "confidence": 0.95 if precision == "exact" else 0.80,
                    }
            else:
                provenance["company_age_years"] = {
                    "value": None,
                    "raw_evidence": raw_found,
                    "source_url": page_url,
                    "retrieval_timestamp": retrieved_at,
                    "status": "unparseable",
                    "reason_for_missingness": "Founding date format unparseable",
                    "confidence": 0.0,
                }
        else:
            provenance["company_age_years"] = {
                "value": None,
                "source_url": page_url,
                "retrieval_timestamp": retrieved_at,
                "status": "unknown",
                "reason_for_missingness": "No founded field in Wikipedia infobox",
                "confidence": 0.0,
            }

        # 2. Employee count; don't use a headcount reported after the cutoff year.
        employees = None
        emp_field = self._parse_infobox_field(
            infobox, ["num_employees", "employees"]
        )
        if emp_field:
            raw_emp, clean_emp = emp_field
            emp_val, emp_year, err = self._parse_headcount(clean_emp)
            if emp_val is not None and emp_year and int(emp_year) > target_date.year:
                err = f"Reported headcount ({emp_year}) is after snapshot ({target_date.year})"
                emp_val = None
                emp_status = "future_dated"
            else:
                emp_status = "unparseable"

            if emp_val is not None:
                employees = emp_val
                provenance["employee_count"] = {
                    "value": employees,
                    "raw_evidence": raw_emp,
                    "as_of_date": emp_year,
                    "source_url": page_url,
                    "retrieval_timestamp": retrieved_at,
                    "status": "reported",
                    "confidence": 0.85 if emp_year else 0.65,
                }
            else:
                provenance["employee_count"] = {
                    "value": None,
                    "raw_evidence": raw_emp,
                    "source_url": page_url,
                    "retrieval_timestamp": retrieved_at,
                    "status": emp_status,
                    "reason_for_missingness": err,
                    "confidence": 0.0,
                }
        else:
            provenance["employee_count"] = {
                "value": None,
                "source_url": page_url,
                "retrieval_timestamp": retrieved_at,
                "status": "unknown",
                "reason_for_missingness": "No employee count in Wikipedia infobox",
                "confidence": 0.0,
            }

        # 3. Sector. Reject unresolved Wiki markup and placeholder template names.
        sector = "Unclassified"
        ind_field = self._parse_infobox_field(infobox, ["industry", "genre"])
        if ind_field:
            raw_ind, clean_ind = ind_field
            candidate = clean_ind.split("\n", 1)[0].split(",", 1)[0].strip()
            if (
                candidate
                and not any(token in candidate for token in ("{{", "}}", "|"))
                and candidate.casefold() not in {
                    "ubl", "ubil", "hlist", "plainlist", "unbulleted list", "flatlist"
                }
            ):
                sector = candidate
                provenance["industry_sector"] = {
                    "value": sector,
                    "raw_evidence": raw_ind,
                    "source_url": page_url,
                    "retrieval_timestamp": retrieved_at,
                    "status": "reported",
                    "confidence": 0.85,
                }
        if "industry_sector" not in provenance:
            provenance["industry_sector"] = {
                "value": None,
                "source_url": page_url,
                "retrieval_timestamp": retrieved_at,
                "status": "unknown",
                "reason_for_missingness": "No usable industry field in Wikipedia infobox",
                "confidence": 0.0,
            }

        # 4. Wikipedia infoboxes do not reliably document venture funding.
        provenance["total_funding_usd_m"] = {
            "value": None,
            "source_url": page_url,
            "retrieval_timestamp": retrieved_at,
            "status": "unknown",
            "reason_for_missingness": (
                "Wikipedia infobox does not provide auditable VC funding rounds"
            ),
            "confidence": 0.0,
        }

        # 5. A single current infobox does not establish longitudinal growth.
        provenance["growth_rate"] = {
            "value": None,
            "retrieval_timestamp": retrieved_at,
            "status": "unknown",
            "reason_for_missingness": (
                "Single point-in-time extraction has no dated growth series"
            ),
            "confidence": 0.0,
        }

        return CompanyProfile(
            company_id=company_id,
            company_name=matched_title or canonical_query,
            snapshot_date=target_date,
            total_funding_usd_m=None,
            employee_count=employees,
            growth_rate=None,
            company_age_years=age_years,
            industry_sector=sector,
            is_bootstrapped=None,
            provenance=provenance,
        )


# Backward-compatible names used elsewhere in the SMA project.
WebFirmographicExtractor = WikipediaPrototypeExtractor
GrowjoExtractor = WikipediaPrototypeExtractor
