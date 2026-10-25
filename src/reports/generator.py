"""
Module: src/reports/generator.py
Role: Formats complete due-diligence audit memos combining predictions, XAI, and fairness.
"""

from datetime import datetime, timezone
from typing import Dict, Any
import json


class DueDiligenceReportGenerator:
    @staticmethod
    def generate_markdown_memo(
        company_profile: Dict[str, Any],
        prediction_results: Dict[str, Any],
        shap_explanation: Dict[str, Any],
        fairness_audit: Dict[str, Any],
    ) -> str:
        """Assembles an executive-ready Markdown due-diligence memo."""
        as_of = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        probs = prediction_results.get("probabilities", {})
        drivers = shap_explanation.get("top_positive_drivers", {})
        drags = shap_explanation.get("top_negative_drags", {})

        memo = f"""# Startup Market Analyzer (SMA) — Due Diligence Memo
**Target Entity:** {company_profile.get('company_name', 'N/A')} (`{company_profile.get('company_id', 'n/a')}`)  
**Generated At:** {as_of}  
**Sector / Geography:** {company_profile.get('industry_sector', 'N/A')} | {company_profile.get('headquarters', 'Tier-1 Hub')}

---

## 1. Executive Viability Summary
* **Predicted Trajectory:** **{prediction_results.get('predicted_label', 'Unknown')}**
* **Viability Breakdown:**
  * High Growth Probability: `{probs.get('high_growth', 0.0) * 100:.1f}%`
  * Stable / Moderate Viability: `{probs.get('stable', 0.0) * 100:.1f}%`
  * Distressed / Capital Risk: `{probs.get('distressed', 0.0) * 100:.1f}%`

---

## 2. Core Firmographics & Point-in-Time Metrics
* **Total Disclosed Funding:** ${company_profile.get('total_funding_usd_m', 0.0):,.2f}M USD
* **Employee Base:** {company_profile.get('employee_count', 0):,} headcount
* **Operating Age:** {company_profile.get('company_age_years', 0)} years
* **Funding Intensity:** ${prediction_results.get('fused_features', {}).get('funding_intensity', 0.0):,.2f}M / yr
* **Ingested & Deduplicated Media Articles:** {int(prediction_results.get('fused_features', {}).get('article_count', 0))} records

---

## 3. Explainable AI (SHAP Feature Attributions)
Attributions derived via game-theoretic tree breakdown (why the model reached this verdict):

**Key Positive Drivers (Strengths):**
"""
        if drivers:
            for feat, val in drivers.items():
                memo += f"* **{feat}**: `+{val:.4f}` impact\n"
        else:
            memo += "* None identified\n"

        memo += "\n**Key Vulnerabilities / Drags (Risks):**\n"
        if drags:
            for feat, val in drags.items():
                memo += f"* **{feat}**: `{val:.4f}` impact\n"
        else:
            memo += "* None identified\n"

        memo += f"""
---

## 4. Ethical AI & Fairness Compliance (AIF360)
* **Audited Subgroup:** {fairness_audit.get('protected_attribute', 'Geographic Hub')}
* **Demographic Parity Gap (DPD):** `{fairness_audit.get('demographic_parity_gap', 0.0):.4f}` (Standard: < 0.05)
* **Equal Opportunity Difference (EOD):** `{fairness_audit.get('equal_opportunity_difference', 0.0):.4f}` (Standard: < 0.04)
* **Status:** {'APPROVED — Bias thresholds within acceptable parameters' if fairness_audit.get('fairness_threshold_met_dpd_05') else 'WARNING — Demographic variance exceeds benchmark'}

---
*Generated autonomously by Startup Market Analyzer Engine.*
"""
        return memo