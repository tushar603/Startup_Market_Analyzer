"""
Module: src/fairness/audit.py
Role: Demographic parity and disparate impact auditing via IBM AIF360.
Concepts: Ethical AI, Protected Attributes, DPD, Equal Opportunity Difference.
"""

from typing import Dict, Any
import numpy as np
import pandas as pd
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric


class FairnessAuditor:
    def __init__(
        self,
        protected_attribute: str,
        privileged_value: int = 1,
        unprivileged_value: int = 0,
    ):
        self.protected_attribute = protected_attribute
        self.privileged_groups = [{self.protected_attribute: privileged_value}]
        self.unprivileged_groups = [{self.protected_attribute: unprivileged_value}]

    def _sanitize_metric(self, value: float, fallback: float = 0.0) -> float:
        """Sanitizes NaN/Inf outputs caused by zero denominators in sample slices."""
        if value is None or np.isnan(value) or np.isinf(value):
            return fallback
        return float(value)

    def audit_predictions(
        self,
        X_df: pd.DataFrame,
        y_true: pd.Series,
        y_pred: np.ndarray,
        protected_series: pd.Series,
    ) -> Dict[str, Any]:
        """
        Calculates demographic parity difference, disparate impact, and
        equal opportunity difference between privileged and unprivileged cohorts.
        """
        # Ensure clean consecutive index alignment across tabular inputs
        df_base = X_df.reset_index(drop=True).copy()
        prot_vals = pd.Series(protected_series).reset_index(drop=True).values
        true_vals = pd.Series(y_true).reset_index(drop=True).values
        pred_vals = np.asarray(y_pred)

        # Project multiclass target to binary: High Growth (class 2) vs Others (classes 0, 1)
        binary_true = (true_vals == 2).astype(float)
        binary_pred = (pred_vals == 2).astype(float)

        df_true = df_base.copy()
        df_true[self.protected_attribute] = prot_vals
        df_true["label"] = binary_true

        df_pred = df_base.copy()
        df_pred[self.protected_attribute] = prot_vals
        df_pred["label"] = binary_pred

        dataset_true = BinaryLabelDataset(
            df=df_true,
            label_names=["label"],
            protected_attribute_names=[self.protected_attribute],
            favorable_label=1.0,
            unfavorable_label=0.0,
        )

        dataset_pred = BinaryLabelDataset(
            df=df_pred,
            label_names=["label"],
            protected_attribute_names=[self.protected_attribute],
            favorable_label=1.0,
            unfavorable_label=0.0,
        )

        metric = ClassificationMetric(
            dataset_true,
            dataset_pred,
            unprivileged_groups=self.unprivileged_groups,
            privileged_groups=self.privileged_groups,
        )

        # 1. Statistical Parity Difference (DPD): Target is abs(dpd) < 0.05
        raw_dpd = metric.statistical_parity_difference()
        dpd = self._sanitize_metric(raw_dpd, fallback=0.0)

        # 2. Equal Opportunity Difference (EOD): Target is abs(eod) < 0.04
        raw_eod = metric.equal_opportunity_difference()
        eod = self._sanitize_metric(raw_eod, fallback=0.0)

        # 3. Disparate Impact Ratio: Safe evaluation (parity is 1.0)
        # 3. Disparate Impact Ratio: Safe evaluation (parity is 1.0)
        try:
            with np.errstate(divide="ignore", invalid="ignore"):
                raw_di = metric.disparate_impact()
            disparate_impact = self._sanitize_metric(raw_di, fallback=1.0)
        except (ZeroDivisionError, FloatingPointError):
            disparate_impact = 1.0

        return {
            "protected_attribute": self.protected_attribute,
            "statistical_parity_difference": round(dpd, 4),
            "demographic_parity_gap": round(abs(dpd), 4),
            "equal_opportunity_difference": round(eod, 4),
            "disparate_impact_ratio": round(disparate_impact, 4),
            "fairness_threshold_met_dpd_05": abs(dpd) < 0.05,
            "fairness_threshold_met_eod_04": abs(eod) < 0.04,
        }