"""LRO evaluator. Ground truth pairs are hardcoded (from lro.py)."""

from pathlib import Path
import sys
from typing import List, Tuple

import pandas as pd

sys.path.append(str(Path(__file__).parent.parent.parent))
from evaluator.generic_evaluator import (  # noqa: E402
    GenericEvaluator,
    QueryMetricRetrieval,
)


# Ground truth pairs: list of (left_id, right_id).
GROUND_TRUTH = {
    113: [
        ("Mark Sedwill ", "Mark Bryson-Richardson"),
        ("Richard Clarke", "Richard Montgomery"),
    ],
    121: [
        ("Nhs Supply Chain", "NHS SUPPLY CHAIN"),
        ("Novartis Pharmaceuticals Uk Ltd", "NOVARTIS PHARMACEUTICALS UK LTD"),
        ("Roche Diagnostics Limited", "ROCHE PRODUCTS LTD"),
        ("Csc", "CSC COMPUTER SCIENCES LTD"),
        ("Philips Healthcare", "PHILIPS HEALTHCARE"),
        ("Nhs Blood And Transplant", "NHS BLOOD & TRANSPLANT"),
        ("NHS Litigation Authority", "NHS LITIGATION AUTHORITY"),
    ],
    202: [
        (2, 844), (72, 1020), (90, 744), (111, 564),
        (198, 1022), (275, 272), (333, 134), (418, 590),
    ],
    305: [
        ("CDSCode", "CDSCode"),
        ("County Code", "County"),
        ("District Code", "NCESDist"),
        ("District Name", "District"),
        ("School Name", "School"),
        ("District Type ", "DOCType"),
        ("School Type", "SOCType"),
        ("Educational Option Type", "EdOpsName"),
        ("Charter School (Y/N)", "Charter"),
        ("Charter School Number", "CharterNum"),
        ("Charter Funding Type", "FundingType"),
    ],
}


class LROEvaluator(GenericEvaluator):
    """Evaluator for the LRO match queries."""

    def __init__(self, use_case: str, scale_factor: int = None) -> None:
        super().__init__(use_case, scale_factor)

    def _load_domain_data(self) -> None:
        # Nothing to load: ground truth is hardcoded.
        pass

    def _get_ground_truth(self, query_id: int) -> pd.DataFrame:
        if query_id not in GROUND_TRUTH:
            raise ValueError(f"Unknown LRO query id: {query_id}")
        rows: List[Tuple] = GROUND_TRUTH[query_id]
        df = pd.DataFrame(rows, columns=["left", "right"])

        gt_path = self._results_path / "ground_truth" / f"Q{query_id}.csv"
        gt_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(gt_path, index=False)
        return df

    def _evaluate_single_query(
        self,
        query_id: int,
        system_results: pd.DataFrame,
        ground_truth: pd.DataFrame,
    ) -> QueryMetricRetrieval:
        return self._evaluate_pair_match(system_results, ground_truth)

    @staticmethod
    def _normalize(value):
        if pd.isna(value):
            return None
        if isinstance(value, str):
            return value.strip().lower()
        try:
            return int(value)
        except (TypeError, ValueError):
            return value

    def _evaluate_pair_match(
        self,
        system_results: pd.DataFrame,
        ground_truth: pd.DataFrame,
    ) -> QueryMetricRetrieval:
        """Compare (left, right) pairs as a set, normalised case-insensitively
        for strings and to ints for numeric ids."""
        if ground_truth.empty:
            return QueryMetricRetrieval(
                precision=1.0 if system_results.empty else 0.0
            )
        if system_results.empty:
            return QueryMetricRetrieval()

        if (
            len(system_results.columns) < 2
            or len(ground_truth.columns) < 2
        ):
            return QueryMetricRetrieval()

        sys_cols = list(system_results.columns[:2])
        gt_cols = list(ground_truth.columns[:2])

        sys_pairs = {
            (self._normalize(r[sys_cols[0]]), self._normalize(r[sys_cols[1]]))
            for _, r in system_results.iterrows()
            if pd.notna(r[sys_cols[0]]) and pd.notna(r[sys_cols[1]])
        }
        gt_pairs = {
            (self._normalize(r[gt_cols[0]]), self._normalize(r[gt_cols[1]]))
            for _, r in ground_truth.iterrows()
            if pd.notna(r[gt_cols[0]]) and pd.notna(r[gt_cols[1]])
        }

        correct = sys_pairs & gt_pairs
        precision = len(correct) / len(sys_pairs) if sys_pairs else 0.0
        recall = len(correct) / len(gt_pairs) if gt_pairs else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        return QueryMetricRetrieval(precision, recall, f1)
