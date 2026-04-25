"""LRO evaluator: compares system results against the ground-truth match pairs."""

import sys
from pathlib import Path
from typing import Union

import pandas as pd

# Add src/ to the path so we can import evaluator/ and scenario/.
sys.path.append(str(Path(__file__).parent.parent.parent))
from evaluator.generic_evaluator import (
    GenericEvaluator,
    QueryMetricRetrieval,
)

sys.path.append(str(Path(__file__).parent.parent))
from lro.lro_scenario import LROScenario, QUERY_REGISTRY


class LROEvaluator(GenericEvaluator):
    """Evaluator for the LRO matching benchmark."""

    def __init__(self, use_case: str, scale_factor: int = None) -> None:
        super().__init__(use_case, scale_factor)
        self.scenario_handler = LROScenario(scale_factor=scale_factor)

    def _load_domain_data(self) -> None:
        # No additional domain data needed beyond what's embedded in the
        # scenario handler / query registry.
        pass

    def _get_ground_truth(self, query_id: int) -> pd.DataFrame:
        return self.scenario_handler.get_ground_truth(query_id)

    def _evaluate_single_query(
        self,
        query_id: int,
        system_results: pd.DataFrame,
        ground_truth: pd.DataFrame,
    ) -> Union[QueryMetricRetrieval]:
        meta = QUERY_REGISTRY[int(query_id)]
        left_alias = meta["left_alias"]
        right_alias = meta["right_alias"]

        gt_pairs = self._normalise_pairs(
            ground_truth, left_alias, right_alias
        )
        sys_pairs = self._normalise_pairs(
            system_results, left_alias, right_alias
        )

        if not gt_pairs and not sys_pairs:
            return QueryMetricRetrieval(1.0, 1.0, 1.0)
        if not sys_pairs:
            return QueryMetricRetrieval(0.0, 0.0, 0.0)
        if not gt_pairs:
            return QueryMetricRetrieval(0.0, 0.0, 0.0)

        correct = sys_pairs & gt_pairs
        precision = len(correct) / len(sys_pairs)
        recall = len(correct) / len(gt_pairs)
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        return QueryMetricRetrieval(precision, recall, f1)

    @staticmethod
    def _normalise_pairs(df: pd.DataFrame, left_col: str, right_col: str):
        if df is None or df.empty:
            return set()
        if left_col not in df.columns or right_col not in df.columns:
            # Fall back to the first two columns: systems may rename them.
            if df.shape[1] < 2:
                return set()
            l, r = df.columns[0], df.columns[1]
        else:
            l, r = left_col, right_col

        out = set()
        for _, row in df[[l, r]].iterrows():
            lv, rv = row[l], row[r]
            if pd.isna(lv) or pd.isna(rv):
                continue
            out.add((str(lv).strip(), str(rv).strip()))
        return out
