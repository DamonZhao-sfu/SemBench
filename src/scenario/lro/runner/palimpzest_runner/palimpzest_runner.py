"""Palimpzest runner for the LRO match queries (113, 121, 202, 305)."""

from pathlib import Path
import sys

import palimpzest as pz
from palimpzest.core.elements.records import DataRecordCollection

sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from runner.generic_palimpzest_runner.generic_palimpzest_runner import (  # noqa: E402
    GenericPalimpzestRunner,
)
from scenario.lro.data_utils import load_query_inputs  # noqa: E402


class PalimpzestRunner(GenericPalimpzestRunner):
    """Palimpzest runner for the LRO scenario."""

    def __init__(
        self,
        use_case: str,
        scale_factor: int,
        model_name: str = "gemini-2.5-flash",
        concurrent_llm_worker: int = 20,
        skip_setup: bool = False,
    ):
        super().__init__(
            use_case,
            scale_factor,
            model_name,
            concurrent_llm_worker,
            skip_setup,
        )

    def _run_match_join(self, query_id: int) -> DataRecordCollection:
        left_df, right_df, meta = load_query_inputs(query_id)

        # Strip placeholders from the prompt because Palimpzest's sem_join
        # condition is plain natural language and column dependencies are
        # passed via depends_on.
        condition = meta["prompt"]
        for col in list(left_df.columns) + list(right_df.columns):
            condition = condition.replace("{" + col + "}", f"<{col}>")

        # Cast Int64 (nullable) ids to plain Python ints since Palimpzest's
        # MemoryDataset doesn't always handle pandas extension types well.
        for col in left_df.columns:
            if str(left_df[col].dtype) == "Int64":
                left_df[col] = left_df[col].astype("int64")
        for col in right_df.columns:
            if str(right_df[col].dtype) == "Int64":
                right_df[col] = right_df[col].astype("int64")

        left = pz.MemoryDataset(id=f"lro_q{query_id}_left", vals=left_df)
        right = pz.MemoryDataset(id=f"lro_q{query_id}_right", vals=right_df)

        joined = left.sem_join(
            right,
            condition=condition,
            depends_on=list(left_df.columns) + list(right_df.columns),
        )
        joined = joined.project([meta["left_key"], meta["right_key"]])
        return joined.run(self.palimpzest_config())

    def _execute_q113(self):
        return self._run_match_join(113)

    def _execute_q121(self):
        return self._run_match_join(121)

    def _execute_q202(self):
        return self._run_match_join(202)

    def _execute_q305(self):
        return self._run_match_join(305)
