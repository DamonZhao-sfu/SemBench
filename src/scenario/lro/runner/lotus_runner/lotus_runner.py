"""LOTUS runner for the LRO match queries (113, 121, 202, 305)."""

from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from runner.generic_lotus_runner.generic_lotus_runner import GenericLotusRunner  # noqa: E402
from scenario.lro.data_utils import load_query_inputs  # noqa: E402


class LotusRunner(GenericLotusRunner):
    """LOTUS runner for the LRO scenario."""

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

    def _run_match_join(self, query_id: int) -> pd.DataFrame:
        left, right, meta = load_query_inputs(query_id)
        prompt = meta["prompt"]

        # Convert {left_xxx} -> {left_xxx:left}, {right_xxx} -> {right_xxx:right}
        # so LOTUS knows which side each column comes from.
        join_instruction = prompt
        for col in left.columns:
            join_instruction = join_instruction.replace(
                "{" + col + "}", "{" + col + ":left}"
            )
        for col in right.columns:
            join_instruction = join_instruction.replace(
                "{" + col + "}", "{" + col + ":right}"
            )

        joined = left.sem_join(right, join_instruction=join_instruction)
        if len(joined) == 0:
            return pd.DataFrame(columns=["left", "right"])

        left_key = meta["left_key"]
        right_key = meta["right_key"]

        result = joined[[f"{left_key}:left", f"{right_key}:right"]].copy()
        result.columns = ["left", "right"]
        return result

    def _execute_q113(self) -> pd.DataFrame:
        return self._run_match_join(113)

    def _execute_q121(self) -> pd.DataFrame:
        return self._run_match_join(121)

    def _execute_q202(self) -> pd.DataFrame:
        return self._run_match_join(202)

    def _execute_q305(self) -> pd.DataFrame:
        return self._run_match_join(305)
