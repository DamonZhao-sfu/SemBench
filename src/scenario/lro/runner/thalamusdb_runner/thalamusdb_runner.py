"""ThalamusDB system runner implementation for the LRO scenario."""

import sys
from pathlib import Path

from overrides import override

sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from runner.generic_thalamusdb_runner.generic_thalamusdb_runner import (
    GenericThalamusDBRunner,
)


class ThalamusDBRunner(GenericThalamusDBRunner):
    def __init__(
        self,
        use_case: str,
        scale_factor: int,
        model_name: str = "gemini-2.5-flash",
        concurrent_llm_worker: int = 20,
        skip_setup: bool = False,
    ):
        # ThalamusDB stores its DuckDB file alongside the source CSVs.  We
        # construct the path before calling the base initialiser because the
        # base class wants to know it.
        from scenario.lro.lro_scenario import get_lrobench_databases_dir

        db_path = str(
            Path(get_lrobench_databases_dir()) / "lro_thalamusdb.duckdb"
        )

        super().__init__(
            use_case,
            scale_factor,
            model_name,
            concurrent_llm_worker,
            db_path,
            skip_setup=skip_setup,
        )

    @override
    def _discover_query_impl(self, query_id) -> callable:
        sql_text = self.scenario_handler.get_query_text(
            query_id, self.get_system_name()
        )
        return lambda: self.execute_thalamusdb_query(sql_text)
