"""ThalamusDB runner for the LRO match queries (113, 121, 202, 305).

Each query is implemented as an `NLjoin` between two DuckDB tables, which we
materialise once on first construction from the raw CSVs.
"""

from pathlib import Path
import sys
from typing import Any, Dict

import duckdb

sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from runner.generic_thalamusdb_runner.generic_thalamusdb_runner import (  # noqa: E402
    GenericThalamusDBRunner,
)
from scenario.lro.data_utils import load_query_inputs  # noqa: E402


_TABLE_NAMES = {
    113: ("officials_left", "officials_right"),
    121: ("suppliers_left", "suppliers_right"),
    202: ("yelp_left", "zomato_right"),
    305: ("frpm_left", "schools_right"),
}


class ThalamusDBRunner(GenericThalamusDBRunner):
    """ThalamusDB runner for the LRO scenario."""

    def __init__(
        self,
        use_case: str,
        scale_factor: int,
        model_name: str = "gemini-2.5-flash",
        concurrent_llm_worker: int = 20,
        skip_setup: bool = False,
    ):
        artifacts_dir = (
            Path(__file__).resolve().parents[5]
            / "files"
            / use_case
            / "data"
            / f"sf_{scale_factor}"
        )
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        db_path = artifacts_dir / "lro.duckdb"

        super().__init__(
            use_case,
            scale_factor,
            model_name,
            concurrent_llm_worker,
            str(db_path),
            skip_setup=skip_setup,
        )

        self._materialise_tables(str(db_path))

    @staticmethod
    def _materialise_tables(db_path: str) -> None:
        """Create per-query left/right tables in DuckDB if they don't exist."""
        conn = duckdb.connect(db_path)
        existing = {row[0] for row in conn.execute("SHOW TABLES").fetchall()}
        try:
            for query_id, (left_tbl, right_tbl) in _TABLE_NAMES.items():
                if left_tbl in existing and right_tbl in existing:
                    continue
                left_df, right_df, _ = load_query_inputs(query_id)
                # DuckDB does not have a native int64-nullable mapping; cast.
                for col in left_df.columns:
                    if str(left_df[col].dtype) == "Int64":
                        left_df[col] = left_df[col].astype("int64")
                for col in right_df.columns:
                    if str(right_df[col].dtype) == "Int64":
                        right_df[col] = right_df[col].astype("int64")

                conn.register("__left_df", left_df)
                conn.register("__right_df", right_df)
                conn.execute(
                    f'CREATE TABLE "{left_tbl}" AS SELECT * FROM __left_df'
                )
                conn.execute(
                    f'CREATE TABLE "{right_tbl}" AS SELECT * FROM __right_df'
                )
                conn.unregister("__left_df")
                conn.unregister("__right_df")
                print(f"  Created tables {left_tbl}, {right_tbl}")
        finally:
            conn.close()

    def _execute_q113(self) -> Dict[str, Any]:
        left_tbl, right_tbl = _TABLE_NAMES[113]
        sql = f"""
        SELECT L.left_name AS "left", R.right_name AS "right"
        FROM "{left_tbl}" AS L JOIN "{right_tbl}" AS R
        ON NLjoin(L.left_name, R.right_name,
                  'these two senior government official names share the same first name (first given name, ignoring middle names or hyphenated parts)')
        """
        return self.execute_thalamusdb_query(sql)

    def _execute_q121(self) -> Dict[str, Any]:
        left_tbl, right_tbl = _TABLE_NAMES[121]
        sql = f"""
        SELECT L.left_Supplier AS "left", R.right_Supplier AS "right"
        FROM "{left_tbl}" AS L JOIN "{right_tbl}" AS R
        ON NLjoin(L.left_Supplier, R.right_Supplier,
                  'these two supplier names refer to the same organisation, ignoring case, abbreviations, punctuation, trailing legal suffixes such as Ltd or Limited, and minor formatting differences')
        """
        return self.execute_thalamusdb_query(sql)

    def _execute_q202(self) -> Dict[str, Any]:
        left_tbl, right_tbl = _TABLE_NAMES[202]
        sql = f"""
        SELECT L.left_ID AS "left", R.right_ID AS "right"
        FROM "{left_tbl}" AS L JOIN "{right_tbl}" AS R
        ON NLjoin(
              L.left_name || ' | ' || L.left_address || ' | ' ||
                  L.left_phone || ' | ' || L.left_cuisine,
              R.right_name || ' | ' || R.right_address || ' | ' ||
                  R.right_phone || ' | ' || R.right_cuisine,
              'these two restaurant records refer to the same real-world restaurant (same establishment at the same address)')
        """
        return self.execute_thalamusdb_query(sql)

    def _execute_q305(self) -> Dict[str, Any]:
        left_tbl, right_tbl = _TABLE_NAMES[305]
        sql = f"""
        SELECT L.left_colname AS "left", R.right_colname AS "right"
        FROM "{left_tbl}" AS L JOIN "{right_tbl}" AS R
        ON NLjoin(
              'name=' || L.left_colname || ' | samples=' || L.left_samples,
              'name=' || R.right_colname || ' | samples=' || R.right_samples,
              'these two columns from different tables describe the same real-world feature (same semantic attribute)')
        """
        return self.execute_thalamusdb_query(sql)
