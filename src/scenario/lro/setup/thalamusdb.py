"""ThalamusDB setup for the LRO scenario.

Loads the source CSVs into a DuckDB database whose path is derived from
``LROBENCH_DATABASES_DIR``.  Q305 (schema matching) is materialised from a
column-descriptor view because ThalamusDB cannot run NL operators over
arbitrary expressions.
"""

import os

import duckdb
import pandas as pd

from scenario.lro.lro_scenario import (
    QUERY_REGISTRY,
    column_descriptor_df,
    get_lrobench_databases_dir,
)


TABLE_BY_QUERY = {
    113: ("lro_q113_left", "lro_q113_right"),
    121: ("lro_q121_left", "lro_q121_right"),
    202: ("lro_q202_left", "lro_q202_right"),
    305: ("lro_q305_left", "lro_q305_right"),
}


class ThalamusDBLROSetup:
    def __init__(self, db_name: str = "lro_thalamusdb"):
        self.db_path = os.path.join(
            get_lrobench_databases_dir(), f"{db_name}.duckdb"
        )

    def setup_data(self, data_dir: str) -> None:
        # Re-create from scratch so query files always see the latest data.
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

        con = duckdb.connect(self.db_path)
        try:
            for query_id, meta in QUERY_REGISTRY.items():
                left_table, right_table = TABLE_BY_QUERY[query_id]
                left_df, right_df = self._materialise(
                    query_id, meta, data_dir
                )

                con.register(f"{left_table}_df", left_df)
                con.register(f"{right_table}_df", right_df)
                con.execute(
                    f"CREATE OR REPLACE TABLE {left_table} AS "
                    f"SELECT * FROM {left_table}_df"
                )
                con.execute(
                    f"CREATE OR REPLACE TABLE {right_table} AS "
                    f"SELECT * FROM {right_table}_df"
                )
        finally:
            con.close()

    def _materialise(self, query_id: int, meta: dict, data_dir: str):
        if query_id == 305:
            left_df = column_descriptor_df(data_dir, meta["left_csv"], "left")
            right_df = column_descriptor_df(
                data_dir, meta["right_csv"], "right"
            )
            return left_df, right_df

        if query_id == 202:
            left_df = self._load_restaurant(data_dir, meta["left_csv"], "left")
            right_df = self._load_restaurant(
                data_dir, meta["right_csv"], "right"
            )
            return left_df, right_df

        left_df = self._load_single_column(data_dir, meta, side="left")
        right_df = self._load_single_column(data_dir, meta, side="right")
        return left_df, right_df

    @staticmethod
    def _load_csv(data_dir: str, relative_path: str) -> pd.DataFrame:
        path = os.path.join(data_dir, relative_path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"LRO source file not found: {path}")
        return pd.read_csv(path, engine="python")

    def _load_single_column(
        self, data_dir: str, meta: dict, side: str
    ) -> pd.DataFrame:
        df = self._load_csv(data_dir, meta[f"{side}_csv"])
        key = meta[f"{side}_key"]
        alias = meta[f"{side}_alias"]
        proj = df[[key]].dropna().drop_duplicates()
        proj = proj.rename(columns={key: alias})
        # ThalamusDB cannot deal with single quotes in literal columns.
        proj[alias] = proj[alias].astype(str).str.replace("'", "", regex=False)
        return proj

    def _load_restaurant(
        self, data_dir: str, relative_path: str, side: str
    ) -> pd.DataFrame:
        df = self._load_csv(data_dir, relative_path)
        if "zip" in df.columns:
            df = df[df["zip"] == 60642]
        rename_map = {
            "ID": f"{side}_ID",
            "name": f"{side}_name",
            "address": f"{side}_address",
            "phone": f"{side}_phone",
            "cuisine": f"{side}_cuisine",
        }
        df = df.rename(columns=rename_map)
        keep = list(rename_map.values())
        df = df[[c for c in keep if c in df.columns]].copy()
        for c in df.columns:
            if df[c].dtype == object:
                df[c] = df[c].astype(str).str.replace("'", "", regex=False)
        # ThalamusDB needs a single text column for NLjoin.
        df[f"{side}_description"] = (
            "name=" + df[f"{side}_name"].astype(str)
            + " | address=" + df[f"{side}_address"].astype(str)
            + " | phone=" + df[f"{side}_phone"].astype(str)
            + " | cuisine=" + df[f"{side}_cuisine"].astype(str)
        )
        return df
