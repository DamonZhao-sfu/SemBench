"""Shared helpers for loading LRO source CSVs and building inputs for each
LRO match query (113, 121, 202, 305)."""

import os
from typing import Tuple

import pandas as pd

from scenario.lro.lro_scenario import LRO_QUERIES, get_lro_databases_dir


def load_csv(rel_path: str) -> pd.DataFrame:
    """Read a CSV from LROBENCH_DATABASES_DIR by relative path."""
    base = get_lro_databases_dir()
    full = os.path.join(base, rel_path)
    return pd.read_csv(full)


def column_descriptor_df(csv_rel_path: str, side: str, n_samples: int = 5) -> pd.DataFrame:
    """Match305 helper: turn a CSV into one row per column with a name and
    sample values, named e.g. left_colname / left_samples."""
    df = load_csv(csv_rel_path)
    rows = []
    for col in df.columns:
        samples = (
            df[col]
            .dropna()
            .astype(str)
            .head(n_samples)
            .tolist()
        )
        rows.append(
            {
                f"{side}_colname": col,
                f"{side}_samples": " | ".join(samples),
            }
        )
    return pd.DataFrame(rows)


def load_query_inputs(query_id: int) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return (left_df, right_df, meta) ready to feed a sem-join.

    meta contains:
      left_key, right_key  - the id columns used in the output
      prompt               - the natural language match instruction
    """
    if query_id == 113:
        left = load_csv(LRO_QUERIES[113]["left"])
        right = load_csv(LRO_QUERIES[113]["right"])
        left = (
            left[["Name of Official"]]
            .rename(columns={"Name of Official": "left_name"})
            .dropna()
            .drop_duplicates("left_name")
            .reset_index(drop=True)
        )
        right = (
            right[["Senior Officials Name"]]
            .rename(columns={"Senior Officials Name": "right_name"})
            .dropna()
            .drop_duplicates("right_name")
            .reset_index(drop=True)
        )
        prompt = (
            "Determine whether the two senior government official names "
            "share the same first name (first given name, ignoring middle "
            "names / hyphenated parts).\n\n"
            "Official A: {left_name}\nOfficial B: {right_name}"
        )
        return left, right, {
            "left_key": "left_name",
            "right_key": "right_name",
            "prompt": prompt,
        }

    if query_id == 121:
        left = load_csv(LRO_QUERIES[121]["left"])
        right = load_csv(LRO_QUERIES[121]["right"])
        left = (
            left[["Supplier"]]
            .rename(columns={"Supplier": "left_Supplier"})
            .dropna()
            .drop_duplicates("left_Supplier")
            .reset_index(drop=True)
        )
        right = (
            right[["Supplier"]]
            .rename(columns={"Supplier": "right_Supplier"})
            .dropna()
            .drop_duplicates("right_Supplier")
            .reset_index(drop=True)
        )
        prompt = (
            "Determine whether these two supplier names refer to the same "
            "organisation (ignoring case, abbreviations, punctuation, "
            "trailing legal suffixes such as Ltd/Limited, and minor "
            "formatting differences).\n\n"
            "Supplier A: {left_Supplier}\nSupplier B: {right_Supplier}"
        )
        return left, right, {
            "left_key": "left_Supplier",
            "right_key": "right_Supplier",
            "prompt": prompt,
        }

    if query_id == 202:
        yelp = load_csv(LRO_QUERIES[202]["left"])
        zomato = load_csv(LRO_QUERIES[202]["right"])
        yelp = yelp[yelp["zip"] == 60642].copy()
        zomato = zomato[zomato["zip"] == 60642].copy()

        left = pd.DataFrame(
            {
                "left_ID": pd.to_numeric(yelp["ID"], errors="coerce").astype("Int64"),
                "left_name": yelp["name"],
                "left_address": yelp["address"],
                "left_phone": yelp["phone"],
                "left_cuisine": yelp["cuisine"],
            }
        ).dropna(subset=["left_ID"]).reset_index(drop=True)
        right = pd.DataFrame(
            {
                "right_ID": pd.to_numeric(zomato["ID"], errors="coerce").astype("Int64"),
                "right_name": zomato["name"],
                "right_address": zomato["address"],
                "right_phone": zomato["phone"],
                "right_cuisine": zomato["cuisine"],
            }
        ).dropna(subset=["right_ID"]).reset_index(drop=True)
        prompt = (
            "Determine whether the two restaurant records refer to the same "
            "real-world restaurant (same establishment at the same address).\n\n"
            "Yelp: name={left_name} | address={left_address} | "
            "phone={left_phone} | cuisine={left_cuisine}\n"
            "Zomato: name={right_name} | address={right_address} | "
            "phone={right_phone} | cuisine={right_cuisine}"
        )
        return left, right, {
            "left_key": "left_ID",
            "right_key": "right_ID",
            "prompt": prompt,
        }

    if query_id == 305:
        left = column_descriptor_df(LRO_QUERIES[305]["left"], "left")
        right = column_descriptor_df(LRO_QUERIES[305]["right"], "right")
        prompt = (
            "Determine whether the two columns from different tables describe "
            "the same real-world feature (same semantic attribute), using the "
            "column names and sample values.\n\n"
            "Table A column: name={left_colname} | samples={left_samples}\n"
            "Table B column: name={right_colname} | samples={right_samples}"
        )
        return left, right, {
            "left_key": "left_colname",
            "right_key": "right_colname",
            "prompt": prompt,
        }

    raise ValueError(f"Unknown LRO query id: {query_id}")
