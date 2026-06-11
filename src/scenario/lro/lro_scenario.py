"""
LRO (LRObench) scenario handler.

This scenario re-uses the matching/entity-resolution workloads outlined in
``lro.py`` (Q113, Q121, Q202, Q305).  Unlike the other scenarios in SemBench,
LRO does not download/generate its own data; it expects the source CSV files
to already exist on disk under a single root directory (one sub-folder per
benchmark dataset).  The default location can be overridden with the
``LROBENCH_DATABASES_DIR`` environment variable.
"""

import glob
import os
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


LRO_FILES_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "files", "lro"
    )
)

DEFAULT_LROBENCH_DATABASES_DIR = "/localhome/hza214/LLMSQL/databases"


def get_lrobench_databases_dir() -> str:
    """Resolve the on-disk root that contains the LRO CSV datasets."""
    return os.environ.get(
        "LROBENCH_DATABASES_DIR", DEFAULT_LROBENCH_DATABASES_DIR
    )


# Per-query metadata: source CSV files + the ID columns used for the join,
# plus the ground-truth pairs (taken verbatim from ``lro.py``).
QUERY_REGISTRY: Dict[int, Dict] = {
    113: {
        "left_csv": "santos/home_office_senior_officials_travel_data_return.csv",
        "right_csv": "santos/travel-exp-April-June-2018.csv",
        "left_key": "Name of Official",
        "right_key": "Senior Officials Name",
        "left_alias": "left_name",
        "right_alias": "right_name",
        "ground_truth": [
            ("Mark Sedwill ", "Mark Bryson-Richardson"),
            ("Richard Clarke", "Richard Montgomery"),
        ],
    },
    121: {
        "left_csv": "santos/01.Apr_2018.csv",
        "right_csv": "santos/2015_05_expenditure.csv",
        "left_key": "Supplier",
        "right_key": "Supplier",
        "left_alias": "left_Supplier",
        "right_alias": "right_Supplier",
        "ground_truth": [
            ("Nhs Supply Chain", "NHS SUPPLY CHAIN"),
            ("Novartis Pharmaceuticals Uk Ltd", "NOVARTIS PHARMACEUTICALS UK LTD"),
            ("Roche Diagnostics Limited", "ROCHE PRODUCTS LTD"),
            ("Csc", "CSC COMPUTER SCIENCES LTD"),
            ("Philips Healthcare", "PHILIPS HEALTHCARE"),
            ("Nhs Blood And Transplant", "NHS BLOOD & TRANSPLANT"),
            ("NHS Litigation Authority", "NHS LITIGATION AUTHORITY"),
        ],
    },
    202: {
        "left_csv": "restaurants2/yelp.csv",
        "right_csv": "restaurants2/zomato.csv",
        "left_key": "ID",
        "right_key": "ID",
        "left_alias": "left_ID",
        "right_alias": "right_ID",
        "ground_truth": [
            (2, 844), (72, 1020), (90, 744), (111, 564),
            (198, 1022), (275, 272), (333, 134), (418, 590),
        ],
    },
    305: {
        "left_csv": "california_schools/frpm.csv",
        "right_csv": "california_schools/schools.csv",
        "left_key": "left_colname",
        "right_key": "right_colname",
        "left_alias": "left_colname",
        "right_alias": "right_colname",
        "ground_truth": [
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
    },
}


class LROScenario:
    """LRO matching benchmark scenario handler."""

    def __init__(self, scale_factor: int = None):
        # ``scale_factor`` is ignored for LRO -- the data set is fixed -- but
        # the rest of the framework expects the parameter to exist.
        self.scale_factor = scale_factor
        self.data_dir = get_lrobench_databases_dir()

        # Expose the ground truth (and other metadata) to the evaluator.
        self.queries = QUERY_REGISTRY

    # ------------------------------------------------------------------
    # GenericRunner / GenericEvaluator hooks
    # ------------------------------------------------------------------
    def get_data_dir(self) -> str:
        return self.data_dir

    def setup_scenario(self, systems: List[str]) -> None:
        """Validate that the source CSVs exist; build per-system artifacts."""
        if not os.path.isdir(self.data_dir):
            raise FileNotFoundError(
                f"LRO databases directory not found: {self.data_dir}. "
                "Set LROBENCH_DATABASES_DIR to point at the parent folder "
                "containing 'santos/', 'restaurants2/', 'california_schools/'."
            )

        for system in systems:
            if system in ("lotus", "palimpzest"):
                # Both systems read the CSVs directly via pandas.
                continue
            if system == "thalamusdb":
                from scenario.lro.setup.thalamusdb import (
                    ThalamusDBLROSetup,
                )

                ThalamusDBLROSetup().setup_data(self.data_dir)
                continue
            raise ValueError(f"Unsupported system for LRO scenario: {system}")

    def discover_available_queries(self, system_name: str = None) -> List[int]:
        all_queries = sorted(QUERY_REGISTRY.keys())
        if system_name is None:
            return all_queries

        system_query_dir = os.path.join(LRO_FILES_DIR, "query", system_name)
        available = []
        for query_id in all_queries:
            matches = glob.glob(os.path.join(system_query_dir, f"Q{query_id}.*"))
            if matches:
                available.append(query_id)
        return available

    def get_query_text(self, query_id: int, system_name: str) -> str:
        system_query_dir = os.path.join(LRO_FILES_DIR, "query", system_name)
        matches = glob.glob(os.path.join(system_query_dir, f"Q{query_id}.*"))
        if not matches:
            raise FileNotFoundError(
                f"No LRO query implementation for Q{query_id} / {system_name}"
            )
        if len(matches) > 1:
            raise ValueError(
                f"Multiple LRO query files for Q{query_id} / {system_name}: {matches}"
            )
        with open(matches[0], "r") as f:
            return f.read()

    # ------------------------------------------------------------------
    # Helpers used by the evaluator
    # ------------------------------------------------------------------
    def get_ground_truth(self, query_id: int) -> pd.DataFrame:
        meta = QUERY_REGISTRY[int(query_id)]
        return pd.DataFrame(
            meta["ground_truth"],
            columns=[meta["left_alias"], meta["right_alias"]],
        )

    def get_query_metadata(self, query_id: int) -> Dict:
        return QUERY_REGISTRY[int(query_id)]


# ----------------------------------------------------------------------
# Helpers shared by lotus / palimpzest query files
# ----------------------------------------------------------------------
def _read_csv(data_dir: str, relative_path: str) -> pd.DataFrame:
    path = os.path.join(data_dir, relative_path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"LRO source file not found: {path}")
    # multiLine + escape='"' parity with the Spark reader in lro.py:
    # the CSVs use standard double-quote escaping (""), which pandas handles
    # via the default doublequote=True. engine="python" allows newlines in
    # quoted fields. Do NOT set escapechar — it disables the doublequote rule.
    # on_bad_lines="skip" drops the few non-RFC-4180 rows that Spark's
    # multiLine reader silently tolerates but the Python csv module rejects.
    return pd.read_csv(path, engine="python", on_bad_lines="skip")


def column_descriptor_df(
    data_dir: str, relative_path: str, side: str, n_samples: int = 5
) -> pd.DataFrame:
    """Build a column-descriptor table used by Q305 schema matching."""
    df = _read_csv(data_dir, relative_path)
    rows = []
    for col in df.columns:
        sample_vals = (
            df[col].dropna().astype(str).head(n_samples).tolist()
        )
        rows.append(
            {
                f"{side}_colname": col,
                f"{side}_samples": " | ".join(sample_vals),
            }
        )
    out = pd.DataFrame(rows)
    # ThalamusDB needs a single text column for NLjoin.
    out[f"{side}_descriptor"] = (
        "name=" + out[f"{side}_colname"].astype(str)
        + " | samples=" + out[f"{side}_samples"].astype(str)
    )
    # Strip single quotes that ThalamusDB struggles with.
    for c in out.columns:
        out[c] = out[c].astype(str).str.replace("'", "", regex=False)
    return out
