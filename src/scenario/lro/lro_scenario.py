"""
LRO (LROBench) scenario handler.

Adapts the four LRO match queries (match113, match121, match202, match305)
from `lro.py` to SemBench's runner/evaluator framework.

Source CSVs are read directly from `LROBENCH_DATABASES_DIR`
(default: /localhome/hza214/LLMSQL/databases). The scenario does not
download data; it just exposes the CSV paths to the runners.
"""

import glob
import os
from pathlib import Path
from typing import List


LRO_FILES_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "files", "lro")
)

DEFAULT_LRO_DATABASES_DIR = "/localhome/hza214/LLMSQL/databases"


# Map query id -> (relative_path_to_csv, relative_path_to_csv) for the
# left/right inputs of the match join.
LRO_QUERIES = {
    113: {
        "left": "santos/home_office_senior_officials_travel_data_return.csv",
        "right": "santos/travel-exp-April-June-2018.csv",
    },
    121: {
        "left": "santos/01.Apr_2018.csv",
        "right": "santos/2015_05_expenditure.csv",
    },
    202: {
        "left": "restaurants2/yelp.csv",
        "right": "restaurants2/zomato.csv",
    },
    305: {
        "left": "california_schools/frpm.csv",
        "right": "california_schools/schools.csv",
    },
}


def get_lro_databases_dir() -> str:
    """Return the absolute path to the LRO source databases directory."""
    return os.environ.get("LROBENCH_DATABASES_DIR", DEFAULT_LRO_DATABASES_DIR)


class LROScenario:
    """LRO scenario handler. No download step; data is on local disk."""

    def __init__(self, scale_factor: int = None):
        self.scale_factor = scale_factor if scale_factor is not None else 1
        self.data_dir = get_lro_databases_dir()

    def setup_scenario(self, systems: List[str]) -> None:
        # Validate the source databases directory exists.
        if not os.path.isdir(self.data_dir):
            raise FileNotFoundError(
                f"LRO databases directory not found: {self.data_dir}. "
                "Set LROBENCH_DATABASES_DIR to the directory containing "
                "santos/, restaurants2/, california_schools/."
            )

        # Make sure files/lro/data/sf_{sf} exists for any system that needs to
        # write artifacts (e.g. ThalamusDB duckdb file).
        sf_dir = Path(LRO_FILES_DIR) / "data" / f"sf_{self.scale_factor}"
        sf_dir.mkdir(parents=True, exist_ok=True)

        for system in systems:
            if system in ("lotus", "palimpzest", "caesura"):
                pass  # Read directly from CSVs.
            elif system == "thalamusdb":
                # ThalamusDB reads a DuckDB database; the runner builds it
                # lazily when not present.
                pass
            elif system in ("bigquery", "flockmtl", "snowflake"):
                raise NotImplementedError(
                    f"LRO scenario setup for system '{system}' is not implemented."
                )
            else:
                raise ValueError(f"Unsupported system: {system}")

    def get_query_text(self, query_id: int, system_name: str) -> str:
        system_query_dir = os.path.abspath(
            os.path.join(LRO_FILES_DIR, "query", system_name)
        )
        matching_files = glob.glob(
            os.path.join(system_query_dir, f"Q{query_id}.*")
        )
        if not matching_files:
            raise FileNotFoundError(
                f"No query file found for Q{query_id} in {system_query_dir}"
            )
        if len(matching_files) > 1:
            raise ValueError(
                f"Multiple query files found for Q{query_id} and "
                f"system '{system_name}': {matching_files}"
            )
        with open(matching_files[0], "r") as f:
            return f.read()

    def get_data_dir(self) -> str:
        """Source CSV directory (LROBENCH_DATABASES_DIR)."""
        return self.data_dir

    def get_artifacts_dir(self) -> str:
        """Per-scale-factor working directory under files/lro/data."""
        return str(
            Path(LRO_FILES_DIR) / "data" / f"sf_{self.scale_factor}"
        )

    def discover_available_queries(self, system_name: str = None) -> List[int]:
        return sorted(LRO_QUERIES.keys())
