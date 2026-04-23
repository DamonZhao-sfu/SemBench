"""
Generate ground-truth CSVs for a given scenario at a given scale factor.

Assumes the underlying dataset has already been produced (see
src/scenario/<use_case>/preparation/generate_data.py). Writes ground-truth
files to files/<use_case>/raw_results/ground_truth/Q<n>.csv.

Usage:
    python3 scripts/generate_ground_truth.py --use-case ecomm   --scale-factor 4000
    python3 scripts/generate_ground_truth.py --use-case mmqa    --scale-factor 800
    python3 scripts/generate_ground_truth.py --use-case animals --scale-factor 1600
    python3 scripts/generate_ground_truth.py --use-case cars    --scale-factor 157376
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def _ecomm(scale_factor: int) -> None:
    import pandas as pd  # noqa: F401
    from scenario.ecomm.ecomm_scenario import EcommScenario

    scenario = EcommScenario(scale_factor=scale_factor)
    data_dir = Path(scenario.get_data_dir())
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Ecomm dataset not found at {data_dir}. Run: "
            f"python3 src/scenario/ecomm/preparation/generate_data.py "
            f"--download-from-drive --scale-factor {scale_factor}"
        )

    out_dir = REPO_ROOT / "files" / "ecomm" / "raw_results" / "ground_truth"
    out_dir.mkdir(parents=True, exist_ok=True)

    for query_id in sorted(scenario.queries.keys()):
        df = scenario.get_ground_truth(query_id)
        path = out_dir / f"Q{query_id}.csv"
        df.to_csv(path, index=False)
        print(f"  wrote {path} ({len(df)} rows)")


def _mmqa(scale_factor: int) -> None:
    # MMQA ground truth is scale-factor-independent; it's stored as static JSON
    # files in files/mmqa/query/natural_language/q*.json. Copy them into the
    # raw_results/ground_truth/ directory for consistency with other scenarios.
    src_dir = REPO_ROOT / "files" / "mmqa" / "query" / "natural_language"
    out_dir = REPO_ROOT / "files" / "mmqa" / "raw_results" / "ground_truth"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not src_dir.exists():
        raise FileNotFoundError(f"MMQA natural_language queries not found at {src_dir}")

    count = 0
    for src in sorted(src_dir.glob("q*.json")):
        # validate it has the ground_truth key
        with open(src) as f:
            payload = json.load(f)
        if "ground_truth" not in payload:
            print(f"  skip {src.name}: no 'ground_truth' key")
            continue
        dst = out_dir / src.name.replace("q", "Q", 1)
        shutil.copyfile(src, dst)
        count += 1
        print(f"  wrote {dst}")
    print(f"  (MMQA ground truth is static; {count} files copied. "
          f"scale_factor={scale_factor} only affects the dataset.)")


def _animals(scale_factor: int) -> None:
    from scenario.animals.evaluation.evaluate import AnimalsEvaluator

    data_dir = REPO_ROOT / "files" / "animals" / "data" / f"sf_{scale_factor}"
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Animals dataset not found at {data_dir}. Run: "
            f"python3 src/scenario/animals/preparation/generate_data.py "
            f"--download-from-drive --scale-factor {scale_factor}"
        )

    evaluator = AnimalsEvaluator(use_case="animals", scale_factor=scale_factor)
    sql_dir = REPO_ROOT / "files" / "animals" / "query" / "gold_sql"
    query_ids = sorted(
        int(p.stem[1:]) for p in sql_dir.glob("Q*.sql")
    )
    for qid in query_ids:
        df = evaluator._get_ground_truth(qid)
        path = evaluator._results_path / "ground_truth" / f"Q{qid}.csv"
        print(f"  wrote {path} ({len(df)} rows)")


def _cars(scale_factor: int) -> None:
    from scenario.cars.evaluation.evaluate import CarsEvaluator

    full_dir = REPO_ROOT / "files" / "cars" / "data" / "full_data"
    if not full_dir.exists():
        raise FileNotFoundError(
            f"Cars full_data directory not found at {full_dir}. Run: "
            f"python3 src/scenario/cars/preparation/generate_data.py "
            f"--scaling_factor {scale_factor}"
        )
    if scale_factor != 157376:
        sample_dir = REPO_ROOT / "files" / "cars" / "data" / f"sf_{scale_factor}"
        if not sample_dir.exists():
            raise FileNotFoundError(
                f"Cars sample dataset not found at {sample_dir}. Run: "
                f"python3 src/scenario/cars/preparation/generate_data.py "
                f"--scaling_factor {scale_factor}"
            )

    evaluator = CarsEvaluator(use_case="cars", scale_factor=scale_factor)
    for qid in range(1, 11):
        df = evaluator._get_ground_truth(qid)
        path = evaluator._results_path / "ground_truth" / f"Q{qid}.csv"
        print(f"  wrote {path} ({len(df)} rows)")


DISPATCH = {
    "ecomm": _ecomm,
    "mmqa": _mmqa,
    "animals": _animals,
    "cars": _cars,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--use-case", required=True, choices=sorted(DISPATCH.keys())
    )
    parser.add_argument("--scale-factor", type=int, required=True)
    args = parser.parse_args()

    print(f"Generating ground truth for {args.use_case} at sf={args.scale_factor}")
    DISPATCH[args.use_case](args.scale_factor)
    print("Done.")


if __name__ == "__main__":
    main()
