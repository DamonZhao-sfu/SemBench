"""
Run the 8 ablation configurations from the experiment table and save the
vLLM token usage and the 4-stage time breakdown (embedding / labeling /
training / prediction) to an xlsx file.

Example:
    python scripts/run_ablation_configs.py
    python scripts/run_ablation_configs.py --system palimpzest --skip-setup
    python scripts/run_ablation_configs.py --output analysis_results/run1.xlsx

Notes on the ablation knobs
---------------------------
The runner under src/ currently does not expose "source" (emb ablation siglip
vs direct-original), "budget", or "k" as first-class arguments. Until those
knobs are wired through, this script surfaces them to the runner via
environment variables so that no matter where they are eventually consumed
(argparse, a config JSON, or directly in a runner) they are visible to the
child process:

    SEMBENCH_ABLATION_SOURCE  -- "emb_ablation_siglip" | "direct_original"
    SEMBENCH_BUDGET_FRACTION  -- e.g. "0.10"
    SEMBENCH_TOPK             -- e.g. "auto" or "10"

The script also parses the child stdout for stage-time markers of the form
"<stage>_time: <seconds>" (e.g. "embedding_time: 12.3") so that once the
runner prints them, they flow through to the xlsx automatically. Fields that
are not found are written as blanks.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_PY = REPO_ROOT / "src" / "run.py"
DEFAULT_OUTPUT = REPO_ROOT / "analysis_results" / "ablation_results.xlsx"


@dataclass(frozen=True)
class Config:
    use_case: str
    query: str          # e.g. "2a", "7", "13"
    display_query: str  # e.g. "Q2a"
    source: str         # "emb_ablation_siglip" | "direct_original"
    budget: float       # fraction, e.g. 0.10
    topk: Union[str, int]  # "auto" or an int


CONFIGS: list[Config] = [
    Config("mmqa",    "2a", "Q2a", "emb_ablation_siglip", 0.10, "auto"),
    Config("mmqa",    "7",  "Q7",  "direct_original",     0.15, "auto"),
    Config("ecomm",   "7",  "Q7",  "emb_ablation_siglip", 0.25, "auto"),
    Config("ecomm",   "8",  "Q8",  "emb_ablation_siglip", 0.05, "auto"),
    Config("ecomm",   "9",  "Q9",  "emb_ablation_siglip", 0.25, "auto"),
    Config("ecomm",   "13", "Q13", "direct_original",     0.10, "auto"),
    Config("cars",    "8",  "Q8",  "emb_ablation_siglip", 0.15, "auto"),
    Config("animals", "1",  "Q1",  "emb_ablation_siglip", 0.15, 10),
]


STAGE_PATTERNS = {
    "t_embedding_s": re.compile(
        r"embedding(?:_time|\s+time)?\s*[:=]\s*([0-9.]+)", re.IGNORECASE
    ),
    "t_labeling_s": re.compile(
        r"labeling(?:_time|\s+time)?\s*[:=]\s*([0-9.]+)", re.IGNORECASE
    ),
    "t_training_s": re.compile(
        r"training(?:_time|\s+time)?\s*[:=]\s*([0-9.]+)", re.IGNORECASE
    ),
    "t_prediction_s": re.compile(
        r"prediction(?:_time|\s+time)?\s*[:=]\s*([0-9.]+)", re.IGNORECASE
    ),
}

VLLM_TOTAL_PATTERN = re.compile(
    r"vllm[^\n]*?(?:total|tokens?)[^0-9]*([0-9][0-9,]*)", re.IGNORECASE
)
VLLM_PROMPT_PATTERN = re.compile(
    r"(?:prompt[_\s]tokens?)[^0-9]*([0-9][0-9,]*)", re.IGNORECASE
)
VLLM_COMPLETION_PATTERN = re.compile(
    r"(?:completion[_\s]tokens?)[^0-9]*([0-9][0-9,]*)", re.IGNORECASE
)


def _search_float(pattern: re.Pattern[str], text: str) -> Optional[float]:
    match = pattern.search(text)
    if not match:
        return None
    try:
        return float(match.group(1).replace(",", ""))
    except (ValueError, IndexError):
        return None


def _search_int(pattern: re.Pattern[str], text: str) -> Optional[int]:
    value = _search_float(pattern, text)
    return int(value) if value is not None else None


def _load_metrics_json(use_case: str, system: str, display_query: str) -> dict:
    path = REPO_ROOT / "files" / use_case / "metrics" / f"{system}.json"
    if not path.exists():
        return {}
    try:
        with path.open("r") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    return data.get(display_query, {}) or {}


def run_one(cfg: Config, system: str, model: str,
            skip_setup: bool, scale_factor: Optional[int]) -> dict:
    env = os.environ.copy()
    env["SEMBENCH_ABLATION_SOURCE"] = cfg.source
    env["SEMBENCH_BUDGET_FRACTION"] = f"{cfg.budget:.4f}"
    env["SEMBENCH_TOPK"] = str(cfg.topk)
    env["PYTHONUNBUFFERED"] = "1"

    cmd = [
        sys.executable,
        str(RUN_PY),
        "--systems", system,
        "--use-cases", cfg.use_case,
        "--queries", cfg.query,
        "--model", model,
    ]
    if skip_setup:
        cmd.append("--skip-setup")
    if scale_factor is not None:
        cmd.extend(["--scale-factor", str(scale_factor)])

    print(f"\n>>> {cfg.use_case} {cfg.display_query} "
          f"source={cfg.source} budget={cfg.budget} k={cfg.topk}")
    print(" ".join(cmd))

    started = time.time()
    proc = subprocess.run(
        cmd, cwd=REPO_ROOT, env=env,
        capture_output=True, text=True,
    )
    elapsed = time.time() - started

    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")

    metrics = _load_metrics_json(cfg.use_case, system, cfg.display_query)

    stage_times = {
        name: _search_float(pat, combined)
        for name, pat in STAGE_PATTERNS.items()
    }

    vllm_total = _search_int(VLLM_TOTAL_PATTERN, combined)
    vllm_prompt = _search_int(VLLM_PROMPT_PATTERN, combined)
    vllm_completion = _search_int(VLLM_COMPLETION_PATTERN, combined)
    if vllm_total is None and (vllm_prompt is not None or vllm_completion is not None):
        vllm_total = (vllm_prompt or 0) + (vllm_completion or 0)
    if vllm_total is None and "token_usage" in metrics:
        vllm_total = int(metrics["token_usage"])

    status = metrics.get("status") or (
        "success" if proc.returncode == 0 else "failed"
    )
    error = None
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-20:]
        error = "\n".join(tail) or f"exit={proc.returncode}"

    return {
        "use_case": cfg.use_case,
        "query": cfg.display_query,
        "source": cfg.source,
        "budget": cfg.budget,
        "k": cfg.topk,
        "system": system,
        "vllm_tokens": vllm_total,
        "vllm_prompt_tokens": vllm_prompt,
        "vllm_completion_tokens": vllm_completion,
        "t_embedding_s": stage_times["t_embedding_s"],
        "t_labeling_s": stage_times["t_labeling_s"],
        "t_training_s": stage_times["t_training_s"],
        "t_prediction_s": stage_times["t_prediction_s"],
        "total_time_s": metrics.get("execution_time", elapsed),
        "wall_clock_s": elapsed,
        "status": status,
        "row_count": metrics.get("row_count"),
        "money_cost": metrics.get("money_cost"),
        "error": error,
    }


def write_xlsx(rows: list[dict], meta: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "use_case", "query", "source", "budget", "k", "system",
        "vllm_tokens", "vllm_prompt_tokens", "vllm_completion_tokens",
        "t_embedding_s", "t_labeling_s", "t_training_s", "t_prediction_s",
        "total_time_s", "wall_clock_s", "status", "row_count", "money_cost",
        "error",
    ]
    results_df = pd.DataFrame(rows, columns=columns)
    meta_df = pd.DataFrame(
        [{"key": k, "value": v} for k, v in meta.items()]
    )
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        results_df.to_excel(writer, sheet_name="results", index=False)
        meta_df.to_excel(writer, sheet_name="meta", index=False)
    print(f"\nWrote {len(rows)} rows to {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--system", default="palimpzest",
        help="System runner to use (default: palimpzest)",
    )
    parser.add_argument(
        "--model", default="gemini-2.5-flash",
        help="Model name passed to src/run.py (default: gemini-2.5-flash)",
    )
    parser.add_argument(
        "--skip-setup", action="store_true",
        help="Pass --skip-setup to src/run.py (reuse already-prepared data)",
    )
    parser.add_argument(
        "--scale-factor", type=int, default=None,
        help="Optional scale factor to pass through to src/run.py",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help=f"Output xlsx path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the planned subprocess calls without executing them",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.dry_run:
        for cfg in CONFIGS:
            print(f"{cfg.use_case}\t{cfg.display_query}\t{cfg.source}\t"
                  f"budget={cfg.budget}\tk={cfg.topk}")
        return 0

    rows: list[dict] = []
    meta = {
        "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "system": args.system,
        "model": args.model,
        "skip_setup": args.skip_setup,
        "scale_factor": args.scale_factor,
        "output": str(args.output),
    }

    for cfg in CONFIGS:
        try:
            row = run_one(
                cfg,
                system=args.system,
                model=args.model,
                skip_setup=args.skip_setup,
                scale_factor=args.scale_factor,
            )
        except Exception as exc:  # pragma: no cover - defensive
            row = {
                "use_case": cfg.use_case,
                "query": cfg.display_query,
                "source": cfg.source,
                "budget": cfg.budget,
                "k": cfg.topk,
                "system": args.system,
                "status": "failed",
                "error": f"wrapper exception: {exc}",
            }
        rows.append(row)
        # Incremental write so a crash keeps partial progress.
        meta["finished_utc"] = datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        )
        write_xlsx(rows, meta, args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
