#!/usr/bin/env python3
"""
compute_selectivity.py
======================

Compute the *selectivity* of every SemBench query, for every scenario, at the
highest scale factor that the benchmark uses for that scenario.

What "selectivity" means here
-----------------------------
Selectivity is the fraction of the base/input table that the query keeps:

        selectivity = (rows selected by the query) / (rows in the base table)

Because SemBench queries have different shapes, two complementary numbers are
reported and the most meaningful one is surfaced in the ``selectivity`` column:

  * result_selectivity    = |ground_truth_result| / |base_table|
        - For a plain semantic *filter* this is exactly the selectivity.
        - For a *join* / pair query the result is a set of tuples, so the ratio
          can exceed 1 (it measures the join fan-out).
        - For a *ranking* / *classification* query (returns one row per input
          row) it is ~1.0.

  * predicate_selectivity = |base rows matching the query predicate| / |base_table|
        - For an *aggregation* (COUNT / AVG / ratio / GROUP BY) the result
          collapses to a scalar, so ``result_selectivity`` is meaningless; the
          predicate selectivity instead reports the fraction of base rows that
          feed the aggregate -- which is the number people actually care about.
        - For a join it reports the per-table filter selectivity before the join.

The headline ``selectivity`` column = predicate_selectivity for aggregations,
otherwise result_selectivity.

Highest scale factors (see files/<scenario>/metrics/*):
    movie = 16000, ecomm = 4000, cars = 157376, animals = 1600,
    mmqa = 800, medical = 11112 (the full, fixed dataset)

Data availability
-----------------
Most scenarios need their multi-modal dataset, which the scenario's own
preparation code downloads from Google Drive. This script does NOT download
anything; it computes from whatever is already on disk:

  * If the dataset for the target scale factor is present  -> exact numbers.
  * If only a smaller scale factor is present              -> computed there and
    flagged (movie / animals demo at the available sf).
  * cars uses the committed ground-truth at sf=157376 (the labelled source data
    is not shipped, but the exact ground-truth results are).
  * mmqa uses the committed JSON ground-truth (the base table size == scale
    factor by construction, so no data download is required).
  * If nothing is available the query is reported as "data unavailable" together
    with the command needed to generate the data.

Usage
-----
    python3 compute_selectivity.py                 # all scenarios, max sf
    python3 compute_selectivity.py --scenarios movie medical
    python3 compute_selectivity.py --scale-factor movie=16000 ecomm=4000
    python3 compute_selectivity.py --out analysis_results/selectivity.csv
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

import pandas as pd

try:  # Python 3.11+
    import tomllib as _toml
    def _toml_load(fp):  # noqa: ANN001
        return _toml.load(fp)
except ModuleNotFoundError:  # pragma: no cover
    import tomli as _toml
    def _toml_load(fp):  # noqa: ANN001
        return _toml.load(fp)

try:
    import duckdb
except ModuleNotFoundError:  # pragma: no cover
    duckdb = None


ROOT = Path(__file__).resolve().parent
FILES = ROOT / "files"

# Highest scale factor used per scenario (largest sf in files/<sc>/metrics/).
DEFAULT_SF = {
    "movie": 16000,
    "ecomm": 4000,
    "cars": 157376,
    "animals": 1600,
    "mmqa": 800,
    "medical": 11112,  # full, fixed-size dataset
}

SCENARIOS = ["movie", "ecomm", "cars", "animals", "mmqa", "medical"]


# --------------------------------------------------------------------------- #
# Result record
# --------------------------------------------------------------------------- #
@dataclass
class Row:
    scenario: str
    query: str
    type: str
    scale_factor: int
    base_table: str
    input_rows: Optional[int] = None
    result_rows: Optional[int] = None
    result_selectivity: Optional[float] = None
    selected_rows: Optional[int] = None
    predicate_selectivity: Optional[float] = None
    selectivity: Optional[float] = None          # headline value
    status: str = "ok"
    notes: str = ""

    def finalize(self) -> "Row":
        """Derive ratios + headline selectivity from the raw counts."""
        if self.input_rows:
            if self.result_rows is not None:
                self.result_selectivity = self.result_rows / self.input_rows
            if self.selected_rows is not None:
                self.predicate_selectivity = self.selected_rows / self.input_rows
        if self.type == "aggregation":
            self.selectivity = self.predicate_selectivity
        else:
            self.selectivity = (
                self.result_selectivity
                if self.result_selectivity is not None
                else self.predicate_selectivity
            )
        return self


def _ratio(num: Optional[int], den: Optional[int]) -> Optional[float]:
    if num is None or not den:
        return None
    return num / den


# --------------------------------------------------------------------------- #
# Helpers to locate data
# --------------------------------------------------------------------------- #
def find_data_dir(scenario: str, target_sf: int, required: list[str]):
    """Return (dir, actual_sf, is_target). Falls back to the largest available
    sf directory that contains every file in *required*."""
    base = FILES / scenario / "data"
    target = base / f"sf_{target_sf}"
    if all((target / f).exists() for f in required):
        return target, target_sf, True

    candidates = []
    for d in sorted(glob.glob(str(base / "sf_*"))):
        name = os.path.basename(d)
        try:
            sf = int(name.split("_")[1])
        except (IndexError, ValueError):
            continue
        if all((Path(d) / f).exists() for f in required):
            candidates.append((sf, Path(d)))
    if not candidates:
        return None, target_sf, False
    candidates.sort()  # largest sf last
    sf, d = candidates[-1]
    return d, sf, (sf == target_sf)


def _gen_hint(scenario: str, sf: int) -> str:
    hints = {
        "movie": f"python3 src/scenario/movie/preparation/generate_data.py --download-from-drive --scale-factor {sf}",
        "ecomm": f"python3 src/scenario/ecomm/preparation/generate_data.py --download-from-drive --scale-factor {sf}",
        "animals": f"python3 src/run.py --systems lotus --use-cases animals --scale-factor {sf}  # triggers data setup",
        "cars": f"python3 src/run.py --systems lotus --use-cases cars --scale-factor {sf}",
        "mmqa": f"python3 src/run.py --systems lotus --use-cases mmqa --scale-factor {sf}",
    }
    return hints.get(scenario, "")


# --------------------------------------------------------------------------- #
# MOVIE  (gold SQL on Movies.csv / Reviews.csv via DuckDB)
# --------------------------------------------------------------------------- #
def selectivity_movie(sf: int) -> list[Row]:
    ANTMAN = "ant_man_and_the_wasp_quantumania"
    cnt_taken3 = "SELECT COUNT(*) AS c FROM Reviews WHERE id='taken_3'"
    cnt_taken3_pos = (
        "SELECT COUNT(*) AS c FROM Reviews "
        "WHERE id='taken_3' AND scoreSentiment='POSITIVE'"
    )
    cnt_antman = f"SELECT COUNT(*) AS c FROM Reviews WHERE id='{ANTMAN}'"
    specs = {
        1:  ("filter",      "Reviews", None,            "clearly positive reviews (NL: LIMIT 5)"),
        2:  ("filter",      "Reviews", None,            "positive reviews for taken_3 (NL: LIMIT 5)"),
        3:  ("aggregation", "Reviews", cnt_taken3_pos,  "COUNT positive reviews for taken_3"),
        4:  ("aggregation", "Reviews", cnt_taken3,      "positivity ratio over taken_3 reviews"),
        5:  ("join",        "Reviews", cnt_antman,      "self-join: same-sentiment pairs (ant_man); result=pairs"),
        6:  ("join",        "Reviews", cnt_antman,      "self-join: opposite-sentiment pairs (ant_man), NL: LIMIT 10"),
        7:  ("join",        "Reviews", cnt_antman,      "self-join: all opposite-sentiment pairs (ant_man)"),
        8:  ("aggregation", "Reviews", cnt_taken3,      "GROUP BY sentiment for taken_3"),
        9:  ("rank",        "Reviews", cnt_antman,      "score each ant_man review (subset of Reviews)"),
        10: ("rank",        "Movies",  None,            "rank every movie (returns all rows)"),
    }
    data_dir, actual_sf, is_target = find_data_dir(
        "movie", sf, ["Movies.csv", "Reviews.csv"]
    )
    if data_dir is None:
        return [
            Row("movie", f"Q{q}", specs[q][0], sf, specs[q][1],
                status="data unavailable", notes=_gen_hint("movie", sf))
            for q in sorted(specs)
        ]

    movies = pd.read_csv(data_dir / "Movies.csv")
    reviews = pd.read_csv(data_dir / "Reviews.csv")
    con = duckdb.connect()
    con.register("Movies", movies)
    con.register("Reviews", reviews)
    n_reviews, n_movies = len(reviews), len(movies)

    note_sf = "" if is_target else f"computed at available sf={actual_sf} (target {sf} not on disk); "
    rows: list[Row] = []
    for q in sorted(specs):
        qtype, base, pred_sql, note = specs[q]
        gold = (FILES / "movie" / "query" / "gold_sql" / f"Q{q}.sql").read_text().strip()
        res = con.execute(gold).fetchdf()
        input_rows = n_reviews if base == "Reviews" else n_movies
        r = Row("movie", f"Q{q}", qtype, actual_sf, base,
                input_rows=input_rows, result_rows=len(res), notes=note_sf + note)
        if pred_sql is not None:
            r.selected_rows = int(con.execute(pred_sql).fetchone()[0])
        elif qtype == "rank" and base == "Movies":
            r.selected_rows = input_rows           # ranks everything
        else:
            r.selected_rows = len(res)             # filter: selected == result
        rows.append(r.finalize())
    con.close()
    return rows


# --------------------------------------------------------------------------- #
# ANIMALS  (gold SQL on image_data.csv / audio_data.csv via DuckDB)
# --------------------------------------------------------------------------- #
def selectivity_animals(sf: int) -> list[Row]:
    # base table(s) per query and an optional predicate-count SQL
    img = "ImageData"
    aud = "AudioData"
    both = "ImageData+AudioData"
    z = "SELECT COUNT(*) AS c FROM ImageData WHERE Species LIKE '%ZEBRA%'"
    e = "SELECT COUNT(*) AS c FROM AudioData WHERE Animal='Elephant'"
    specs = {
        1:  ("aggregation", img,  z,    "COUNT zebra images"),
        2:  ("aggregation", aud,  e,    "COUNT elephant audio"),
        3:  ("rank",        img,  z,    "city with most zebra images (result=cities)"),
        4:  ("rank",        aud,  e,    "city with most elephant audio (result=cities)"),
        5:  ("set",         both, None, "cities with elephant image OR audio (result=cities)"),
        6:  ("set",         both, None, "cities with monkey image but NO monkey audio (result=cities)"),
        7:  ("set",         img,  None, "cities where zebra AND impala co-occur in images (result=cities)"),
        8:  ("set",         both, None, "cities with elephant AND monkey across modalities (result=cities)"),
        9:  ("set",         both, None, "cities with monkey image AND monkey audio (result=cities)"),
        10: ("rank",        img,  z,    "city+station with most zebra images (result=tuples)"),
    }
    data_dir, actual_sf, is_target = find_data_dir(
        "animals", sf, ["image_data.csv", "audio_data.csv"]
    )
    if data_dir is None:
        return [
            Row("animals", f"Q{q}", specs[q][0], sf, specs[q][1],
                status="data unavailable", notes=_gen_hint("animals", sf))
            for q in sorted(specs)
        ]

    image_df = pd.read_csv(data_dir / "image_data.csv")
    audio_df = pd.read_csv(data_dir / "audio_data.csv")
    con = duckdb.connect()
    con.register("ImageData", image_df)
    con.register("AudioData", audio_df)
    n_img, n_aud = len(image_df), len(audio_df)
    base_rows = {img: n_img, aud: n_aud, both: n_img + n_aud}

    note_sf = "" if is_target else f"computed at available sf={actual_sf} (target {sf} not on disk); "
    rows: list[Row] = []
    for q in sorted(specs):
        qtype, base, pred_sql, note = specs[q]
        gold = (FILES / "animals" / "query" / "gold_sql" / f"Q{q}.sql").read_text().strip()
        res = con.execute(gold).fetchdf()
        r = Row("animals", f"Q{q}", qtype, actual_sf, base,
                input_rows=base_rows[base], result_rows=len(res), notes=note_sf + note)
        if pred_sql is not None:
            r.selected_rows = int(con.execute(pred_sql).fetchone()[0])
        else:
            r.selected_rows = len(res)
        rows.append(r.finalize())
    con.close()
    return rows


# --------------------------------------------------------------------------- #
# MEDICAL  (replicates the pandas ground-truth logic of the MedicalEvaluator)
# --------------------------------------------------------------------------- #
def selectivity_medical(sf: int) -> list[Row]:
    data = FILES / "medical" / "data"
    suffix = "" if sf == 11112 else f"_{sf}"

    def load(name: str) -> pd.DataFrame:
        return pd.read_csv(data / f"{name}{suffix}.csv")

    try:
        patients = load("patient_data_with_labels")
        symptoms = load("text_symptoms_data")
    except FileNotFoundError:
        specs = [f"Q{i}" for i in range(1, 11)]
        return [
            Row("medical", q, "filter", sf, "patients",
                status="data unavailable",
                notes="missing patient_data_with_labels.csv")
            for q in specs
        ]

    n = len(patients)
    p = patients

    abn_audio = ~p["audio_diagnosis"].isin(["none", "normal"])
    abn_xray = ~p["x_ray_diagnosis"].isin(["none", "00_normal"])

    # (query, type, result_rows, selected_rows, note)
    q1_sel = int((p["text_diagnosis"] == "allergy").sum())
    q2_sel = int(((p["smoking_history"] != "Current") & (p["audio_diagnosis"] == "normal")).sum())
    q3_sel = int(((p["did_family_have_cancer"] == 1) & abn_xray).sum())
    q4_sel = int((p["text_diagnosis"] == "acne").sum())
    q5_sel = int(((p["smoking_history"] == "Current") & abn_audio & abn_xray).sum())
    q6_cond = ~(
        (p["text_diagnosis"] == "none")
        & p["audio_diagnosis"].isin(["normal", "none"])
        & p["x_ray_diagnosis"].isin(["none", "00_normal"])
    )
    q6_sick = p[q6_cond]
    q6_mask = (
        q6_sick["text_diagnosis"] + ", "
        + q6_sick["audio_diagnosis"] + ", "
        + q6_sick["x_ray_diagnosis"]
    ).apply(lambda x: (x.split(", ").count("normal") > 0) or (x.split(", ").count("00_normal") > 0))
    q6_sel = int(q6_mask.sum())
    q7_sel = int((p["is_sick"] == 1).sum())
    q8_sel = int(((p["did_family_have_cancer"] == 1) & (p["skin_cancer_diagnosis"] == "malignant")).sum())
    q9_sel = int(((p["skin_cancer_diagnosis"] == "malignant") & abn_xray).sum())
    q10_rows = len(patients.join(symptoms.set_index("patient_id"), on="patient_id", how="inner"))

    specs = [
        ("Q1",  "filter",      q1_sel,  q1_sel,  "patients with text_diagnosis=allergy"),
        ("Q2",  "filter",      q2_sel,  q2_sel,  "non-current smokers with normal audio"),
        ("Q3",  "filter",      q3_sel,  q3_sel,  "family cancer & abnormal x-ray (NL: LIMIT 5)"),
        ("Q4",  "aggregation", 1,       q4_sel,  "AVG age of acne patients"),
        ("Q5",  "aggregation", 1,       q5_sel,  "GROUP BY smoking: current smokers, abnormal audio+xray"),
        ("Q6",  "filter",      q6_sel,  q6_sel,  "sick in >=1 modality but normal in >=1 (NL: youngest)"),
        ("Q7",  "aggregation", q7_sel,  q7_sel,  "is_sick patients (NL: AVG age); GT = matching rows"),
        ("Q8",  "filter",      q8_sel,  q8_sel,  "family cancer & malignant skin (NL: LIMIT 100)"),
        ("Q9",  "filter",      q9_sel,  q9_sel,  "malignant skin & abnormal x-ray"),
        ("Q10", "classification", q10_rows, q10_rows, "diagnose symptom text for every patient w/ symptoms"),
    ]
    rows = []
    for q, qtype, res_rows, sel_rows, note in specs:
        r = Row("medical", q, qtype, sf, "patients",
                input_rows=n, result_rows=res_rows, selected_rows=sel_rows, notes=note)
        rows.append(r.finalize())
    return rows


# --------------------------------------------------------------------------- #
# CARS  (committed ground-truth at sf=157376; labelled source data not shipped)
# --------------------------------------------------------------------------- #
def selectivity_cars(sf: int) -> list[Row]:
    gt_dir = FILES / "cars" / "raw_results" / "ground_truth"
    specs = {
        1:  ("filter",         "cars in a crash/accident (text)"),
        2:  ("filter",         "electric cars with dead-battery audio"),
        3:  ("filter",         "manual cars, no damage in images (NL: LIMIT 10)"),
        4:  ("aggregation",    "AVG age of cars with engine problems (text)"),
        5:  ("aggregation",    "COUNT automatic cars damaged in audio AND image"),
        6:  ("filter",         "cars damaged in one modality but not another"),
        7:  ("filter",         "dented(img) OR worn brakes(audio) OR electrical(text)"),
        8:  ("filter",         "punctures AND paint scratches in images (NL: LIMIT 100)"),
        9:  ("filter",         "torn(img) AND bad ignition(audio)"),
        10: ("classification", "classify every complaint into a component class"),
    }
    base_label = "Car (#cars=scale_factor)"
    rows: list[Row] = []
    for q in sorted(specs):
        qtype, note = specs[q]
        gt_file = gt_dir / f"Q{q}_{sf}.csv"
        if not gt_file.exists():
            gt_file = gt_dir / f"Q{q}.csv"
        if not gt_file.exists():
            rows.append(Row("cars", f"Q{q}", qtype, sf, base_label,
                            status="ground-truth unavailable", notes=note))
            continue
        try:
            res_rows = len(pd.read_csv(gt_file))
        except Exception as exc:  # noqa: BLE001
            rows.append(Row("cars", f"Q{q}", qtype, sf, base_label,
                            status=f"read error: {exc}", notes=note))
            continue
        r = Row("cars", f"Q{q}", qtype, sf, base_label,
                input_rows=sf, result_rows=res_rows,
                notes=f"from committed ground-truth ({gt_file.name}); " + note)
        if qtype == "aggregation":
            # labelled source data not shipped -> cannot recount the predicate
            r.notes += " | predicate selectivity needs labelled source data"
        elif qtype == "classification":
            r.selected_rows = res_rows
        else:
            r.selected_rows = res_rows
        rows.append(r.finalize())
    return rows


# --------------------------------------------------------------------------- #
# MMQA  (committed JSON ground-truth; base table size == scale factor)
# --------------------------------------------------------------------------- #
def _count_mmqa_ground_truth(gt) -> int:
    if isinstance(gt, dict):                       # q4: genre -> [movies]
        return sum(len(v) if isinstance(v, list) else 1 for v in gt.values())
    if isinstance(gt, list):
        return len(gt)
    return 1


def _mmqa_sort_key(name: str):
    # q3a -> (3, 'a'); q10 -> (10, '')
    digits = "".join(ch for ch in name if ch.isdigit())
    letters = "".join(ch for ch in name if ch.isalpha() and ch != "q")
    return (int(digits) if digits else 0, letters)


def selectivity_mmqa(sf: int) -> list[Row]:
    qdir = FILES / "mmqa" / "query" / "natural_language"
    files = sorted(qdir.glob("q*.json"), key=lambda p: _mmqa_sort_key(p.stem))
    if not files:
        return [Row("mmqa", "-", "retrieval", sf, "scaled table",
                    status="queries unavailable")]
    # type hints: q4 groups movies, q2*/q7 are image retrieval, rest are filters
    rows: list[Row] = []
    for f in files:
        qid = f.stem
        data = json.loads(f.read_text())
        gt = data.get("ground_truth", [])
        cnt = _count_mmqa_ground_truth(gt)
        if qid == "q4":
            qtype = "classification"
        elif qid in ("q2a", "q2b", "q7"):
            qtype = "retrieval"   # multimodal image retrieval
        else:
            qtype = "filter"
        nl = data.get("nl_question", "").strip()
        note = f"base table padded to scale_factor rows; {nl[:60]}"
        r = Row("mmqa", qid.upper(), qtype, sf, "scaled table (rows=scale_factor)",
                input_rows=sf, result_rows=cnt, selected_rows=cnt, notes=note)
        rows.append(r.finalize())
    return rows


# --------------------------------------------------------------------------- #
# ECOMM  (ground-truth SQL embedded in q*.toml, run with DuckDB on parquet)
# --------------------------------------------------------------------------- #
def selectivity_ecomm(sf: int) -> list[Row]:
    qdir = FILES / "ecomm" / "queries"
    toml_files = sorted(qdir.glob("q*.toml"), key=lambda p: int("".join(c for c in p.stem if c.isdigit())))
    queries = {}
    for tf in toml_files:
        with open(tf, "rb") as fh:
            data = _toml_load(fh)
        qid = data.get("metadata", {}).get("query_id")
        if qid is not None:
            queries[int(qid)] = data

    data_dir, actual_sf, is_target = find_data_dir(
        "ecomm", sf, ["styles_details.parquet"]
    )

    def qtype_of(meta) -> str:
        ops = meta.get("required_operators", [])
        if any("RANK" in o for o in ops):          # SEMANTIC_RANKING / RANK / TOP_K
            return "rank"
        if any("JOIN" in o for o in ops):
            return "join"
        if any("CLASSIF" in o for o in ops):       # CLASSIFICATION / CLASSIFY
            return "classification"
        if any("MAP" in o for o in ops):
            return "map"
        return "filter"

    if data_dir is None:
        return [
            Row("ecomm", f"Q{q}", qtype_of(queries[q]["metadata"]), sf, "styles_details",
                status="data unavailable",
                notes=_gen_hint("ecomm", sf))
            for q in sorted(queries)
        ]

    con = duckdb.connect()
    con.execute(f"SET file_search_path='{data_dir}'")
    base_rows = int(con.execute(
        "SELECT COUNT(*) FROM read_parquet('styles_details.parquet')"
    ).fetchone()[0])

    note_sf = "" if is_target else f"computed at available sf={actual_sf} (target {sf} not on disk); "
    rows: list[Row] = []
    for q in sorted(queries):
        meta = queries[q]["metadata"]
        qtype = qtype_of(meta)
        gt_sql = queries[q]["definition"]["ground_truth"]
        try:
            res = con.execute(gt_sql).fetchdf()
            r = Row("ecomm", f"Q{q}", qtype, actual_sf, "styles_details",
                    input_rows=base_rows, result_rows=len(res), selected_rows=len(res),
                    notes=note_sf + "result/base; joins can exceed 1")
            rows.append(r.finalize())
        except Exception as exc:  # noqa: BLE001
            rows.append(Row("ecomm", f"Q{q}", qtype, actual_sf, "styles_details",
                            input_rows=base_rows, status=f"query error: {exc}"))
    con.close()
    return rows


# --------------------------------------------------------------------------- #
# Orchestration / output
# --------------------------------------------------------------------------- #
COMPUTERS = {
    "movie": selectivity_movie,
    "ecomm": selectivity_ecomm,
    "cars": selectivity_cars,
    "animals": selectivity_animals,
    "mmqa": selectivity_mmqa,
    "medical": selectivity_medical,
}


def _fmt(v, pct=False):
    if v is None:
        return "-"
    if pct:
        return f"{v*100:.3f}%"
    if isinstance(v, float):
        return f"{v:.6f}"
    return str(v)


def print_table(rows: list[Row]) -> None:
    header = ["scenario", "query", "type", "sf", "input", "result",
              "selected", "selectivity", "status"]
    widths = [9, 6, 14, 8, 9, 9, 9, 12, 22]
    line = "  ".join(h.ljust(w) for h, w in zip(header, widths))
    print(line)
    print("-" * len(line))
    for r in rows:
        cells = [
            r.scenario, r.query, r.type, str(r.scale_factor),
            _fmt(r.input_rows), _fmt(r.result_rows), _fmt(r.selected_rows),
            _fmt(r.selectivity, pct=True) if r.selectivity is not None else "-",
            r.status,
        ]
        print("  ".join(str(c).ljust(w) for c, w in zip(cells, widths)))


def write_outputs(rows: list[Row], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([asdict(r) for r in rows])
    cols = ["scenario", "query", "type", "scale_factor", "base_table",
            "input_rows", "result_rows", "result_selectivity",
            "selected_rows", "predicate_selectivity", "selectivity",
            "status", "notes"]
    df = df[cols]
    df.to_csv(out_csv, index=False)

    md = out_csv.with_suffix(".md")
    with md.open("w") as fh:
        fh.write("# SemBench query selectivity (highest scale factor per scenario)\n\n")
        fh.write("selectivity = rows selected by the query / rows in the base table. ")
        fh.write("For aggregations the headline value is the predicate selectivity ")
        fh.write("(fraction of base rows feeding the aggregate); for joins it is the ")
        fh.write("result/base ratio (can exceed 1); for ranking/classification it is ~1.\n\n")
        fh.write("| scenario | query | type | sf | input rows | result rows | selected rows | selectivity | status | notes |\n")
        fh.write("|---|---|---|---:|---:|---:|---:|---:|---|---|\n")
        for r in rows:
            sel = f"{r.selectivity*100:.3f}%" if r.selectivity is not None else "-"
            fh.write(
                f"| {r.scenario} | {r.query} | {r.type} | {r.scale_factor} | "
                f"{_fmt(r.input_rows)} | {_fmt(r.result_rows)} | {_fmt(r.selected_rows)} | "
                f"{sel} | {r.status} | {r.notes} |\n"
            )
    print(f"\nWrote {out_csv}\nWrote {md}")


def parse_sf_overrides(items: list[str]) -> dict:
    out = {}
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"--scale-factor expects scenario=value, got '{it}'")
        k, v = it.split("=", 1)
        out[k.strip()] = int(v)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenarios", nargs="+", default=SCENARIOS,
                    choices=SCENARIOS, help="scenarios to compute (default: all)")
    ap.add_argument("--scale-factor", nargs="+", default=[],
                    help="override sf per scenario, e.g. movie=16000 ecomm=4000")
    ap.add_argument("--out", default=str(ROOT / "analysis_results" / "selectivity_max_sf.csv"),
                    help="output CSV path (a .md sibling is also written)")
    args = ap.parse_args()

    if duckdb is None:
        print("WARNING: duckdb not installed; movie/animals/ecomm cannot be computed.\n"
              "         pip install duckdb pandas pyarrow\n", file=sys.stderr)

    sf = dict(DEFAULT_SF)
    sf.update(parse_sf_overrides(args.scale_factor))

    all_rows: list[Row] = []
    for sc in args.scenarios:
        print(f"\n=== {sc} (target sf={sf[sc]}) ===")
        try:
            rows = COMPUTERS[sc](sf[sc])
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            rows = [Row(sc, "-", "-", sf[sc], "-", status=f"error: {exc}")]
        print_table(rows)
        all_rows.extend(rows)

    # Coverage summary: which scenarios produced exact target-sf numbers, which
    # fell back to a smaller available sf, and which need the dataset.
    print("\n" + "=" * 70)
    print("COVERAGE SUMMARY")
    print("=" * 70)
    for sc in args.scenarios:
        srows = [r for r in all_rows if r.scenario == sc]
        ok = [r for r in srows if r.status == "ok" and r.input_rows]
        target = sf[sc]
        if not ok:
            print(f"  {sc:8s}: UNAVAILABLE here -> needs dataset. {_gen_hint(sc, target) or ''}")
            continue
        sfs = sorted({r.scale_factor for r in ok})
        if sfs == [target]:
            print(f"  {sc:8s}: exact @ target sf={target}")
        else:
            print(f"  {sc:8s}: FALLBACK @ sf={sfs} (target sf={target} not on disk; "
                  f"run on a machine with the dataset for the target sf)")

    write_outputs(all_rows, Path(args.out))


if __name__ == "__main__":
    main()
