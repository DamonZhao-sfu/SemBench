import os
import pandas as pd


PROMPT = (
    "Determine whether the two columns from different tables describe the same "
    "real-world feature (same semantic attribute), using the column names and "
    "sample values.\n\n"
    "Table A column: name={left_colname} | samples={left_samples}\n"
    "Table B column: name={right_colname} | samples={right_samples}"
)


def _column_descriptor_df(path: str, side: str, n_samples: int = 5) -> pd.DataFrame:
    df = pd.read_csv(path, escapechar="\\", engine="python")
    rows = []
    for col in df.columns:
        sample_vals = df[col].dropna().astype(str).head(n_samples).tolist()
        rows.append(
            {
                f"{side}_colname": col,
                f"{side}_samples": " | ".join(sample_vals),
            }
        )
    return pd.DataFrame(rows)


def run(data_dir: str, scale_factor: int = None):
    left = _column_descriptor_df(
        os.path.join(data_dir, "california_schools/frpm.csv"), side="left"
    )
    right = _column_descriptor_df(
        os.path.join(data_dir, "california_schools/schools.csv"), side="right"
    )

    matched = left.sem_join(right, PROMPT)
    return matched[["left_colname", "right_colname"]]
