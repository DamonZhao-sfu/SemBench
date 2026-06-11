import os
import pandas as pd
import palimpzest as pz


PROMPT = (
    "Determine whether the two columns from different tables describe the same "
    "real-world feature (same semantic attribute), using the column names and "
    "sample values."
)


def _column_descriptor_df(path: str, side: str, n_samples: int = 5) -> pd.DataFrame:
    df = pd.read_csv(path, engine="python", on_bad_lines="skip")
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


def run(pz_config, data_dir: str, scale_factor: int = None):
    left = _column_descriptor_df(
        os.path.join(data_dir, "california_schools/frpm.csv"), side="left"
    )
    right = _column_descriptor_df(
        os.path.join(data_dir, "california_schools/schools.csv"), side="right"
    )

    left_ds = pz.MemoryDataset(id="lro_q305_left", vals=left)
    right_ds = pz.MemoryDataset(id="lro_q305_right", vals=right)

    joined = left_ds.sem_join(
        right_ds,
        PROMPT,
        depends_on=[
            "left_colname", "left_samples",
            "right_colname", "right_samples",
        ],
    )
    joined = joined.project(["left_colname", "right_colname"])

    return joined.run(pz_config)
