import os
import pandas as pd
import palimpzest as pz


PROMPT = (
    "Determine whether these two supplier names refer to the same organisation "
    "(ignoring case, abbreviations, punctuation, trailing legal suffixes such "
    "as Ltd/Limited, and minor formatting differences)."
)


def run(pz_config, data_dir: str, scale_factor: int = None):
    april = pd.read_csv(
        os.path.join(data_dir, "santos/01.Apr_2018.csv"),
        engine="python",
    )
    left = (
        april[["Supplier"]]
        .rename(columns={"Supplier": "left_Supplier"})
        .dropna()
        .drop_duplicates()
        .reset_index(drop=True)
    )

    may = pd.read_csv(
        os.path.join(data_dir, "santos/2015_05_expenditure.csv"),
        engine="python",
    )
    right = (
        may[["Supplier"]]
        .rename(columns={"Supplier": "right_Supplier"})
        .dropna()
        .drop_duplicates()
        .reset_index(drop=True)
    )

    left_ds = pz.MemoryDataset(id="lro_q121_left", vals=left)
    right_ds = pz.MemoryDataset(id="lro_q121_right", vals=right)

    joined = left_ds.sem_join(
        right_ds,
        PROMPT,
        depends_on=["left_Supplier", "right_Supplier"],
    )
    joined = joined.project(["left_Supplier", "right_Supplier"])

    return joined.run(pz_config)
