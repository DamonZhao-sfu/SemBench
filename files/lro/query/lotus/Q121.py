import os
import pandas as pd


PROMPT = (
    "Determine whether these two supplier names refer to the same organisation "
    "(ignoring case, abbreviations, punctuation, trailing legal suffixes such "
    "as Ltd/Limited, and minor formatting differences).\n\n"
    "Supplier A: {left_Supplier}\n"
    "Supplier B: {right_Supplier}"
)


def run(data_dir: str, scale_factor: int = None):
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

    matched = left.sem_join(right, PROMPT)
    return matched[["left_Supplier", "right_Supplier"]]
