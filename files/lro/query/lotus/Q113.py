import os
import pandas as pd


PROMPT = (
    "Determine whether the two senior government official names share the same "
    "first name (first given name, ignoring middle names / hyphenated parts).\n\n"
    "Official A: {left_name}\n"
    "Official B: {right_name}"
)


def run(data_dir: str, scale_factor: int = None):
    home = pd.read_csv(
        os.path.join(
            data_dir,
            "santos/home_office_senior_officials_travel_data_return.csv",
        ),
        escapechar="\\",
        engine="python",
    )
    left = (
        home[["Name of Official"]]
        .rename(columns={"Name of Official": "left_name"})
        .dropna()
        .drop_duplicates()
        .reset_index(drop=True)
    )

    travel = pd.read_csv(
        os.path.join(data_dir, "santos/travel-exp-April-June-2018.csv"),
        escapechar="\\",
        engine="python",
    )
    right = (
        travel[["Senior Officials Name"]]
        .rename(columns={"Senior Officials Name": "right_name"})
        .dropna()
        .drop_duplicates()
        .reset_index(drop=True)
    )

    matched = left.sem_join(right, PROMPT)
    return matched[["left_name", "right_name"]]
