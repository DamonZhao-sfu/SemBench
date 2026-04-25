import os
import pandas as pd
import palimpzest as pz


PROMPT = (
    "Determine whether the two senior government official names share the same "
    "first name (first given name, ignoring middle names / hyphenated parts)."
)


def run(pz_config, data_dir: str, scale_factor: int = None):
    home = pd.read_csv(
        os.path.join(
            data_dir,
            "santos/home_office_senior_officials_travel_data_return.csv",
        ),
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
        engine="python",
    )
    right = (
        travel[["Senior Officials Name"]]
        .rename(columns={"Senior Officials Name": "right_name"})
        .dropna()
        .drop_duplicates()
        .reset_index(drop=True)
    )

    left_ds = pz.MemoryDataset(id="lro_q113_left", vals=left)
    right_ds = pz.MemoryDataset(id="lro_q113_right", vals=right)

    joined = left_ds.sem_join(
        right_ds,
        PROMPT,
        depends_on=["left_name", "right_name"],
    )
    joined = joined.project(["left_name", "right_name"])

    return joined.run(pz_config)
