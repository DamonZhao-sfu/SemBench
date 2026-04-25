import os
import pandas as pd
import palimpzest as pz


PROMPT = (
    "Determine whether the two restaurant records refer to the same real-world "
    "restaurant (same establishment at the same address)."
)


def _load_side(path: str, side: str) -> pd.DataFrame:
    df = pd.read_csv(path, engine="python", on_bad_lines="skip")
    if "zip" in df.columns:
        df = df[df["zip"] == 60642]
    rename_map = {
        "ID": f"{side}_ID",
        "name": f"{side}_name",
        "address": f"{side}_address",
        "phone": f"{side}_phone",
        "cuisine": f"{side}_cuisine",
    }
    df = df.rename(columns=rename_map)
    keep = list(rename_map.values())
    return df[[c for c in keep if c in df.columns]].copy().reset_index(drop=True)


def run(pz_config, data_dir: str, scale_factor: int = None):
    left = _load_side(os.path.join(data_dir, "restaurants2/yelp.csv"), "left")
    right = _load_side(os.path.join(data_dir, "restaurants2/zomato.csv"), "right")

    left_ds = pz.MemoryDataset(id="lro_q202_left", vals=left)
    right_ds = pz.MemoryDataset(id="lro_q202_right", vals=right)

    joined = left_ds.sem_join(
        right_ds,
        PROMPT,
        depends_on=[
            "left_name", "left_address", "left_phone", "left_cuisine",
            "right_name", "right_address", "right_phone", "right_cuisine",
        ],
    )
    joined = joined.project(["left_ID", "right_ID"])

    return joined.run(pz_config)
