import os
import pandas as pd


PROMPT = (
    "Determine whether the two restaurant records refer to the same real-world "
    "restaurant (same establishment at the same address).\n\n"
    "Yelp: name={left_name} | address={left_address} | phone={left_phone} | "
    "cuisine={left_cuisine}\n"
    "Zomato: name={right_name} | address={right_address} | phone={right_phone} | "
    "cuisine={right_cuisine}"
)


def _load_side(path: str, side: str) -> pd.DataFrame:
    df = pd.read_csv(path, escapechar="\\", engine="python")
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
    df = df[[c for c in keep if c in df.columns]].copy().reset_index(drop=True)
    return df


def run(data_dir: str, scale_factor: int = None):
    left = _load_side(
        os.path.join(data_dir, "restaurants2/yelp.csv"), side="left"
    )
    right = _load_side(
        os.path.join(data_dir, "restaurants2/zomato.csv"), side="right"
    )

    matched = left.sem_join(right, PROMPT)
    return matched[["left_ID", "right_ID"]]
