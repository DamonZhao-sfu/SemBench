import os
import pandas as pd
from lotus.dtype_extensions import ImageArray


def run(data_dir: str, scale_factor: int = 157376):
    # Load data
    cars = pd.read_csv(
        os.path.join(data_dir, "data", f"sf_{scale_factor}", f"car_data_{scale_factor}.csv")
    )
    images = pd.read_csv(
        os.path.join(data_dir, "data", f"sf_{scale_factor}", f"image_car_data_{scale_factor}.csv")
    )

    # Make image paths absolute (the CSV stores paths relative to the repo
    # root; the LLMSQL harness does the same via
    #   concat(lit("/localhome/hza214/SemBench/"), col("image_path"))
    # data_dir is <repo_root>/files/cars, so grandparent = repo_root.
    repo_root = os.path.abspath(os.path.join(data_dir, "..", ".."))
    images["image_path"] = images["image_path"].astype(str).map(
        lambda p: p if os.path.isabs(p) else os.path.join(repo_root, p)
    )

    # Join cars with images
    joined = cars.merge(images, on="car_id", how="inner")

    # Apply semantic filter on images
    joined["image_path"] = ImageArray(joined["image_path"])
    joined = joined.sem_filter(
        "You are given an image of a vehicle or its parts. "
        "Return true if car has both, puncture and paint scratches. "
        "Image: {image_path}"
    )

    # Return every predicted car_id. The LIMIT=100 semantics are applied by
    # the SemBench evaluator (_evaluate_q8, evaluate.py:422-464) and match
    # the harness's _score_cars_q8, which also evaluates on the full
    # prediction set.
    return joined["car_id"]
