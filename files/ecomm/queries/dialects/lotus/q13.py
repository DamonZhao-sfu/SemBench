import os
import pandas as pd
from lotus.dtype_extensions import ImageArray


def _extract_description_text(row):
    desc = row.get('productDescriptors')
    if isinstance(desc, dict):
        inner = desc.get('description')
        if isinstance(inner, dict):
            val = inner.get('value')
            if val is not None:
                return val
    return ''


def run(data_dir: str):
    # Load data
    styles_details = pd.read_parquet(os.path.join(data_dir, 'styles_details.parquet'))
    image_mapping = pd.read_parquet(os.path.join(data_dir, 'image_mapping.parquet'))
    styles_details['id'] = styles_details['id'].astype('int')
    image_mapping['image_id'] = image_mapping['id'].astype('int')
    styles_details = styles_details.merge(image_mapping, left_on='id', right_on='image_id', suffixes=('', '_img'))

    styles_details['description_text'] = styles_details.apply(_extract_description_text, axis=1)
    styles_details['image_filepath'] = ImageArray(
        styles_details.filename.apply(lambda s: os.path.join(data_dir, 'images', s))
    )

    filtered = styles_details.sem_filter(
        "You will receive a description of what a customer is looking for together with an image and a textual description of the product.\n"
        "Determine if they both match.\n"
        "\n"
        "I am looking for a running shirt for men with a round neck and short sleeves,\n"
        "preferably in blue or black, but not bright colors like white.\n"
        "Also definitely not green.\n"
        "It should be suitable for outdoor running in warm weather.\n"
        "If the t-shirt is not green, it should at least feature a striped design.\n"
        "\n"
        "The product has the following image {image_filepath} and textual description {productDisplayName} {description_text}"
    )

    return filtered['id']
