import os
from pathlib import Path
from random import random

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from tqdm import tqdm


def convert_dataset_to_images(
    images_path: str, image_shape=(28, 28), validation_percentage: float = 0.2
):
    # Iterate over each row in the DataFrame
    for dataset in ["train", "test"]:
        images_folder = f"{images_path}/images/{dataset}"
        val_images_folder = f"{images_path}/images/val"

        # Ensure the output directory exists
        Path(images_folder).mkdir(parents=True, exist_ok=True)
        df = pl.read_csv(f"{images_path}/{dataset}.csv")
        for i, row in enumerate(
            tqdm(
                df.iter_rows(),
                total=df.shape[0],
                desc=f"Converting {dataset} dataset to images",
            )
        ):
            if dataset == "train":
                row_data = np.array(row[:])
                image_data = row_data[1:].astype("uint8").reshape(image_shape)
                target_label = row_data[0]

                if random() < validation_percentage:
                    label_dir = Path(val_images_folder) / str(target_label)
                else:
                    label_dir = Path(images_folder) / str(target_label)
                label_dir.mkdir(parents=True, exist_ok=True)
            else:
                row_data = np.array(row[:])
                image_data = row_data.astype("uint8").reshape(image_shape)
                label_dir = Path(images_folder)

            # Save the image
            image_path = label_dir / f"image_{i}.jpg"
            plt.imsave(image_path, image_data, cmap="gray")
