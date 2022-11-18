import json
from fire import Fire
import os
import numpy as np
from typing import Dict, Optional

from constants import DEPTH_DIR, IMAGE_DIR, ORIGINAL_SUFFIX, TRANSFORMS
from point_cloud import Pointcloud

"""
Dataset structure
dataset
    - depth
        - 0000.exr  (for now these simply match the image names)
        - 0001.exr
        - ...
    - depth_original (optional, same structure as depth, depths before the cube fit)
    - train
        - 0000.jpg
        - 0001.jpg
        - ...
    - test
        - 0002.jpg
        - 0005.jpg
        - ...
    - transforms_train.json
    - transforms_test.json
    - transforms_train_original.json  (optional, same structure as transforms_train, transforms before the cube fit)
    - transforms_test_original.json  (optional, same structure as transforms_test, transforms before the cube fit)
"""

def _raise_if_invalid_dataset(dataset_path) -> None:
    depths_dir = os.path.join(dataset_path, DEPTH_DIR)
    images_dir = os.path.join(dataset_path, IMAGE_DIR)

    # make sure the datset dir contains everything we need
    # if not os.path.isdir(depths_dir):
    #     raise ValueError(f"Depth directory {depths_dir} does not exist")

    # if not os.path.isdir(images_dir):
    #     raise ValueError(f"Image directory {images_dir} does not exist")

    if "transforms_train_original.json" in os.listdir(dataset_path):
        raise ValueError("Dataset already transformed")

    transforms = ["transforms_train.json", "transforms_test.json"]

    for transform in transforms:
        transform_file = os.path.join(dataset_path, transform)
        if not os.path.isfile(transform_file):
            raise ValueError(f"Transform file {transform_file} does not exist")


def transform_frame(frame: Dict, transform_matrix: np.ndarray) -> Dict:
    matrix = np.array(frame["transform_matrix"])
    new_matrix = transform_matrix @ matrix
    frame["transform_matrix"] = new_matrix.tolist()
    return frame


def main(dataset_path: str, clipping_distance: Optional[float] = None) -> None:
    _raise_if_invalid_dataset(dataset_path)
    transforms = ["transforms_train.json", "transforms_test.json"]
    point_cloud = Pointcloud.from_dataset(dataset_path, transforms, clipping_distance=clipping_distance)
    transform_to_fit_to_unit_cube = point_cloud.fit_to_unit_cube()

    print(transform_to_fit_to_unit_cube)

    # copy old data (and make this script idempotent)
    for transform in transforms:
        transform_file = os.path.join(dataset_path, transform)
        original_transform_file = os.path.join(dataset_path, f"{os.path.splitext(transform)[0]}{ORIGINAL_SUFFIX}.json")
        os.rename(transform_file, original_transform_file)

    fitting_matrix = transform_to_fit_to_unit_cube

    for transform in transforms:
        original_transform_file = os.path.join(dataset_path, f"{os.path.splitext(transform)[0]}{ORIGINAL_SUFFIX}.json")
        new_transform_file = os.path.join(dataset_path, transform)

        with open(original_transform_file, "r") as f:
            transforms = json.load(f)

        transforms["frames"] = [transform_frame(frame, fitting_matrix.numpy()) for frame in transforms["frames"]]

        with open(new_transform_file, "w") as f:
            json.dump(transforms, f, indent=4)

        with open(os.path.join(dataset_path, "fitting_matrix.json"), "w") as f:
            json.dump(fitting_matrix.numpy().tolist(), f, indent=4)


if __name__ == '__main__':
    Fire(main)