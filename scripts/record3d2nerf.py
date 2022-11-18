import json
import os
from typing import Dict, List, Tuple
import numpy as np
from fire import Fire
from pyquaternion import Quaternion
import shutil

# poses_path = '/path_to_poses/poses_bounds.npy'
# data = np.load(poses_path)

def load_metadata(metadata_path: str) -> dict:
    with open(metadata_path) as f:
        metadata = json.load(f)

    return metadata

def parse_pose(pose: List[int]) -> Tuple[np.ndarray, Quaternion]:
    if len(pose) != 7:
        raise ValueError(f"Pose should be a list of 7 elements, got {len(pose)}")

    translation = np.array(pose[4:])
    # quaternion = Quaternion(*pose[3:])
    quaternion = Quaternion(pose[3], *pose[:3])

    return translation, quaternion


def pose_to_transform(pose: List[int]) -> np.ndarray:
    translation, quaternion = parse_pose(pose)
    transform = np.eye(4)
    transform[:3, :3] = quaternion.rotation_matrix
    transform[:3, 3] = translation

    return transform


def parse_metadata(metadata: dict, train_test_ratio: Tuple[int, int]) -> Tuple[Dict, Dict]:
    poses = metadata['poses']
    frames_train = []
    frames_test = []
    image_width = metadata['w']
    focal_length = metadata['K'][0]
    camera_angle_x = np.arctan(focal_length / (2 * image_width)) * 2

    split_idx = ['train'] * train_test_ratio[0] + ['test'] * train_test_ratio[1]

    for i, pose in enumerate(poses):
        transform = pose_to_transform(pose)
        split = split_idx[i % sum(train_test_ratio)]
        frames_to_append = frames_train if split == 'train' else frames_test

        frames_to_append.append({
            "image_id": i,
            "transform_matrix": transform.tolist(),
            "file_path": f"{split}/{i:04d}.jpg",
            "depth_path": f"{split}_depth/{i:04d}.exr",
        })

    base_json = {
        "camera_angle_x": camera_angle_x,
        "camera_intrinsics": metadata['K'],
    }

    train_json = {**base_json, "frames": frames_train}

    test_json = {**base_json, "frames": frames_test}

    return train_json, test_json

def copy_images(json_data: Dict, dataset_dir: str, src_images_path: str, src_depth_path: str):
    for frame in json_data['frames']:
        image_path = frame['file_path']
        depth_path = frame['depth_path']
        img_id = frame['image_id']
        src_img_name = f'{img_id}'

        # copy image from rgb to train
        shutil.copy(f"{src_images_path}/{src_img_name}.jpg", os.path.join(dataset_dir, image_path))
        shutil.copy(f"{src_depth_path}/{src_img_name}.exr", os.path.join(dataset_dir, depth_path))

def main(dataset_dir: str):
    metadata_path = os.path.join(dataset_dir, 'metadata.json')
    depth_path = os.path.join(dataset_dir, 'depth')
    rgb_path = os.path.join(dataset_dir, 'rgb')

    print("Loading metadata...")
    metadata = load_metadata(metadata_path)
    train_json, test_json = parse_metadata(metadata, (8, 2))

    print("Creating directories...")
    os.makedirs(os.path.join(dataset_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, 'test'), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, 'train_depth'), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, 'test_depth'), exist_ok=True)

    print("Writing transform files...")
    with open(os.path.join(dataset_dir, 'transforms_train.json'), 'w') as f:
        json.dump(train_json, f, indent=4)

    with open(os.path.join(dataset_dir, 'transforms_test.json'), 'w') as f:
        json.dump(test_json, f, indent=4)

    print("Copying images...")
    copy_images(train_json, dataset_dir, rgb_path, depth_path)
    copy_images(test_json, dataset_dir, rgb_path, depth_path)


def vector_to_extrinsics(vec: np.array) -> np.array:
    matrix = np.reshape(vec[:-2], (3, 5))
    return matrix

def split_ext_and_int(matrix):
    intrinsics = matrix[:, 4]
    extrinsics = matrix[:, :4]
    return intrinsics, extrinsics


if __name__ == '__main__':
    Fire(main)