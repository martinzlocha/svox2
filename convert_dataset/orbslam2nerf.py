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

def parse_quaternion_pose(pose: List[float], format: str = "qx qy qz qw tx ty tz") -> Tuple[np.ndarray, Quaternion]:
    if len(pose) != 7:
        raise ValueError(f"Pose should be a list of 7 elements, got {len(pose)}")

    components = format.split()
    ii = {component: i for i, component in enumerate(components)}

    qw = pose[ii["qw"]]
    qx = pose[ii["qx"]]
    qy = pose[ii["qy"]]
    qz = pose[ii["qz"]]
    tx = pose[ii["tx"]]
    ty = pose[ii["ty"]]
    tz = pose[ii["tz"]]

    translation = np.array([tx, ty, tz])
    quaternion = Quaternion(qw, qx, qy, qz)

    return translation, quaternion


def pose_to_transform(pose: List[float]) -> np.ndarray:
    translation, quaternion = parse_quaternion_pose(pose, format="tx ty tz qx qy qz qw")
    transform = np.eye(4)
    transform[:3, :3] = quaternion.rotation_matrix
    transform[:3, 3] = translation

    # https://github.com/sxyu/rgbdrec/blob/1e4ea36c0b078eed10637d08648f81f7027d30d9/rgbdreg-orbslam2.cpp#L266
    transform[0, 1] *= -1
    transform[0, 2] *= -1
    transform[1, 0] *= -1
    transform[1, 3] *= -1
    transform[2, 0] *= -1
    transform[2, 3] *= -1

    return transform


def parse_metadata_with_trajectories(metadata: dict, train_test_ratio: Tuple[int, int], trajectories: List[str]) -> Tuple[Dict, Dict]:
    poses = trajectories
    frames_train = []
    frames_test = []
    image_width = metadata['w']
    focal_length = metadata['K'][0]
    camera_angle_x = np.arctan(focal_length / (2 * image_width)) * 2

    split_idx = ['train'] * train_test_ratio[0] + ['test'] * train_test_ratio[1]

    for i, pose in enumerate(poses):
        pose = list(map(float, pose.split()))[1:]  # remove timestamp
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
        if not os.path.exists(os.path.join(dataset_dir, image_path)):
            src_img_name = f'{img_id:04d}'



        # copy image from rgb to train
        shutil.copy(f"{src_images_path}/{src_img_name}.jpg", os.path.join(dataset_dir, image_path))
        shutil.copy(f"{src_depth_path}/{src_img_name}.exr", os.path.join(dataset_dir, depth_path))

def main(dataset_dir: str):
    metadata_path = os.path.join(dataset_dir, 'metadata.json')  # from record3d
    depth_path = os.path.join(dataset_dir, 'depth')
    rgb_path = os.path.join(dataset_dir, 'rgb')
    camera_trajectory = os.path.join(dataset_dir, 'CameraTrajectory.txt')

    # load camera_trajectory file line by line to trajectories variable
    trajectories = []
    with open(camera_trajectory, 'r') as f:
        for line in f:
            trajectories.append(line)

    print("Loading metadata...")
    metadata = load_metadata(metadata_path)
    train_json, test_json = parse_metadata_with_trajectories(metadata, (8, 2), trajectories)

    print("Creating directories...")
    os.makedirs(os.path.join(dataset_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, 'test'), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, 'train_depth'), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, 'test_depth'), exist_ok=True)

    print("Writing transform files...")
    train_json["frames"] = sorted(train_json["frames"], key=lambda x: x["image_id"])
    with open(os.path.join(dataset_dir, 'transforms_train.json'), 'w') as f:
        json.dump(train_json, f, indent=4)

    test_json["frames"] = sorted(test_json["frames"], key=lambda x: x["image_id"])
    with open(os.path.join(dataset_dir, 'transforms_test.json'), 'w') as f:
        json.dump(test_json, f, indent=4)

    print("Copying images...")
    copy_images(train_json, dataset_dir, rgb_path, depth_path)
    copy_images(test_json, dataset_dir, rgb_path, depth_path)


def vector_to_extrinsics(vec: np.ndarray) -> np.ndarray:
    matrix = np.reshape(vec[:-2], (3, 5))
    return matrix

def split_ext_and_int(matrix):
    intrinsics = matrix[:, 4]
    extrinsics = matrix[:, :4]
    return intrinsics, extrinsics


if __name__ == '__main__':
    Fire(main)