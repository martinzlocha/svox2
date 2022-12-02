import json
import os

import imageio.v2 as imageio
import numpy as np
from fire import Fire
from tqdm import tqdm


def parse_poses_file(poses_file):
    with open(poses_file, 'r') as f:
        lines = f.readlines()
    poses = []

    for i in range(0, len(lines), 4):
        pose = []
        for j in range(4):
            pose.append(list(map(float, lines[i + j].split())))

        poses.append(pose)

    return poses

def main(dataset_dir: str):
    poses_file = os.path.join(dataset_dir, 'trainval_poses.txt')
    images_dir = os.path.join(dataset_dir, 'images')
    depths_dir = os.path.join(dataset_dir, 'depth')
    focal_file = os.path.join(dataset_dir, 'focal.txt')

    poses = parse_poses_file(poses_file)

    os.makedirs(os.path.join(dataset_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, 'test'), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, 'train_depth'), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, 'test_depth'), exist_ok=True)

    train = 8
    test = 2
    split_index = ['train'] * train + ['test'] * test

    # read focal length from focal
    with open(focal_file, 'r') as f:
        focal = float(f.read())

    camera_angle_x = 2 * np.arctan(320 / focal)

    train_json_data = {"frames": [], "camera_angle_x": camera_angle_x}
    test_json_data = {"frames": [], "camera_angle_x": camera_angle_x}

    for i, pose in enumerate(tqdm(poses)):
        image_file = os.path.join(images_dir, f'img{i}.png')
        depth_file = os.path.join(depths_dir, f'depth{i}.png')

        # img = imageio.imread(image_file)
        depth = imageio.imread(depth_file)
        depth = (np.array(depth) / 1000).astype(np.float32)
        depth = np.stack([depth, depth, depth], axis=2)

        # print(depth.shape)

        split = split_index[i % (train + test)]

        img_datadir = os.path.join(dataset_dir, split)
        depth_datadir = os.path.join(dataset_dir, f'{split}_depth')

        target_img_file_path = os.path.join(img_datadir, f'{i:04}.png')
        target_depth_file_path = os.path.join(depth_datadir, f'{i:04}.exr')

        # imageio.imwrite(target_img_file_path, img)
        # copy image to the target directory
        os.system(f'cp {image_file} {target_img_file_path}')
        imageio.imwrite(target_depth_file_path, depth)

        if split == 'train':
            data = train_json_data
        else:
            data = test_json_data

        data['frames'].append({
            "image_id": i,
            "transform_matrix": pose,
            "file_path": target_img_file_path,
            "depth_path": target_depth_file_path,
        })

    with open(os.path.join(dataset_dir, 'transforms_train.json'), 'w') as f:
        json.dump(train_json_data, f)

    with open(os.path.join(dataset_dir, 'transforms_test.json'), 'w') as f:
        json.dump(test_json_data, f)


if __name__ == "__main__":
    Fire(main)
