import os
from dataset_utils.framedata import load_frame_data_from_dataset
from fire import Fire
import open3d as o3d

from dataset_utils.point_cloud import Pointcloud, stack_pointclouds

def load_pointcloud_from_dataset(dataset_dir: str, transforms_file: str = "transforms_train.json") -> Pointcloud:
    print("Loading pointcloud from dataset...")
    frame_data = load_frame_data_from_dataset(dataset_dir, transforms_file)
    print("Stacking pointclouds...")
    pointcloud = stack_pointclouds([frame.pointcloud for frame in frame_data])

    return pointcloud


def main(dataset_dir: str, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    pointcloud = load_pointcloud_from_dataset(dataset_dir)
    o3d_pcd = pointcloud.as_open3d()
    del pointcloud
    print("Removing duplicated points...")
    # pcd_no_duplicates = o3d_pcd.remove_duplicated_points()
    # del o3d_pcd
    print("Writing pointcloud to file...")
    o3d.io.write_point_cloud(os.path.join(save_dir, "pcd.ply"), o3d_pcd, print_progress=True, write_ascii=True, compressed=True)


if __name__ == "__main__":
    Fire(main)