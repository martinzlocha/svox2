from dataclasses import dataclass
import json
import os
from typing import List, Optional, TypeVar
import numpy as np
import torch
import imageio
import cv2
from tqdm import tqdm

from constants import DEPTH_DIR, IMAGE_DIR, ORIGINAL_SUFFIX, TRANSFORMS

@dataclass
class Rays:
    origins: torch.Tensor
    dirs: torch.Tensor

    def __post_init__(self):
        self.origins = self.origins.reshape(-1, 3)
        self.dirs = self.dirs.reshape(-1, 3)

    def __len__(self):
        return self.origins.shape[0]


def get_rays(transform: torch.Tensor, width: int, height: int, focal: float) -> Rays:
    cx = width / 2
    cy = height / 2

    yy, xx = torch.meshgrid(
            torch.flip(torch.arange(height, dtype=torch.float32), dims=[-1]) + 0.5,
            torch.arange(width, dtype=torch.float32) + 0.5,
        )

    xx = (xx - cx) / focal
    yy = (yy - cy) / focal
    zz = -torch.ones_like(xx)
    dirs = torch.stack((xx, yy, zz), dim=-1)  # OpenCV convention
    # dirs /= torch.norm(dirs, dim=-1, keepdim=True)
    dirs = dirs.reshape(-1, 3, 1)  # [H*W, 3, 1], the trailing 1 is for batch matmul
    del xx, yy, zz

    dirs = (transform[:3, :3] @ dirs)[..., 0]  # ([3, 3] @ [H*W, 3, 1])[..., 0] -> [H*W, 3]
    origins = transform[:3, 3].flatten().unsqueeze(0).expand(height * width, 3).contiguous()  # [H*W, 3]
    return Rays(origins=origins, dirs=dirs)

def load_depth_file(fpath: str) -> torch.Tensor:
    os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

    img = cv2.imread(fpath, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    depth = img[:,:,2]

    return torch.from_numpy(depth)

class Pointcloud:
    def __init__(self, points: torch.Tensor, features: Optional[torch.Tensor]=None):
        self.points = points
        self.features = features

    @classmethod
    def from_dataset(cls, dataset_path: str, transforms_to_load: List[str]) -> 'Pointcloud':
        depths_dir = os.path.join(dataset_path, DEPTH_DIR)
        images_dir = os.path.join(dataset_path, IMAGE_DIR)

        points_list: List[torch.Tensor] = []
        features_list: List[torch.Tensor] = []

        for transforms_name in transforms_to_load:
            transforms_file = os.path.join(dataset_path, f"{transforms_name}")

            with open(transforms_file) as f:
                transforms = json.load(f)

            camera_angle_x = transforms["camera_angle_x"]

            for i, frame in enumerate(tqdm(transforms["frames"], desc=f"Loading {transforms_name} pointcloud")):
                img_name = frame["file_path"]
                img_path = os.path.join(images_dir, f"{img_name}.jpg")
                img = imageio.imread(img_path)
                height, width, _ = img.shape

                depth_path = os.path.join(depths_dir, f"{img_name}.exr")
                depth = load_depth_file(depth_path)
                depth_height, depth_width = depth.shape
                depth = depth.reshape(-1, 1)

                focal = float(0.5 * depth_width / np.tan(0.5 * camera_angle_x))
                rays = get_rays(torch.tensor(frame["transform_matrix"]), depth_width, depth_height, focal)

                img_points = rays.origins + rays.dirs * depth
                points_list.append(img_points)

                img = cv2.resize(img, (depth_width, depth_height), interpolation=cv2.INTER_CUBIC)
                img = torch.from_numpy(img).reshape(-1, 3)

                assert img.shape[0] == img_points.shape[0], f"img.shape: {img.shape}, img_points.shape: {img_points.shape}"

                features_list.append(img)


        points = torch.cat(points_list, dim=0)
        features = torch.cat(features_list, dim=0)

        return cls(points=points, features=features)

    def fit_to_unit_cube(self) -> torch.Tensor:
        """
        Fits the pointcloud to a unit cube centered at the origin.
        returns the transformation matrix used to do so.
        """
        min_point = self.points.min(dim=0)
        max_point = self.points.max(dim=0)

        center = (min_point + max_point) / 2
        scale = (max_point - min_point).max()

        transform = torch.eye(4)
        transform[:3, 3] = -center
        transform[:3, :3] /= scale
        return transform

    def get_centre_of_weight(self) -> torch.Tensor:
        return self.points.mean(dim=0)

    def get_pruned_pointcloud(self, n_points) -> 'Pointcloud':
        """
        Useful for visualizing the pointcloud. Returns a pointcloud with at most n_points.
        """

        if self.points.shape[0] <= n_points:
            return self

        idx = torch.randperm(self.points.shape[0])[:n_points]
        features = self.features[idx] if self.features is not None else None
        return Pointcloud(points=self.points[idx], features=features)
