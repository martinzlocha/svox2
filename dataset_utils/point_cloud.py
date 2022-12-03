from dataclasses import dataclass
from functools import partial
import json
import os
import liblzfse
from typing import Dict, List, Optional
import numpy as np
import torch
import imageio
import cv2
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

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

def load_confidence_file(fpath: str) -> torch.Tensor:
    with open(fpath, 'rb') as confidence_fh:
        raw_bytes = confidence_fh.read()
        decompressed_bytes = liblzfse.decompress(raw_bytes)
        confidence_img = np.frombuffer(decompressed_bytes, dtype=np.uint8)
    return torch.from_numpy(confidence_img)

def _img_file_path_from_frame(frame: dict, dataset_dir: str) -> str:
    return os.path.join(dataset_dir, frame['file_path'])

def _depth_file_path_from_frame(frame: dict, dataset_dir: str) -> str:
    return os.path.join(dataset_dir, frame['depth_path'])

def _confidence_file_path_from_frame(frame: dict, dataset_dir: str) -> str:
    return os.path.join(dataset_dir, frame['confidence_path'])

def _get_points_and_features(frame: Dict, dataset_path: str, camera_angle_x: float, clipping_distance: Optional[float] = None, translation: Optional[torch.Tensor]=None, scaling: Optional[float]=None):
    img_path = _img_file_path_from_frame(frame, dataset_path)
    img = imageio.imread(img_path)

    depth_path = _depth_file_path_from_frame(frame, dataset_path)
    depth = load_depth_file(depth_path)
    depth_height, depth_width = depth.shape
    depth = depth.reshape(-1, 1)
    if clipping_distance is not None:
        depth = torch.clip(depth, 0, clipping_distance)

    focal = float(0.5 * depth_width / np.tan(0.5 * camera_angle_x))
    transformation_matrix = torch.tensor(frame["transform_matrix"])
    if translation is not None:
        transformation_matrix[:3, 3] += translation

    if scaling is not None:
        scale_mat = scaling * torch.eye(4)
        scale_mat[-1, -1] = 1.

        transformation_matrix = scale_mat @ transformation_matrix

    rays = get_rays(transformation_matrix, depth_width, depth_height, focal)

    img_points = rays.origins + rays.dirs * depth

    img = cv2.resize(img, (depth_width, depth_height), interpolation=cv2.INTER_CUBIC)
    img = torch.from_numpy(img).reshape(-1, 3)

    assert img.shape[0] == img_points.shape[0], f"img.shape: {img.shape}, img_points.shape: {img_points.shape}"

    if 'confidence_path' in frame:
        confidence_path = _confidence_file_path_from_frame(frame, dataset_path)
        confidence = load_confidence_file(confidence_path)
        confidence = confidence.reshape(-1, 1)

        img = img[confidence[:, 0] == 2, :]
        img_points = img_points[confidence[:, 0] == 2, :]

    return img_points, img


class Pointcloud:
    def __init__(self, points: torch.Tensor, features: Optional[torch.Tensor]=None):
        self.points = points
        self.features = features

    @classmethod
    def from_dataset(cls, dataset_path: str,
                          transforms_to_load: List[str],
                          clipping_distance: Optional[float]=None,
                          translation: Optional[torch.Tensor]=None,
                          scaling: Optional[float]=None) -> 'Pointcloud':

        points_list: List[torch.Tensor] = []
        features_list: List[torch.Tensor] = []

        for transforms_name in transforms_to_load:
            transforms_file = os.path.join(dataset_path, f"{transforms_name}")

            with open(transforms_file) as f:
                transforms = json.load(f)

            camera_angle_x = transforms["camera_angle_x"]

            with ThreadPoolExecutor() as executor:
                points_features = list(tqdm(executor.map(partial(_get_points_and_features, dataset_path=dataset_path, camera_angle_x=camera_angle_x, clipping_distance=clipping_distance, translation=translation, scaling=scaling), transforms["frames"]), total=len(transforms["frames"])))

            points, features = zip(*points_features)
            points_list.extend(points)
            features_list.extend(features)


        points = torch.cat(points_list, dim=0)
        features = torch.cat(features_list, dim=0)

        return cls(points=points, features=features)

    def fit_to_unit_cube(self) -> torch.Tensor:
        """
        Fits the pointcloud to a unit cube centered at the origin.
        returns the transformation matrix used to do so.
        """
        min_point, _ = self.points.min(dim=0)
        max_point, _ = self.points.max(dim=0)

        print(f"min_point: {min_point}, max_point: {max_point}")

        center = (min_point + max_point) / 2
        scale_matrix = self.get_scale_to_unit_cube() * torch.eye(4)
        scale_matrix[3, 3] = 1

        translation_matrix = torch.eye(4)
        translation_matrix[:3, 3] = -center
        transform = torch.eye(4)
        transform = translation_matrix @ transform
        transform = scale_matrix @ transform
        return transform

    def get_scale_to_unit_cube(self) -> float:
        min_point, _ = self.points.min(dim=0)
        max_point, _ = self.points.max(dim=0)

        scale = (max_point - min_point).max()
        return 2/scale.item()

    def fit_to_sphere(self, radius) -> torch.Tensor:
        min_point, _ = self.points.min(dim=0)
        max_point, _ = self.points.max(dim=0)

        center = (min_point + max_point) / 2

        points_distances = ((self.points - center) ** 2).sum(dim=1)
        max_dist = points_distances.max().sqrt()

        scale_matrix = (radius / max_dist) * torch.eye(4)
        scale_matrix[3, 3] = 1

        translation_matrix = torch.eye(4)
        translation_matrix[:3, 3] = -center
        transform = torch.eye(4)
        transform = translation_matrix @ transform
        transform = scale_matrix @ transform
        return transform


    def get_centre_of_mass(self) -> torch.Tensor:
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
