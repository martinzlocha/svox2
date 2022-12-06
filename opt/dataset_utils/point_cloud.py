from functools import partial
import json
import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import imageio
import cv2
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import open3d as o3d
import open3d.core as o3c
from dataset_utils.utils import get_rays, img_file_path_from_frame, depth_file_path_from_frame, confidence_file_path_from_frame, load_depth_file, load_confidence_file


def _get_points_and_features(frame: Dict, dataset_path: str, camera_angle_x: float, clipping_distance: Optional[float] = None, translation: Optional[torch.Tensor]=None, scaling: Optional[float]=None):
    img_path = img_file_path_from_frame(frame, dataset_path)
    img = imageio.imread(img_path)

    depth_path = depth_file_path_from_frame(frame, dataset_path)
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
        confidence_path = confidence_file_path_from_frame(frame, dataset_path)
        confidence = load_confidence_file(confidence_path)
        confidence = confidence.reshape(-1, 1)

        max_confidence = torch.max(confidence)
        img = img[confidence[:, 0] == max_confidence, :]
        img_points = img_points[confidence[:, 0] == max_confidence, :]

    return img_points, img


class Pointcloud:
    """
    Handy wrapper for all variants of pointclouds we might want to use.
    Currently supports:
        - Open3D pointclouds (o3d.geometry.PointCloud)
        - Open3d Tensor pointclouds (o3d.t.geometry.PointCloud)
        - Numpy arrays
        - PyTorch tensors
    """
    def __init__(self, points: np.ndarray, colors: Optional[np.ndarray]=None):
        if colors is not None and points.shape != colors.shape:
            raise ValueError(f"points.shape: {points.shape}, colors.shape: {colors.shape}")

        self._points = points.astype(np.float32)
        if colors is not None:
            colors = colors.astype(np.float32)
        self._colors = colors

        self._open3d_pcd = None
        self._open3d_tensor_pcd = None
        self._centre_of_mass = None

    def as_numpy(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Returns a tuple of (points, colors)
        """
        return self._points, self._colors

    def as_torch_tensor(self) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Returns a tuple of (points, colors)
        """
        pcd = torch.from_numpy(self._points), torch.from_numpy(self._colors) if self._colors is not None else None

        return pcd


    def as_open3d(self) -> o3d.geometry.PointCloud:
        """
        Returns an Open3D pointcloud
        """
        if self._open3d_pcd is not None:
            return self._open3d_pcd

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self._points)

        if self._colors is not None:
            colors = self._colors / 255
            pcd.colors = o3d.utility.Vector3dVector(colors)

        self._open3d_pcd = pcd
        return pcd

    def as_open3d_tensor(self, device: o3c.Device = o3c.Device("CPU:0"), estimate_normals: bool = False) -> o3d.t.geometry.PointCloud:
        if self._open3d_tensor_pcd is not None:
            if estimate_normals and 'normals' not in self._open3d_tensor_pcd.point:
                self._open3d_tensor_pcd.estimate_normals(nn=30, radius=0.15)
            return self._open3d_tensor_pcd.to(device)

        pcd = o3d.t.geometry.PointCloud(device)
        pcd.point.positions = o3c.Tensor(self._points, device=device, dtype=o3c.Dtype.Float32)
        if self._colors is not None:
            pcd.point.colors = o3c.Tensor(np.asarray(self._colors), device=device, dtype=o3c.Dtype.Float32)

        if estimate_normals:
            pcd.estimate_normals(max_nn=30, radius=0.15)

        self._open3d_tensor_pcd = pcd
        return pcd

    def transform_(self, transformation_matrix: np.ndarray):
        """
        In-place transformation of the pointcloud
        """
        # for performance, we might want to store everything as open3d tensor pointclouds
        points = self._points.reshape(-1, 3, 1)
        points = np.concatenate([points, np.ones((points.shape[0], 1, 1))], axis=1)
        points = transformation_matrix @ points
        self._points = points[:, :3, 0].reshape(-1, 3)
        self._open3d_pcd = None

        self._open3d_tensor_pcd = None
        self._centre_of_mass = None

    @classmethod
    def from_camera_transform(cls, camera_transform_: np.ndarray, depth_map_: np.ndarray, rgb_: Optional[np.ndarray], camera_angle_x: float, clipping_distance: Optional[float]=None):
        """
        Create a pointcloud from a depth map and a camera transform.
        """

        camera_transform = torch.from_numpy(camera_transform_)

        depth_map = torch.from_numpy(depth_map_)
        height, width = depth_map.shape

        if rgb_ is not None:
            rgb_ = cv2.resize(rgb_, (width, height), interpolation=cv2.INTER_CUBIC)
        rgb = torch.from_numpy(rgb_) if rgb_ is not None else None

        focal = float(0.5 * width / np.tan(0.5 * camera_angle_x))

        rays = get_rays(camera_transform.float(), width, height, focal)

        if clipping_distance is not None:
            depth_map = torch.clip(depth_map, 0, clipping_distance)

        points = rays.origins + rays.dirs * depth_map.reshape(-1, 1)

        if rgb is not None:
            rgb = rgb.reshape(-1, 3)

        return cls(points.numpy(), rgb.numpy() if rgb is not None else None)

    def __add__(self, other: "Pointcloud") -> "Pointcloud":
        """
        Add two pointclouds together.
        """
        points = np.concatenate([self._points, other._points], axis=0)

        if self._colors is not None and other._colors is not None:
            colors = np.concatenate([self._colors, other._colors], axis=0)
        elif self._colors is None and other._colors is None:
            colors = None
        else:
            raise(ValueError("Cannot add pointclouds with and without colors"))

        return Pointcloud(points, colors)

    def centre_of_mass(self):
        """
        Returns the centre of mass of the pointcloud
        """
        if self._centre_of_mass is not None:
            return self._centre_of_mass

        centre_of_mass = np.mean(self._points, axis=0)
        self._centre_of_mass = centre_of_mass
        return centre_of_mass

    def prune(self, n_points) -> 'Pointcloud':
        """
        Useful for visualizing the pointcloud. Returns a pointcloud with at most n_points.
        """

        print("Pruning pointcloud from {} to {}".format(self._points.shape[0], n_points))

        if self._points.shape[0] <= n_points:
            return self

        indices = np.random.choice(self._points.shape[0], n_points, replace=False)
        points = self._points[indices]
        colors = self._colors[indices] if self._colors is not None else None

        return Pointcloud(points, colors)


def stack_pointclouds(pointclouds: List[Pointcloud]) -> Pointcloud:
    """
    Stack a list of pointclouds into a single pointcloud
    """
    print("Stacking pointclouds")

    # points = np.concatenate([pc._points for pc in pointclouds], axis=0)
    with ThreadPoolExecutor() as executor:
        torch_points = list(executor.map(lambda pc: pc.as_torch_tensor()[0], pointclouds))

    points = torch.concat(torch_points, dim=0).numpy()

    if all(pc._colors is not None for pc in pointclouds):
        # colors = np.concatenate([pc._colors for pc in pointclouds], axis=0)  # type: ignore
        colors = torch.concat([pcd.as_torch_tensor()[1] for pcd in pointclouds], dim=0).numpy()  # type: ignore
    elif all(pc._colors is None for pc in pointclouds):
        colors = None
    else:
        raise(ValueError("Cannot add pointclouds with and without colors"))

    pcd = Pointcloud(points, colors)

    return pcd


class Pointcloud_DEPRECATED:
    def __init__(self, points: torch.Tensor, features: Optional[torch.Tensor]=None):
        self.points = points
        self.features = features
        self._from_frame_available = False
        self._n_frames = None
        self._frame_order = None

    @classmethod
    def from_dataset(cls, dataset_path: str,
                          transforms_to_load: List[str],
                          clipping_distance: Optional[float]=None,
                          translation: Optional[torch.Tensor]=None,
                          scaling: Optional[float]=None) -> 'Pointcloud':

        points_list: List[torch.Tensor] = []
        features_list: List[torch.Tensor] = []
        frame_ids = []

        _from_frame_available = len(transforms_to_load) == 1

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
            frame_ids.extend([frame["image_id"] for frame in transforms["frames"]])


        points = torch.cat(points_list, dim=0)
        features = torch.cat(features_list, dim=0)

        point_cloud = cls(points, features)
        if _from_frame_available:
            point_cloud._from_frame_available = True
            point_cloud._n_frames = len(frame_ids)
            point_cloud._frame_order = frame_ids

        return point_cloud

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


    def get_pruned_pointcloud(self, n_points) -> 'Pointcloud_DEPRECATED':
        """
        Useful for visualizing the pointcloud. Returns a pointcloud with at most n_points.
        """

        if self.points.shape[0] <= n_points:
            return self

        idx = torch.randperm(self.points.shape[0])[:n_points]
        features = self.features[idx] if self.features is not None else None
        return Pointcloud_DEPRECATED(points=self.points[idx], features=features)

    def from_frame(self, idx: int) -> 'Pointcloud_DEPRECATED':
        """
        Returns a pointcloud with only the points from the frame with index idx.
        """
        if not self._from_frame_available:
            raise ValueError("Cannot access frames from this pointcloud. Did you load multiple transform files?")

        assert self._n_frames is not None

        start_idx = idx * self.points.shape[0] // self._n_frames
        end_idx = (idx + 1) * self.points.shape[0] // self._n_frames

        points = self.points[start_idx:end_idx]
        features = self.features[start_idx:end_idx] if self.features is not None else None

        return Pointcloud_DEPRECATED(points=points, features=features)

    def to_open3d(self) -> o3d.geometry.PointCloud:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points.numpy())

        features = self.features
        if features is None:
            features = np.ones((self.points.shape[0], 3))
        else:
            features = features.numpy() / 255

        pcd.colors = o3d.utility.Vector3dVector(features)
        return pcd
