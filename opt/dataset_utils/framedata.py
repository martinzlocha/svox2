import json
import os
from concurrent.futures import ThreadPoolExecutor

import open3d as o3d

if o3d.__DEVICE_API__ == 'cuda':
    import open3d.cuda.pybind.t.pipelines.registration as treg
    device = o3d.core.Device("CUDA:0")
else:
    import open3d.cpu.pybind.t.pipelines.registration as treg
    device = o3d.core.Device("CPU:0")

from typing import Dict, List, Optional

import imageio.v2 as imageio
import numpy as np
from tqdm import tqdm

from dataset_utils.point_cloud import Pointcloud, stack_pointclouds
from dataset_utils.utils import (confidence_file_path_from_frame, depth_file_path_from_frame,
                                 img_file_path_from_frame, load_confidence_file, load_depth_file,
                                 invert_transformation_matrix)


class FrameData:
    def __init__(self, frame_data: Dict, dataset_dir: str, camera_angle_x: float):
        self.frame_data = frame_data
        self.dataset_dir = dataset_dir
        self.camera_angle_x = camera_angle_x

        rgb_file_path = img_file_path_from_frame(frame_data, dataset_dir)
        self.rgb: np.ndarray = imageio.imread(rgb_file_path)

        depth_file_path = depth_file_path_from_frame(frame_data, dataset_dir)
        self.depth: np.ndarray = load_depth_file(depth_file_path).numpy()

        if 'confidence_path' in frame_data:
            confidence_path = confidence_file_path_from_frame(frame_data, dataset_dir)
            confidence = load_confidence_file(confidence_path)
            confidence = confidence.reshape(self.depth.shape)

            # max_confidence = torch.max(confidence)
            # img = img[confidence[:, 0] == max_confidence, :]
            # img_points = img_points[confidence[:, 0] == max_confidence, :]
            self.confidence_map: Optional[np.ndarray] = confidence.numpy()
        else:
            self.confidence_map: Optional[np.ndarray] = None

        self.transform_matrix = np.array(frame_data["transform_matrix"])
        self.pointcloud = Pointcloud.from_camera_transform(self.transform_matrix,
                                                           self.depth,
                                                           self.rgb,
                                                           self.confidence_map,
                                                           self.camera_angle_x)

        self._matching = None

    def transform_(self, transform_matrix: np.ndarray) -> None:
        """
        In-place frame transformation
        """
        self.transform_matrix = transform_matrix @ self.transform_matrix
        self.pointcloud.transform_(transform_matrix)

    def similarity_score(self, other: "FrameData") -> float:
        # TODO: we should use covisibility matrix instead
        # return self.matching.similarity_score(other.matching)
        pointcloud_distance = self.pointcloud.centre_of_mass() - other.pointcloud.centre_of_mass()
        squared_distance = np.sum(pointcloud_distance ** 2)

        return 1 / (1 + squared_distance)

class ParentFrame:
    def __init__(self, frames: List[FrameData]):
        self.frames = frames
        self.transform_matrix = frames[0].transform_matrix

        self.pointcloud = stack_pointclouds([frame.pointcloud for frame in frames])

        # self.pointcloud = frames[0].pointcloud
        # # Aggregate point clouds
        # for frame in frames[1:]:
        #     self.pointcloud = self.pointcloud + frame.pointcloud

    def get_all_frame_transforms(self) -> List[np.ndarray]:
        """Uses original transforms and ICP to compute transforms for all frames in group."""

        criteria_list = [
            treg.ICPConvergenceCriteria(relative_fitness=0.001,
                                        relative_rmse=0.001,
                                        max_iteration=50),
            treg.ICPConvergenceCriteria(0.0001, 0.0001, 50),
            treg.ICPConvergenceCriteria(0.00001, 0.00001, 30),
            treg.ICPConvergenceCriteria(0.000001, 0.000001, 20),
            treg.ICPConvergenceCriteria(0.000001, 0.000001, 10)
        ]
        voxel_sizes = o3d.utility.DoubleVector([0.2, 0.09, 0.03, 0.008, 0.002])
        max_correspondence_distances = o3d.utility.DoubleVector([0.4, 0.2, 0.09, 0.04, 0.01])
        # new_pose_0 -> default
        trans_inv = invert_transformation_matrix(self.transform_matrix)
        transforms = [self.transform_matrix]
        for frame in self.frames[1:]:
            # new_pose_0 -> old_pose_i
            # trans_init = frame.transform_matrix @ trans_inv
            trans_init = np.eye(4)
            registration_icp = treg.multi_scale_icp(self.frames[0].pointcloud.as_open3d_tensor(),
                                  frame.pointcloud.as_open3d_tensor(),
                                  voxel_sizes,
                                  criteria_list,
                                  max_correspondence_distances,
                                  trans_init,
                                  treg.TransformationEstimationPointToPoint())
            np_transform = registration_icp.transformation.numpy()
            transforms.append(np_transform @ self.transform_matrix)
        return transforms

    def as_dict(self) -> Dict:
        return {
            "frame_ids": [frame.frame_data["image_id"] for frame in self.frames],
            "transform_matrix": self.transform_matrix.tolist(),
        }

    @classmethod
    def from_dict(cls, data: Dict, frames: Dict[int, FrameData]) -> "ParentFrame":
        parent_frames = []
        for frame_id in data["frame_ids"]:
            parent_frames.append(frames[frame_id])
        parent_frame = cls(parent_frames)
        parent_frame.transform_matrix = np.array(data["transform_matrix"])
        return parent_frame


def load_frame_data_from_dataset(dataset_dir: str,
                                 transforms_json_file: str) -> List[FrameData]:
    with open(os.path.join(dataset_dir, transforms_json_file), "r") as f:
        transforms = json.load(f)

    camera_angle_x = transforms["camera_angle_x"]

    frames = transforms["frames"]
    with ThreadPoolExecutor() as executor:
        frame_data = list(tqdm(executor.map(lambda frame: FrameData(frame,
                                                                    dataset_dir,
                                                                    camera_angle_x),
                                            frames),
                                total=len(frames)))

    return frame_data
