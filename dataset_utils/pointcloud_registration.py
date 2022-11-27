import copy
import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

import cv2
import imageio.v2 as imageio
import numpy as np
import open3d as o3d
from fire import Fire
from tqdm import tqdm
from utils import (depth_file_path_from_frame,
                                 img_file_path_from_frame, load_depth_file)
from point_cloud import Pointcloud, stack_pointclouds
import time


if o3d.__DEVICE_API__ == 'cuda':
    import open3d.cuda.pybind.t.pipelines.registration as treg
    device = o3d.core.Device("CUDA:0")
else:
    import open3d.cpu.pybind.t.pipelines.registration as treg
    device = o3d.core.Device("CPU:0")


class FrameData:
    # class Matching:
    #     def __init__(self, rgb: np.ndarray):
    #         # detector = cv2.xfeatures2d.SIFT_create()
    #         # self.keypoints, self.descriptors = detector.detectAndCompute(rgb, None)
    #         self.hist = self.get_hist(rgb)

    #     @staticmethod
    #     def get_hist(img):
    #         hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #         h_bins = 50
    #         s_bins = 60
    #         histSize = [h_bins, s_bins]

    #         # hue varies from 0 to 179, saturation from 0 to 255
    #         h_ranges = [0, 180]
    #         s_ranges = [0, 256]
    #         ranges = h_ranges + s_ranges  # concat lists
    #         channels = [0, 1]

    #         hist = cv2.calcHist([hsv], channels, None, histSize, ranges, accumulate=False)
    #         cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    #         return hist

    #     def similarity_score(self, other: "FrameData.Matching") -> float:
    #         # matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
    #         # knn_matches = matcher.knnMatch(self.descriptors, other.descriptors, 2)

    #         # ratio_thresh = 0.7
    #         # good_matches = []
    #         # for m,n in knn_matches:
    #         #     if m.distance < ratio_thresh * n.distance:
    #         #         good_matches.append(m)

    #         # return len(good_matches) / len(self.descriptors)

    #         compare_method = cv2.HISTCMP_BHATTACHARYYA
    #         res = cv2.compareHist(self.hist, other.hist, compare_method)

    #         return 1 - res


    def __init__(self, frame_data: Dict, dataset_dir: str, camera_angle_x: float):
        self.camera_angle_x = camera_angle_x

        rgb_file_path = img_file_path_from_frame(frame_data, os.path.join(dataset_dir, "images"), dataset_dir)
        self.rgb: np.ndarray = imageio.imread(rgb_file_path)

        depth_file_path = depth_file_path_from_frame(frame_data, os.path.join(dataset_dir, "depth"), dataset_dir)
        self.depth: np.ndarray = load_depth_file(depth_file_path).numpy()

        self.transform_matrix = np.array(frame_data["transform_matrix"])
        self.pointcloud = Pointcloud.from_camera_transform(self.transform_matrix, self.depth, self.rgb, self.camera_angle_x)

        self._matching = None

    def transform_(self, transform_matrix: np.ndarray) -> None:
        """
        In-place frame transformation
        """
        self.transform_matrix = transform_matrix @ self.transform_matrix
        self.pointcloud.transform_(transform_matrix)

    # @property
    # def matching(self) -> Matching:
    #     if self._matching is None:
    #         self._matching = FrameData.Matching(self.rgb)
    #     return self._matching

    def similarity_score(self, other: "FrameData") -> float:
        # TODO: we should use covisibility matrix instead
        # return self.matching.similarity_score(other.matching)
        pointcloud_distance = self.pointcloud.centre_of_mass() - other.pointcloud.centre_of_mass()
        squared_distance = np.sum(pointcloud_distance ** 2)

        return 1 / (1 + squared_distance)


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4459,
                                      front=[0.9288, -0.2951, -0.2242],
                                      lookat=[1.6784, 2.0612, 1.4451],
                                      up=[-0.3402, -0.9189, -0.1996])


def load_frame_data_from_dataset(dataset_dir: str, transforms_json_file: str) -> List[FrameData]:
    with open(os.path.join(dataset_dir, transforms_json_file), "r") as f:
        transforms = json.load(f)

    camera_angle_x = transforms["camera_angle_x"]

    frames = transforms["frames"]
    with ThreadPoolExecutor() as executor:
        frame_data = list(tqdm(executor.map(lambda frame: FrameData(frame, dataset_dir, camera_angle_x), frames), total=len(frames)))

    return frame_data

class Camera:
    def __init__(self, transform: np.ndarray, width: int, height: int, focal: float, pointcloud: o3d.geometry.PointCloud):
        self._transform = transform
        self.width = width
        self.height = height
        self.focal = focal

        self.pointcloud = pointcloud
        self.t_pointcloud = self.get_t_pointcloud()

        self.frustum = self._get_frustum()

    def _get_frustum(self):
        """
        Returns the frustum of the camera in the form of a list of 5 points:
        [top_left, top_right, bottom_left, bottom_right, center]
        """

        xx, yy = np.meshgrid(
            np.asarray([self.height + 0.5, 0.5]),
            np.asarray([0.5, self.width + 0.5])
        )

        zz = -np.ones_like(xx)
        dirs = np.stack((xx, yy, zz), axis=-1)  # OpenCV convention

        dirs = dirs.reshape(-1, 3, 1)
        del xx, yy, zz

        dirs = (self._transform[:3, :3] @ dirs)[..., 0] * 6
        origin = self._transform[:3, 3]
        vertices = origin + dirs

        return vertices

    def transform(self, transform: np.ndarray) -> None:
        self._transform: np.ndarray = transform @ self._transform
        self.frustum = self._get_frustum()

        self.pointcloud.transform(transform)
        self.t_pointcloud = self.get_t_pointcloud()

    def get_t_pointcloud(self) -> o3d.t.geometry.PointCloud:
        pcd = o3d.t.geometry.PointCloud(device)
        pcd.point.positions = o3d.core.Tensor(np.asarray(self.pointcloud.points), device=device)
        pcd.point.colors = o3d.core.Tensor(np.asarray(self.pointcloud.colors), device=device)
        pcd.estimate_normals(max_nn=30, radius=0.15)

        return pcd

    def frustum_squared_distance(self, other: 'Camera'):
        """
        Returns the squared distance between the frustums of the two cameras.
        """
        return np.sum((self.frustum - other.frustum) ** 2)


def merge_pointclouds(pointclouds: List[o3d.t.geometry.PointCloud]) -> o3d.t.geometry.PointCloud:
    merged = pointclouds[0]
    for pcd in pointclouds[1:]:
        merged += pcd
    return merged


def get_n_closest_frames(frame_data: List[FrameData], frame: FrameData, n: int) -> List[FrameData]:
    """
    Returns the n closest frames to the given frame.
    """
    scores = np.array([frame.similarity_score(other) for other in frame_data])
    closest_indices = np.argsort(scores)[-n:]
    return [frame_data[i] for i in closest_indices]


def main(dataset_dir: str):
    transforms_train = os.path.join(dataset_dir, 'transforms_train_original.json')
    # transforms_test = os.path.join(dataset_dir, 'transforms_test.json')

    with open(transforms_train, 'r') as f:
        train_json = json.load(f)
    image_ids = [frame['image_id'] for frame in train_json['frames']]
    n_images = len(image_ids)
    # point_cloud_train = Pointcloud.from_dataset(dataset_dir, [transforms_train])

    # print(point_cloud_train._n_frames)

    frame_data = load_frame_data_from_dataset(dataset_dir, transforms_train)

    registered_frames = []

    rolling_init_transform = np.eye(4)

    big_boy_pointcloud = None

    for i, frame in enumerate(tqdm(frame_data)):
        if len(registered_frames) == 0:
            registered_frames.append(frame)
            big_boy_pointcloud = frame.pointcloud.as_open3d_tensor()
            big_boy_pointcloud.estimate_normals(max_nn=30, radius=0.2)
            continue

        tic = time.time()
        # closest_frames = get_n_closest_frames(registered_frames, frame, 5)

        # print(f"Closest frames found in {toc - tic:.2f}s")
        # tic = time.time()
        # merged_pointcloud = stack_pointclouds([f.pointcloud for f in closest_frames])

        assert big_boy_pointcloud is not None

        frame_pcd = frame.pointcloud.as_open3d_tensor().transform(rolling_init_transform)
        bounding_box = frame_pcd.get_axis_aligned_bounding_box()
        bounding_box.scale(2, center=bounding_box.get_center())
        merged_pointcloud = big_boy_pointcloud.crop(bounding_box)
        # merged_pointcloud.estimate_normals(max_nn=30, radius=0.15)

        # toc = time.time()
        # print(f"Pointclouds stacked in {toc - tic:.2f}s")

        # print("merged pointcloud has", merged_pointcloud._points.shape[0], "points")

        # reg_p2p = o3d.pipelines.registration.registration_icp(
        #     frame_pcd, merged_pointcloud, 1e-2, np.eye(4),
        #     o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        #     o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200))

        max_correspondence_distance = 0.25

        # Initial alignment or source to target transform.

        # Select the `Estimation Method`, and `Robust Kernel` (for outlier-rejection).
        # estimation = treg.TransformationEstimationPointToPoint()
        estimation = treg.TransformationEstimationPointToPlane()

        # Convergence-Criteria for Vanilla ICP
        criteria = treg.ICPConvergenceCriteria(relative_fitness=0.000001,
                                            relative_rmse=0.000001,
                                            max_iteration=50)

        voxel_sizes = o3d.utility.DoubleVector([0.25, 0.15, 0.03, 0.008, 0.002])

        criteria_list = [
            treg.ICPConvergenceCriteria(relative_fitness=0.001,
                                        relative_rmse=0.001,
                                        max_iteration=50),
            treg.ICPConvergenceCriteria(0.0001, 0.0001, 50),
            treg.ICPConvergenceCriteria(0.00001, 0.00001, 30),
            treg.ICPConvergenceCriteria(0.000001, 0.000001, 20),
            treg.ICPConvergenceCriteria(0.000001, 0.000001, 10)
        ]

        max_correspondence_distances = o3d.utility.DoubleVector([0.6, 0.4, 0.09, 0.04, 0.01])

        # Run the ICP algorithm.
        # Down-sampling voxel-size.
        voxel_size = 0.025

        tic = time.time()
        # frame_pcd = frame.pointcloud.as_open3d_tensor()
        # merged_pcd = merged_pointcloud.as_open3d_tensor()
        merged_pcd = merged_pointcloud
        toc = time.time()
        # print(f"Pointclouds converted to Open3D in {toc - tic:.2f}s")

        # registration_icp = treg.icp(frame.pointcloud.as_open3d_tensor(), merged_pointcloud.as_open3d_tensor(), max_correspondence_distance,
        #                     rolling_init_transform, estimation, criteria, voxel_size)

        tic = time.time()
        registration_icp = treg.multi_scale_icp(frame_pcd, merged_pcd, voxel_sizes,
                                           criteria_list,
                                           max_correspondence_distances,
                                           np.eye(4), estimation)

        toc = time.time()
        # print(f"ICP done in {toc - tic:.2f}s")


        # print("Transformation is:")
        # print(reg_p2p.transformation)
        # draw_registration_result(frame_pcd, overall_point_cloud, reg_p2p.transformation)

        # print(transformation.rot)
        # print(transformation.t)

        np_transform = registration_icp.transformation.numpy()
        rolling_init_transform = np_transform @ rolling_init_transform

        # frame_transforms.append(np_transform)
        # camera.transform(np_transform)
        frame.transform_(np_transform)

        if np.max(np_transform[:3, 3])>0.1:
            print(np_transform)

        if i % 20 == 0:
            registered_frames.append(frame)
            p = frame_pcd.transform(registration_icp.transformation)
            p.estimate_normals(max_nn=30, radius=0.2)

            big_boy_pointcloud += p

        # frame_pcd.points = transformation.transform(frame_pcd.points)

        # if i % 200 == 0:
        #     big_boy_pointcloud.estimate_normals(max_nn=30, radius=0.2)


    transforms_train_shifted = os.path.join(dataset_dir, 'transforms_train_original_shifted.json')

    with open(transforms_train_shifted, 'w') as f:
        new_frames = []
        for frame, registered_frame in zip(train_json['frames'], registered_frames):
            # matrix = additional_transform @ np.asarray(frame['transform_matrix'])
            matrix = registered_frame.transform_matrix
            frame['transform_matrix'] = matrix.tolist()
            new_frames.append(frame)
        json.dump(train_json, f, indent=4)


if __name__ == "__main__":
    Fire(main)
