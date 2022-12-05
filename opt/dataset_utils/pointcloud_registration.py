import open3d as o3d

if o3d.__DEVICE_API__ == 'cuda':
    import open3d.cuda.pybind.t.pipelines.registration as treg
    device = o3d.core.Device("CUDA:0")
else:
    import open3d.cpu.pybind.t.pipelines.registration as treg
    device = o3d.core.Device("CPU:0")
import copy
import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Literal, Union
import itertools

import imageio.v2 as imageio
import numpy as np
from fire import Fire
from tqdm import tqdm
import torch
import cv2
import matplotlib.pyplot as plt
from utils import (depth_file_path_from_frame,
                   img_file_path_from_frame,
                   load_depth_file,
                   get_rays)
from point_cloud import Pointcloud
from aabb_iou import aabb_intersection_ratios, aabb_intersection_ratios_open3d



def invert_transformation_matrix(matrix):
    # http://www.info.hiroshima-cu.ac.jp/~miyazaki/knowledge/teche0053.html
    R = matrix[:3, :3]
    t = matrix[:3, 3:]
    inverted_matrix = np.concatenate([R.T, -R.T @ t], axis=-1)
    inverted_matrix = np.concatenate([inverted_matrix,
                                 np.array([[0., 0., 0., 1.]])],
                                 axis=0)
    return inverted_matrix


class FrameData:
    def __init__(self, frame_data: Dict, dataset_dir: str, camera_angle_x: float):
        self.frame_data = frame_data
        self.dataset_dir = dataset_dir
        self.camera_angle_x = camera_angle_x

        rgb_file_path = img_file_path_from_frame(frame_data, dataset_dir)
        self.rgb: np.ndarray = imageio.imread(rgb_file_path)

        depth_file_path = depth_file_path_from_frame(frame_data, dataset_dir)
        self.depth: np.ndarray = load_depth_file(depth_file_path).numpy()

        self.transform_matrix = np.array(frame_data["transform_matrix"])
        self.pointcloud = Pointcloud.from_camera_transform(self.transform_matrix,
                                                           self.depth,
                                                           self.rgb,
                                                           self.camera_angle_x)

        self._matching = None

    def as_dict(self) -> Dict:
        return {
            "frame_data": self.frame_data,
            "dataset_dir": self.dataset_dir,
            "camera_angle_x": self.camera_angle_x,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "FrameData":
        return cls(**data)

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

        self.pointcloud = frames[0].pointcloud
        # Aggregate point clouds
        for frame in frames[1:]:
            self.pointcloud = self.pointcloud + frame.pointcloud

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
            "frames": [frame.as_dict() for frame in self.frames],
            "transform_matrix": self.transform_matrix.tolist(),
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ParentFrame":
        frames = [FrameData(**frame_data) for frame_data in data["frames"]]
        parent_frame = cls(frames)
        parent_frame.transform_matrix = np.array(data["transform_matrix"])
        return parent_frame


class PairwiseRegistration:
    def __init__(self, source: ParentFrame, target: ParentFrame, transform_matrix: np.ndarray, edge_type: Literal["loop", "odometry"], iteration_data: List[Dict]):
        self.source = source
        self.target = target
        self.transform_matrix = transform_matrix
        self.edge_type = edge_type
        self.iteration_data = iteration_data

    def as_dict(self) -> Dict:
        iteration_data = self.iteration_data
        for data in iteration_data:
            data["transformation"] = data["transformation"].cpu().numpy().tolist()
            data["inlier_rmse"] = data["inlier_rmse"].cpu().item()
            data["scale_index"] = data["scale_index"].cpu().item();
            data["fitness"] = data["fitness"].cpu().item()
            data["scale_iteration_index"] = data["scale_iteration_index"].cpu().item()
            data["iteration_index"] = data["iteration_index"].cpu().item();


        return {
            "source": self.source.as_dict(),
            "target": self.target.as_dict(),
            "transform_matrix": self.transform_matrix.tolist(),
            "type": self.edge_type,
            "iteration_data": iteration_data,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "PairwiseRegistration":
        source = ParentFrame.from_dict(data["source"])
        target = ParentFrame.from_dict(data["target"])
        transform_matrix = np.array(data["transform_matrix"])
        edge_type = data["type"]
        iteration_data = data["iteration_data"]
        for data in iteration_data:
            data["transformation"] = np.array(data["transformation"])

        return cls(source, target, transform_matrix, edge_type, iteration_data)


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


def run_pairwise_icp(dataset_dir: str):
    transforms_train = os.path.join(dataset_dir, 'transforms_train.json')
    with open(transforms_train, 'r') as f:
        train_json = json.load(f)

    frame_data = load_frame_data_from_dataset(dataset_dir, transforms_train)
    frame = frame_data[0]
    registered_frames = [frame]
    rolling_init_transform = np.eye(4)
    big_boy_pointcloud = frame.pointcloud.as_open3d_tensor()
    big_boy_pointcloud.estimate_normals(max_nn=30, radius=0.2)

    for i, frame in enumerate(tqdm(frame_data)):
        frame_pcd = frame.pointcloud.as_open3d_tensor().transform(rolling_init_transform)
        bounding_box = frame_pcd.get_axis_aligned_bounding_box()
        bounding_box.scale(2, center=bounding_box.get_center())
        merged_pointcloud = big_boy_pointcloud.crop(bounding_box)
        estimation = treg.TransformationEstimationPointToPlane()
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

        max_correspondence_distances = o3d.utility.DoubleVector([1.0, 0.4, 0.09, 0.04, 0.01])
        merged_pcd = merged_pointcloud
        registration_icp = treg.multi_scale_icp(frame_pcd,
                                  merged_pcd,
                                  voxel_sizes,
                                  criteria_list,
                                  max_correspondence_distances,
                                  np.eye(4),
                                  estimation)


        np_transform = registration_icp.transformation.numpy()
        rolling_init_transform = np_transform @ rolling_init_transform
        frame.transform_(np_transform)

        if i % 20 == 0:
            registered_frames.append(frame)
            p = frame_pcd.transform(registration_icp.transformation)
            p.estimate_normals(max_nn=30, radius=0.2)
            big_boy_pointcloud += p

    transforms_train_shifted = os.path.join(dataset_dir, 'transforms_train_original_shifted.json')

    with open(transforms_train_shifted, 'w') as f:
        new_frames = []
        for frame, registered_frame in zip(train_json['frames'], registered_frames):
            matrix = registered_frame.transform_matrix
            frame['transform_matrix'] = matrix.tolist()
            new_frames.append(frame)
        json.dump(train_json, f, indent=4)

def pairwise_registration(source, target, trans_init, max_correspondence_distance, voxel_size=0.01):
    iteration_data = []
    def iteration_callback(data: Dict):
        iteration_data.append(data)

    criteria_list = [
            treg.ICPConvergenceCriteria(relative_fitness=0.001,
                                        relative_rmse=0.001,
                                        max_iteration=50),
            treg.ICPConvergenceCriteria(0.0001, 0.0001, 50),
            treg.ICPConvergenceCriteria(0.00001, 0.00001, 30),
            treg.ICPConvergenceCriteria(0.000001, 0.000001, 20),
            treg.ICPConvergenceCriteria(0.000001, 0.000001, 10)
        ]
    voxel_sizes = o3d.utility.DoubleVector([0.1, 0.07, 0.03, 0.008, 0.003])
    max_correspondence_distances = o3d.utility.DoubleVector([0.3, 0.1, 0.07, 0.024, 0.01])

    mu, sigma = 0, 0.5  # mean and standard deviation
    estimation = treg.TransformationEstimationPointToPlane(
        treg.robust_kernel.RobustKernel(
        treg.robust_kernel.RobustKernelMethod.TukeyLoss, sigma))

    try:
        registration_icp = treg.multi_scale_icp(source,
                                target,
                                voxel_sizes,
                                criteria_list,
                                max_correspondence_distances,
                                trans_init,
                                estimation, iteration_callback)
    except RuntimeError as e:
        print(e)
        return None, None, None

    if registration_icp.fitness == 0 and registration_icp.inlier_rmse == 0:
        # no correspondence
        return None, None, None
    transformation_icp = registration_icp.transformation.numpy()
    try:
        information_icp = treg.get_information_matrix(source,
                                                      target,
                                                      max_correspondence_distances[-1],
                                                      transformation_icp).numpy()
    except RuntimeError as e:
        information_icp = np.eye(6)
    return transformation_icp, information_icp, iteration_data


def has_aabb_one_dimensional_overlap(segment1: np.ndarray, segment2: np.ndarray) -> bool:
    # Segement: [2] (min, max)
    return segment1[1] >= segment2[0] and segment2[1] >= segment1[0]

def should_add_edge_intersection(pcd1: o3d.t.geometry.PointCloud,
                                 pcd2: o3d.t.geometry.PointCloud,
                                 scale: float = 0.3,
                                 iou_threshold = 0.99) -> bool:
    # # Downscaled AABB
    # bounding_box1 = pcd1.get_axis_aligned_bounding_box()
    # bounding_box2 = pcd2.get_axis_aligned_bounding_box()
    # bounding_box1.scale(scale, center=bounding_box1.get_center())
    # bounding_box2.scale(scale, center=bounding_box2.get_center())

    # # Shape: [3, 2]
    # print(bounding_box1.get_min_bound())
    # min_max_extents1 = np.stack([bounding_box1.get_min_bound().numpy(),
    #                              bounding_box1.get_max_bound().numpy()], axis=-1)
    # # Shape: [3, 2]
    # min_max_extents2 = np.stack([bounding_box2.get_min_bound().numpy(),
    #                              bounding_box2.get_max_bound().numpy()], axis=-1)

    # has_overlap = True
    # for i in range(3):
    #     has_overlap = has_overlap and (has_aabb_one_dimensional_overlap(min_max_extents1[i],
    #                                                                     min_max_extents2[i]))
    # return has_overlap


    bounding_box1 = pcd1.get_axis_aligned_bounding_box()
    bounding_box2 = pcd2.get_axis_aligned_bounding_box()
    # min_max_1 = np.stack([bounding_box1.get_min_bound().cpu().numpy(), bounding_box1.get_max_bound().cpu().numpy()], axis=0)  # Shape: (2, 3)
    # min_max_2 = np.stack([bounding_box2.get_min_bound().cpu().numpy(), bounding_box2.get_max_bound().cpu().numpy()], axis=0)  # Shape: (2, 3)

    # iou, aabb1_intersection_ratio, aabb2_intersection_ratio = aabb_intersection_ratios(min_max_1, min_max_2)
    iou, aabb1_intersection_ratio, aabb2_intersection_ratio = aabb_intersection_ratios_open3d(bounding_box1, bounding_box2)

    loop_should_be_closed = iou > iou_threshold or aabb1_intersection_ratio > iou_threshold or aabb2_intersection_ratio > iou_threshold

    # if loop_should_be_closed:
    #     #debugging
    #     print(iou, aabb1_intersection_ratio, aabb2_intersection_ratio)

    return loop_should_be_closed


def unpack_parent_frame(parent_frame: ParentFrame) -> List[np.ndarray]:
    return parent_frame.get_all_frame_transforms()

def cluster_frame_data(frames: List[FrameData],
                       frames_per_cluster: int = 2) -> List[ParentFrame]:
    return [ParentFrame(frames[i:i + frames_per_cluster]) for i in range(0, len(frames), frames_per_cluster)]


def run_full_icp(dataset_dir: str,
                 max_correspondence_distance: float = 0.4,
                 pose_graph_optimization_iterations: int = 300,
                 forward_frame_step_size: int = 1,
                 no_loop_closure_within_frames: int = 12,
                 frames_per_cluster: int = 1,
                 thread_pool_size: int = 8) -> None:
    transforms_train = os.path.join(dataset_dir,
                                    'transforms_train.json')
    with open(transforms_train, 'r') as f:
        train_json = json.load(f)

    frame_data = load_frame_data_from_dataset(dataset_dir, transforms_train)
    frame_data = cluster_frame_data(frame_data, frames_per_cluster=frames_per_cluster)
    pcds = frame_data
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
    n_pcds = len(pcds)
    n_pcds = 400 # debug
    print('Building pose graph ...')
    pairwise_registrations = []

    for source_id in tqdm(range(n_pcds)):
        source_pcd = pcds[source_id].pointcloud.as_open3d_tensor(estimate_normals=True, device=device)
        # source_trans_inv = invert_transformation_matrix(pcds[source_id].transform_matrix)
        last_loop_closure = source_id
        for target_id in range(source_id+1, n_pcds, forward_frame_step_size):
            target_pcd = pcds[target_id].pointcloud.as_open3d_tensor(estimate_normals=True, device=device)
            if not (target_id == source_id + 1 or (target_id >= source_id + no_loop_closure_within_frames and target_id >= last_loop_closure + no_loop_closure_within_frames and should_add_edge_intersection(source_pcd, target_pcd))):
                continue
            target_trans = pcds[target_id].transform_matrix
            # trans_init = target_trans @ source_trans_inv
            trans_init = np.eye(4)
            transformation_icp, information_icp, iteration_data = pairwise_registration(source_pcd,
                                                                        target_pcd,
                                                                        o3d.core.Tensor(trans_init),
                                                                        max_correspondence_distance)
            if target_id == source_id + 1:  # odometry case
                if transformation_icp is None or information_icp is None:
                    # no correspondence found
                    raise ValueError("no transformation found for odometry case")
                odometry = np.dot(transformation_icp, odometry)
                pose_graph.nodes.append(
                    o3d.pipelines.registration.PoseGraphNode(
                        invert_transformation_matrix(odometry)))
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                             target_id,
                                                             transformation_icp,
                                                             information_icp,
                                                             uncertain=False))

                registration_res = PairwiseRegistration(pcds[source_id], pcds[target_id], transformation_icp, "odometry", iteration_data)
                pairwise_registrations.append(registration_res)
            else:  # loop closure case
                if transformation_icp is None or information_icp is None:
                    continue
                # print(f"closing the loop for frames {source_id} and {target_id}")
                last_loop_closure = target_id
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                             target_id,
                                                             transformation_icp,
                                                             information_icp,
                                                             uncertain=True))

                registration_res = PairwiseRegistration(pcds[source_id], pcds[target_id], transformation_icp, "loop", iteration_data)
                pairwise_registrations.append(registration_res)

    pairwise_registrations = list(map(lambda x: x.as_dict(), pairwise_registrations))
    with open(os.path.join(dataset_dir, 'pairwise_registrations.json'), 'w') as f:
        json.dump(pairwise_registrations, f, indent=4)

    print("Optimizing pose graph ...")
    option = o3d.pipelines.registration.GlobalOptimizationOption(
            max_correspondence_distance=max_correspondence_distance,
            edge_prune_threshold=0.25,
            reference_node=0)
    criteria = o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria()
    criteria.max_iteration = pose_graph_optimization_iterations
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        o3d.pipelines.registration.global_optimization(
            pose_graph,
            o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
            criteria,
            option)


    # for parent_frame, node in zip(frame_data, pose_graph.nodes):
    for i in range(n_pcds):
        parent_frame = frame_data[i]
        node = pose_graph.nodes[i]
        parent_frame.transform_matrix = node.pose @ parent_frame.transform_matrix

    print("Optimizing local transformations...")
    # Get sequential transforms in parallel
    # with Pool(thread_pool_size) as p:
    new_transforms = list(tqdm(map(unpack_parent_frame, frame_data), total=len(frame_data)))
    new_transforms = itertools.chain(*new_transforms)

    print("Writing results ...")
    transforms_train_shifted = os.path.join(dataset_dir, 'transforms_train_original_shifted.json')
    with open(transforms_train_shifted, 'w') as f:
        for i, (transform, json_frame) in enumerate(zip(new_transforms, train_json['frames'])):
            json_frame['transform_matrix'] = transform.tolist()

        train_json['frames'] = train_json['frames'][:n_pcds]  # debug
        json.dump(train_json, f, indent=4)

def get_transformation_mat(source, target, steps=200):
    R = torch.nn.Parameter(torch.eye(3))
    t = torch.nn.Parameter(torch.zeros(3))
    loss = torch.nn.MSELoss()
    source = torch.Tensor(source)
    target = torch.Tensor(target)
    optimizer = torch.optim.Adam([R, t], lr=0.001)
    for _ in range(steps):
        optimizer.zero_grad()
        transformed_source = R @ torch.transpose(source, 0, 1) + t[..., None]
        loss_val = loss(torch.transpose(transformed_source, 0, 1), target)
        loss_val.backward()
        optimizer.step()
    print('loss is', loss_val.item())
    T = np.concatenate([R.data.numpy(),
                        t.data.numpy()[..., None]], axis=-1)
    T = np.concatenate([T, np.array([[0., 0., 0., 1.]])], axis=0)
    return T

def run_sift(dataset_dir: str,
             top_match_limit: int = 200) -> None:
    n_pcds = 100
    transforms_train = os.path.join(dataset_dir,
                                    'transforms_train.json')
    with open(transforms_train, 'r') as f:
        train_json = json.load(f)

    frame_data = load_frame_data_from_dataset(dataset_dir, transforms_train)
    new_transforms = [frame_data[0].transform_matrix]
    orb = cv2.SIFT_create()
    kp = orb.detect(frame_data[0].rgb, None)
    kp, des = orb.compute(frame_data[0].rgb, kp)
    descriptors = [(kp, des)]
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    odometry = new_transforms[0]

    for i in tqdm(range(1, n_pcds)):
        frame = frame_data[i]
        rgb_image = frame.rgb

        # Get and update ORB descriptors
        kp = orb.detect(rgb_image, None)
        kp, des = orb.compute(rgb_image, kp)
        past_kp, past_des = descriptors[-1]
        descriptors.append((past_kp, past_des))
        matches = matcher.match(des, past_des)
        matches = sorted(matches, key = lambda x: x.distance)[:top_match_limit]
        source_points, target_points = [], []
        source_pcd_points = np.reshape(frame_data[i - 1].pointcloud.as_numpy()[0], rgb_image.shape)
        target_pcd_points = np.reshape(frame.pointcloud.as_numpy()[0], rgb_image.shape)
        """
        cvimg = cv2.drawMatches(rgb_image,
                                kp,
                                frame_data[i - 1].rgb,
                                past_kp,
                                matches[:10],
                                None,
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        """
        for match in matches:
            source_point = past_kp[match.trainIdx].pt
            target_point = kp[match.queryIdx].pt
            source_points.append(source_pcd_points[int(source_point[1]),
                                                   int(source_point[0])])
            target_points.append(target_pcd_points[int(target_point[1]),
                                                   int(target_point[0])])
        source_points = np.stack(source_points, axis=0)
        target_points = np.stack(target_points, axis=0)
        T = get_transformation_mat(source_points, target_points)
        odometry = T @ odometry
        new_transforms.append(odometry)


    print("Writing results ...")
    transforms_train_shifted = os.path.join(dataset_dir, 'transforms_train_original_shifted.json')
    with open(transforms_train_shifted, 'w') as f:
        for i, (transform, json_frame) in enumerate(zip(new_transforms, train_json['frames'])):
            json_frame['transform_matrix'] = transform.tolist()

        train_json['frames'] = train_json['frames'][:n_pcds]  # debug
        json.dump(train_json, f, indent=4)


def main(dataset_dir: str):
    run_full_icp(dataset_dir)

if __name__ == "__main__":
    Fire(main)
