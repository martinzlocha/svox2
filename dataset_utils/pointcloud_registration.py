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
from typing import Dict, List

import imageio.v2 as imageio
import numpy as np
from fire import Fire
from tqdm import tqdm
from utils import (depth_file_path_from_frame,
                                 img_file_path_from_frame, load_depth_file)
from point_cloud import Pointcloud



class FrameData:
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


def load_frame_data_from_dataset(dataset_dir: str,
                                 transforms_json_file: str) -> List[FrameData]:
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


def run_pairwise_icp(dataset_dir: str):
    transforms_train = os.path.join(dataset_dir, 'transforms_train_original.json')
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

def pairwise_registration(source, target, trans_init, max_correspondence_distance=0.4):
    registration_icp = treg.icp(
                source,
                target,
                max_correspondence_distance=max_correspondence_distance,
                init_source_to_target=trans_init,
                estimation_method=treg.TransformationEstimationPointToPoint(),
                criteria=treg.ICPConvergenceCriteria(relative_fitness=0.001,
                                        relative_rmse=0.001,
                                        max_iteration=30),
                voxel_size=0.4)
    transformation_icp = registration_icp.transformation.numpy()

    try:
        information_icp = treg.get_information_matrix(source,
                                                    target,
                                                    max_correspondence_distance,
                                                    transformation_icp).numpy()
    except RuntimeError as e:
        information_icp = np.eye(6)
    return transformation_icp, information_icp


def invert_transformation_matrix(matrix):
    R = matrix[:3, :3]
    t = matrix[:3, 3:]
    inverted_matrix = np.concatenate([R.T, -R.T @ t], axis=-1)
    inverted_matrix = np.concatenate([inverted_matrix,
                                 np.array([[0., 0., 0., 1.]])],
                                 axis=0)
    return inverted_matrix


def run_full_icp(dataset_dir: str,
                 loop_closing_skip_size: int = 4,
                 max_correspondence_distance: float = 0.4) -> None:
    transforms_train = os.path.join(dataset_dir,
                                    'transforms_train_original.json')
    with open(transforms_train, 'r') as f:
        train_json = json.load(f)

    frame_data = load_frame_data_from_dataset(dataset_dir, transforms_train)
    pcds = frame_data
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
    n_pcds = len(pcds)
    num_points = 20
    print('Building pose graph ...')
    for source_id in tqdm(range(num_points)):
        source_pcd = pcds[source_id].pointcloud.as_open3d_tensor().transform(odometry)
        source_trans_inv = invert_transformation_matrix(pcds[source_id].transform_matrix)
        targets = [(source_id + 1) % num_points, (source_id + loop_closing_skip_size) % num_points]
        for target_id in targets:
            target_pcd = pcds[target_id].pointcloud.as_open3d_tensor().transform(odometry)
            target_trans = pcds[target_id].transform_matrix
            trans_init = target_trans @ source_trans_inv
            transformation_icp, information_icp = pairwise_registration(source_pcd,
                                                                        target_pcd,
                                                                        o3d.core.Tensor(trans_init),
                                                                        max_correspondence_distance)
            print(source_id, target_id)
            if target_id == source_id + 1:  # odometry case
                odometry = transformation_icp @ odometry
                pose_graph.nodes.append(
                    o3d.pipelines.registration.PoseGraphNode(
                        np.linalg.inv(odometry)))
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                             target_id,
                                                             transformation_icp,
                                                             information_icp,
                                                             uncertain=False))
            else:  # loop closure case
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                             target_id,
                                                             transformation_icp,
                                                             information_icp,
                                                             uncertain=True))
    print("Starting to optimize pose graph ...")
    option = o3d.pipelines.registration.GlobalOptimizationOption(
            max_correspondence_distance=max_correspondence_distance,
            edge_prune_threshold=0.25,
            reference_node=0)
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        o3d.pipelines.registration.global_optimization(
            pose_graph,
            o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
            o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
            option)
    
    new_frames = []
    print("Writing results ...")
    transforms_train_shifted = os.path.join(dataset_dir, 'transforms_train_original_shifted.json')
    with open(transforms_train_shifted, 'w') as f:
        for i, node in enumerate(pose_graph.nodes):
            new_frame = frame_data[i]
            new_frame.transform_matrix = node.pose
            new_frames.append(new_frame)
        json.dump(train_json, f, indent=4)


def main(dataset_dir: str):
    run_full_icp(dataset_dir)

if __name__ == "__main__":
    Fire(main)
