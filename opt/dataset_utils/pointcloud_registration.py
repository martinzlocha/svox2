from collections import namedtuple
from dataclasses import dataclass
import time
import open3d as o3d

if o3d.__DEVICE_API__ == 'cuda':
    import open3d.cuda.pybind.t.pipelines.registration as treg
    device = o3d.core.Device("CUDA:0")
else:
    import open3d.cpu.pybind.t.pipelines.registration as treg
    device = o3d.core.Device("CPU:0")
import json
import os
from typing import Any, Dict, List, Literal, Optional, Tuple
import itertools

import numpy as np
from fire import Fire
from tqdm import tqdm, trange
import torch
import cv2
from dataset_utils.aabb_iou import aabb_intersection_ratios_open3d
from dataset_utils.framedata import FrameData, ParentFrame, load_frame_data_from_dataset, invert_transformation_matrix


class PairwiseRegistrationLog:
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
    def from_dict(cls, data: Dict, frames: Dict[int, FrameData]) -> "PairwiseRegistrationLog":
        source = ParentFrame.from_dict(data["source"], frames)
        target = ParentFrame.from_dict(data["target"], frames)
        transform_matrix = np.array(data["transform_matrix"])
        edge_type = data["type"]
        iteration_data = data["iteration_data"]
        for data in iteration_data:
            data["transformation"] = np.array(data["transformation"])

        return cls(source, target, transform_matrix, edge_type, iteration_data)


@dataclass
class EdgeCandidate:
    source_id: int
    target_id: int
    edge_type: Literal["loop", "odometry"]


def pairwise_registration(source, target, trans_init):
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

def should_add_edge_intersection(pcd1: o3d.t.geometry.PointCloud,
                                 pcd2: o3d.t.geometry.PointCloud,
                                 scale: float = 0.3,
                                 iou_threshold = 0.99) -> bool:
    bounding_box1 = pcd1.get_axis_aligned_bounding_box()
    bounding_box2 = pcd2.get_axis_aligned_bounding_box()
    iou, aabb1_intersection_ratio, aabb2_intersection_ratio = aabb_intersection_ratios_open3d(bounding_box1, bounding_box2)
    loop_should_be_closed = iou > iou_threshold or aabb1_intersection_ratio > iou_threshold or aabb2_intersection_ratio > iou_threshold
    return loop_should_be_closed


def unpack_parent_frame(parent_frame: ParentFrame) -> List[np.ndarray]:
    return parent_frame.get_all_frame_transforms()

def cluster_frame_data(frames: List[FrameData],
                       frames_per_cluster: int = 2) -> List[ParentFrame]:
    return [ParentFrame(frames[i:i + frames_per_cluster]) for i in range(0, len(frames), frames_per_cluster)]


def _load_transforms_json(dataset_dir: str, json_file_name: str) -> Dict[str, Any]:
    transforms_json = os.path.join(dataset_dir, json_file_name)
    with open(transforms_json, 'r') as f:
        json_data = json.load(f)
    return json_data


def build_edge_candidates(fragment_data: List[ParentFrame], no_loop_closure_within_frames: int) -> List[EdgeCandidate]:
    edge_candidates = []

    # odometry candidates
    for source_id in trange(len(fragment_data) - 1, desc="Odometry candidates"):
        edge_candidates.append(EdgeCandidate(source_id, source_id+1, "odometry"))

    for source_id in trange(len(fragment_data) - 1, desc="Loop candidates"):
        last_loop_closure = source_id
        source_pcd = fragment_data[source_id].pointcloud.as_open3d_tensor(device=device)
        for target_id in range(source_id + no_loop_closure_within_frames, len(fragment_data)):
            if target_id < last_loop_closure + no_loop_closure_within_frames:
                continue

            target_pcd = fragment_data[target_id].pointcloud.as_open3d_tensor(device=device)
            if should_add_edge_intersection(source_pcd, target_pcd):
                edge_candidates.append(EdgeCandidate(source_id, target_id, "loop"))
                last_loop_closure = target_id

    return edge_candidates


@dataclass
class RegistrationResult:
    transformation: np.ndarray
    information: np.ndarray
    iteration_data: List[Dict[str, Any]]


def register_pointclouds(fragment_data: List[ParentFrame], edge_candidate: EdgeCandidate, init_transform: np.ndarray) -> Optional[RegistrationResult]:
    source_pcd = fragment_data[edge_candidate.source_id].pointcloud.as_open3d_tensor(estimate_normals=True, device=device)
    target_pcd = fragment_data[edge_candidate.target_id].pointcloud.as_open3d_tensor(estimate_normals=True, device=device)

    transformation_icp, information_icp, iteration_data = pairwise_registration(source_pcd, target_pcd, init_transform)
    if transformation_icp is None or information_icp is None or iteration_data is None:
        return None

    return RegistrationResult(transformation_icp, information_icp, iteration_data)


def register_candidates(fragment_data: List[ParentFrame], edge_candidates: List[EdgeCandidate]) -> Tuple[List[RegistrationResult], List[PairwiseRegistrationLog]]:
    # TODO: multi-thread this
    registration_results = []
    registration_logs = []
    for edge_candidate in tqdm(edge_candidates):
        registration_result = register_pointclouds(fragment_data, edge_candidate, np.eye(4))
        if registration_result is None and edge_candidate.edge_type == "odometry":
            raise RuntimeError("No correspondence found for the odometry case")

        registration_results.append(registration_result)
        if registration_result is not None:
            registration_logs.append(PairwiseRegistrationLog(
                fragment_data[edge_candidate.source_id],
                fragment_data[edge_candidate.target_id],
                registration_result.transformation,
                edge_candidate.edge_type,
                registration_result.iteration_data,
            ))

    return registration_results, registration_logs


def build_pose_graph(edge_candidates: List[EdgeCandidate], registration_results: List[RegistrationResult]) -> o3d.pipelines.registration.PoseGraph:
    if len(edge_candidates) != len(registration_results):
        raise RuntimeError("The number of edge candidates and registration results must be the same")

    # TODO: merge edge_candidates and registration_results into a single dataclass
    pose_graph = o3d.pipelines.registration.PoseGraph()

    # add first node
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))

    for registration_result, edge_candidate in zip(registration_results, tqdm(edge_candidates)):
        if registration_result is None:
            if edge_candidate.edge_type == "loop":
                continue
            else:
                raise RuntimeError("No correspondence found for the odometry case")

        if edge_candidate.edge_type == "odometry":
            odometry = np.dot(registration_result.transformation, odometry)
            pose_graph.nodes.append(
                    o3d.pipelines.registration.PoseGraphNode(
                        invert_transformation_matrix(odometry)))

        pose_graph.edges.append(
            o3d.pipelines.registration.PoseGraphEdge(edge_candidate.source_id, edge_candidate.target_id,
                                                     registration_result.transformation,
                                                     registration_result.information,
                                                     uncertain=edge_candidate.edge_type == "loop"))

    return pose_graph


def run_full_icp(dataset_dir: str,
                 max_correspondence_distance: float = 0.4,
                 pose_graph_optimization_iterations: int = 300,
                 no_loop_closure_within_frames: int = 12,
                 frames_per_cluster: int = 1) -> None:
    # CONFIG
    json_file_name = 'transforms_train.json'
    transforms_train = os.path.join(dataset_dir, json_file_name)
    max_fragments = 200 # debug, set to None to disable

    # ICP
    json_data = _load_transforms_json(dataset_dir, json_file_name)
    frame_data = load_frame_data_from_dataset(dataset_dir, transforms_train)
    print("clustering frames...")
    fragment_data = cluster_frame_data(frame_data, frames_per_cluster=frames_per_cluster)
    if max_fragments is not None:
        fragment_data = fragment_data[:max_fragments]  # debug

    print("precomputing pointclouds...")
    tic = time.time()
    for fragment in tqdm(fragment_data):
        # TODO: multi-thread this
        fragment.pointcloud = fragment.pointcloud.take_only_with_max_confidence()  # confidence pruning
        fragment.pointcloud.as_open3d_tensor(estimate_normals=True, device=device)  # subesquent calls will be cached
    toc = time.time()
    print(f'precomputing pointclouds took {toc-tic:.2f}s')

    # get edge candidates
    # time build_edge_candidates
    print("building edge candidates...")
    tic = time.time()
    edge_candidates = build_edge_candidates(fragment_data, no_loop_closure_within_frames)
    toc = time.time()
    print(f'build_edge_candidates took {toc-tic:.2f}s')


    print('Registering frames ...')
    tic = time.time()
    registration_results, registration_logs = register_candidates(fragment_data, edge_candidates)
    toc = time.time()
    print(f'register_candidates took {toc-tic:.2f}s')

    pairwise_registrations_logs = list(map(lambda x: x.as_dict(), registration_logs))
    registration_data = {
        "transforms_json_file": json_file_name,
        "pairwise_registrations": pairwise_registrations_logs,
    }
    with open(os.path.join(dataset_dir, 'pairwise_registrations.json'), 'w') as f:
        json.dump(registration_data, f, indent=4)

    del registration_data
    del pairwise_registrations_logs
    del registration_logs  # can easily take a couple of GBs of memory

    print('Building pose graph ...')
    tic = time.time()
    pose_graph = build_pose_graph(edge_candidates, registration_results)
    toc = time.time()
    print(f'build_pose_graph took {toc-tic:.2f}s')

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

    print("Done optimizing pose graph")

    # for source_id in tqdm(range(n_pcds)):
    #     source_pcd = pcds[source_id].pointcloud.as_open3d_tensor(estimate_normals=True, device=device)
    #     # source_trans_inv = invert_transformation_matrix(pcds[source_id].transform_matrix)
    #     last_loop_closure = source_id
    #     for target_id in range(source_id+1, n_pcds):
    #         target_pcd = pcds[target_id].pointcloud.as_open3d_tensor(estimate_normals=True, device=device)
    #         if not (target_id == source_id + 1 or (target_id >= source_id + no_loop_closure_within_frames and target_id >= last_loop_closure + no_loop_closure_within_frames and should_add_edge_intersection(source_pcd, target_pcd))):
    #             continue
    #         target_trans = pcds[target_id].transform_matrix
    #         # trans_init = target_trans @ source_trans_inv
    #         trans_init = np.eye(4)

    #         if target_id == source_id + 1:  # odometry case
    #             # if transformation_icp is None or information_icp is None:
    #             #     # no correspondence found
    #             #     raise ValueError("no transformation found for odometry case")
    #             transformation_icp = np.eye(4)
    #             information_icp = treg.get_information_matrix(source_pcd, target_pcd, 0.03, transformation_icp).numpy()
    #             # odometry = np.dot(transformation_icp, odometry)
    #             pose_graph.nodes.append(
    #                 o3d.pipelines.registration.PoseGraphNode(
    #                     invert_transformation_matrix(odometry)))
    #             # print(transformation_icp)
    #             # print(information_icp)
    #             pose_graph.edges.append(
    #                 o3d.pipelines.registration.PoseGraphEdge(source_id,
    #                                                          target_id,
    #                                                          transformation_icp,
    #                                                          information_icp,
    #                                                          uncertain=False))

    #             # registration_res = PairwiseRegistrationLog(pcds[source_id], pcds[target_id], transformation_icp, "odometry", iteration_data)
    #             # pairwise_registrations.append(registration_res)
    #         else:  # loop closure case
    #             transformation_icp, information_icp, iteration_data = pairwise_registration(source_pcd,
    #                                                                     target_pcd,
    #                                                                     o3d.core.Tensor(trans_init),
    #                                                                     max_correspondence_distance)
    #             if transformation_icp is None or information_icp is None:
    #                 continue
    #             print(f"closing the loop for frames {source_id} and {target_id}")
    #             last_loop_closure = target_id
    #             pose_graph.edges.append(
    #                 o3d.pipelines.registration.PoseGraphEdge(source_id,
    #                                                          target_id,
    #                                                          transformation_icp,
    #                                                          information_icp,
    #                                                          uncertain=True))

    #             registration_res = PairwiseRegistrationLog(pcds[source_id], pcds[target_id], transformation_icp, "loop", iteration_data)
    #             pairwise_registrations.append(registration_res)




    # for parent_frame, node in zip(frame_data, pose_graph.nodes):
    for i in range(len(fragment_data)):
        parent_frame = fragment_data[i]
        node = pose_graph.nodes[i]
        parent_frame.transform_matrix = node.pose @ parent_frame.transform_matrix

    print("Optimizing local transformations...")
    # Get sequential transforms in parallel
    # with Pool(thread_pool_size) as p:
    new_transforms = list(tqdm(map(unpack_parent_frame, fragment_data), total=len(fragment_data)))
    new_transforms = itertools.chain(*new_transforms)

    print("Writing results ...")
    transforms_train_shifted = os.path.join(dataset_dir, 'transforms_train_original_shifted.json')
    with open(transforms_train_shifted, 'w') as f:
        for i, (transform, json_frame) in enumerate(zip(new_transforms, json_data['frames'])):
            json_frame['transform_matrix'] = transform.tolist()

        if max_fragments is not None:
            json_data['frames'] = json_data['frames'][:max_fragments]  # debug
        json.dump(json_data, f, indent=4)

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
