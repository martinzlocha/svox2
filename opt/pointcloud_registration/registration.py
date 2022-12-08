import multiprocessing
import time
from dataclasses import dataclass
from dataset_utils.point_cloud import Pointcloud

import open3d as o3d

from pointcloud_registration.config import (EdgeCandidatesConfig,
                                            FrameClusteringConfig,
                                            PipelineConfig, RegistrationConfig)

if o3d.__DEVICE_API__ == 'cuda':
    import open3d.cuda.pybind.t.pipelines.registration as treg  # type: ignore
    device = o3d.core.Device("CUDA:0")  # type: ignore
else:
    import open3d.cpu.pybind.t.pipelines.registration as treg  # type: ignore
    device = o3d.core.Device("CPU:0")  # type: ignore
import itertools
import json
import os
from typing import Any, Dict, List, Literal, Optional, Tuple

import cv2
import numpy as np
import torch
from dataset_utils.aabb_iou import aabb_intersection_ratios_open3d
from dataset_utils.framedata import (FrameData, ParentFrame,
                                     invert_transformation_matrix,
                                     load_frame_data_from_dataset)
from fire import Fire
from joblib import Parallel, delayed
from tqdm import tqdm, trange


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


def pairwise_registration(source, target, trans_init, cfg: RegistrationConfig):
    iteration_data = []
    def iteration_callback(data: Dict):
        iteration_data.append(data)

    # criteria_list = [
    #         treg.ICPConvergenceCriteria(relative_fitness=0.001,
    #                                     relative_rmse=0.001,
    #                                     max_iteration=50),
    #         treg.ICPConvergenceCriteria(0.0001, 0.0001, 50),
    #         treg.ICPConvergenceCriteria(0.00001, 0.00001, 30),
    #         treg.ICPConvergenceCriteria(0.000001, 0.000001, 20),
    #         treg.ICPConvergenceCriteria(0.000001, 0.000001, 10)
    #     ]

    criteria_list = cfg.convergence_criteria.get_criteria()
    # voxel_sizes = o3d.utility.DoubleVector([0.1, 0.07, 0.03, 0.008, 0.003])
    # max_correspondence_distances = o3d.utility.DoubleVector([0.3, 0.1, 0.07, 0.024, 0.01])
    voxel_sizes = cfg.voxel_sizes.get_o3d_vector()
    max_correspondence_distances = cfg.max_correspondence_distances.get_o3d_vector()


    # mu, sigma = 0, 0.5  # mean and standard deviation
    # estimation = treg.TransformationEstimationPointToPlane(
    #     treg.robust_kernel.RobustKernel(
    #     treg.robust_kernel.RobustKernelMethod.TukeyLoss, sigma))
    estimation = cfg.estimation.get_method()

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
        print(f"No correspondence due to low fitness. Fitness: {registration_icp.fitness}, RMSE: {registration_icp.inlier_rmse}")
        return None, None, None
    transformation_icp = registration_icp.transformation.numpy()
    try:
        information_icp = treg.get_information_matrix(source,
                                                      target,
                                                      max_correspondence_distances[-1],
                                                      transformation_icp).numpy()
    except RuntimeError as e:
        information_icp = np.eye(6)

    if information_icp[5, 5] / min(len(source.point), len(target.point)) < 0.3:
        # too few correspondences
        print("Too few correspondences")
        return None, None, None

    return transformation_icp, information_icp, iteration_data

def iou_overlaps(pcd1: Pointcloud,
                 pcd2: Pointcloud,
                 iou_threshold = 0.99) -> bool:
    bounding_box1 = pcd1.get_axis_aligned_bounding_box()
    bounding_box2 = pcd2.get_axis_aligned_bounding_box()
    iou, aabb1_intersection_ratio, aabb2_intersection_ratio = aabb_intersection_ratios_open3d(bounding_box1, bounding_box2)
    loop_should_be_closed = iou > iou_threshold or aabb1_intersection_ratio > iou_threshold or aabb2_intersection_ratio > iou_threshold
    return loop_should_be_closed


def unpack_parent_frame(parent_frame: ParentFrame) -> List[np.ndarray]:
    return parent_frame.get_all_frame_transforms()

def cluster_frame_data(frames: List[FrameData],
                       cfg: FrameClusteringConfig) -> List[ParentFrame]:
    return [ParentFrame(frames[i:i + cfg.frames_per_cluster]) for i in range(0, len(frames), cfg.frames_per_cluster)]


def _load_transforms_json(dataset_dir: str, json_file_name: str) -> Dict[str, Any]:
    transforms_json = os.path.join(dataset_dir, json_file_name)
    with open(transforms_json, 'r') as f:
        json_data = json.load(f)
    return json_data

def get_fragement_descriptor_distance(source: ParentFrame,
                                      target: ParentFrame,
                                      matcher,
                                      top_match_count: int = 10) -> bool:
    """Takes best top_match_count matches returns average distance."""
    matches = matcher.match(source.descriptors,
                            target.descriptors)
    if not matches:
        return -1.

    get_distance = lambda x: x.distance
    matches = sorted(matches, key = get_distance)[:top_match_count]
    average_dist = sum(map(get_distance, matches)) / min(len(matches), top_match_count)
    return average_dist

def fragments_covisible(source: ParentFrame,
                        target: ParentFrame,
                        matcher,
                        threshold: float,
                        top_match_count: int = 10) -> bool:
    distance = get_fragement_descriptor_distance(source,
                                                 target,
                                                 matcher,
                                                 top_match_count)
    return distance < threshold


def build_edge_candidates(fragment_data: List[ParentFrame],
                          cfg: EdgeCandidatesConfig) -> List[EdgeCandidate]:
    edge_candidates = []
    if cfg.use_covisibility:
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        distance_mean, distance_squared_mean = 0., 0.
        odometry_match_count = 0

    # odometry candidates
    for source_id in trange(len(fragment_data) - 1, desc="Odometry candidates"):
        edge_candidates.append(EdgeCandidate(source_id, source_id+1, "odometry"))
        if cfg.use_covisibility:
            descriptor_distance = get_fragement_descriptor_distance(fragment_data[source_id],
                                                                    fragment_data[source_id + 1],
                                                                    matcher)
            if descriptor_distance >= 0.:
                odometry_match_count += 1
                distance_mean += descriptor_distance
                distance_squared_mean += descriptor_distance**2

    if cfg.use_covisibility:
        distance_mean /= odometry_match_count
        distance_squared_mean /= odometry_match_count
        distance_sigma = np.sqrt(distance_squared_mean - distance_mean**2)
        match_threshold = distance_mean + distance_sigma

    for source_id in trange(len(fragment_data) - 1, desc="Loop candidates"):
        last_loop_closure = source_id
        # source_pcd = fragment_data[source_id].pointcloud.as_open3d_tensor(device=device)
        for target_id in range(source_id + cfg.no_loop_closure_within_frames, len(fragment_data)):
            if target_id < last_loop_closure + cfg.no_loop_closure_within_frames:
                # TODO: we might want to deprecate this
                continue

            should_add_edge = True
            if cfg.use_iou:
                # target_pcd = fragment_data[target_id].pointcloud.as_open3d_tensor(device=device)
                should_add_edge = should_add_edge and iou_overlaps(fragment_data[source_id].pointcloud, fragment_data[target_id].pointcloud, cfg.iou_threshold)

            if cfg.use_covisibility:
                should_add_edge = should_add_edge and fragments_covisible(fragment_data[source_id],
                                                                          fragment_data[target_id],
                                                                          matcher,
                                                                          match_threshold)
            if should_add_edge:
                edge_candidates.append(EdgeCandidate(source_id, target_id, "loop"))
                last_loop_closure = target_id

    return edge_candidates


@dataclass
class RegistrationResult:
    transformation: np.ndarray
    information: np.ndarray
    iteration_data: List[Dict[str, Any]]


def register_pointclouds(fragment_data: List[ParentFrame], edge_candidate: EdgeCandidate,
                         init_transform: np.ndarray, cfg: RegistrationConfig) -> Optional[RegistrationResult]:
    if not cfg.register_odometry_edges and edge_candidate.edge_type == "odometry":
        transformation_icp = np.eye(4)
        information_icp = treg.get_information_matrix(fragment_data[edge_candidate.source_id].pointcloud.as_open3d_tensor(device=device),
                                                        fragment_data[edge_candidate.target_id].pointcloud.as_open3d_tensor(device=device),
                                                        cfg.last_correspondence_distance(),
                                                        transformation_icp).numpy()
        registration_result = RegistrationResult(
            transformation_icp,
            information_icp,
            [],
        )
    else:
        source_pcd = fragment_data[edge_candidate.source_id].pointcloud.as_open3d_tensor(estimate_normals=True, device=device)
        target_pcd = fragment_data[edge_candidate.target_id].pointcloud.as_open3d_tensor(estimate_normals=True, device=device)

        transformation_icp, information_icp, iteration_data = None, None, None  # silencing linter
        for _ in range(10):
            # TODO add to config
            transformation_icp, information_icp, iteration_data = pairwise_registration(source_pcd, target_pcd, init_transform, cfg)
            if edge_candidate.edge_type == "loop" or (transformation_icp is not None and information_icp is not None and iteration_data is not None):
                break

        if transformation_icp is None or information_icp is None or iteration_data is None:
            return None

        registration_result = RegistrationResult(transformation_icp, information_icp, iteration_data)

    return registration_result

def register_pointclouds_bare(source_pcd: o3d.t.geometry.PointCloud, target_pcd: o3d.t.geometry.PointCloud,
                              init_transform: np.ndarray, edge_type: Literal["loop", "odometry"], cfg: RegistrationConfig) -> Optional[RegistrationResult]:
    if edge_type == "odometry":
        raise NotImplementedError("Bare odometry registration is not implemented yet")

    transformation_icp, information_icp, iteration_data = pairwise_registration(source_pcd, target_pcd, init_transform, cfg)
    if transformation_icp is None or information_icp is None or iteration_data is None:
            return None

    registration_result = RegistrationResult(transformation_icp, information_icp, iteration_data)

    return registration_result

def register_loop_candidates(fragment_data: List[ParentFrame],
                                      edge_candidates: List[EdgeCandidate],
                                      cfg: RegistrationConfig) -> Tuple[List[Optional[RegistrationResult]], List[PairwiseRegistrationLog]]:
    all_args = [
        [fragment_data[edge_candidate.source_id].pointcloud.as_open3d_tensor(estimate_normals=True, device=device),
        fragment_data[edge_candidate.target_id].pointcloud.as_open3d_tensor(estimate_normals=True, device=device),
        np.eye(4), "loop", cfg] for edge_candidate in edge_candidates
    ]

    if device == o3d.core.Device("CPU:0"):
        # if we are on CPU, we probably don't want to paralelise
        registration_results = [register_pointclouds_bare(*args) for args in tqdm(all_args, desc="loop")]
    else:
        n_cpus = multiprocessing.cpu_count()
        registration_results = Parallel(n_jobs=n_cpus, verbose=1)(
            delayed(register_pointclouds_bare)(*args) for args in all_args)

    assert registration_results is not None

    registration_logs = [PairwiseRegistrationLog(
        fragment_data[edge_candidate.source_id],
        fragment_data[edge_candidate.target_id],
        registration_result.transformation,
        edge_candidate.edge_type,
        registration_result.iteration_data)
        for edge_candidate, registration_result in zip(edge_candidates, registration_results) if registration_result is not None]

    return registration_results, registration_logs

def register_candidates(fragment_data: List[ParentFrame],
                        edge_candidates: List[EdgeCandidate],
                        cfg: RegistrationConfig) -> Tuple[List[Optional[RegistrationResult]], List[PairwiseRegistrationLog]]:
    # TODO: multi-thread this
    registration_results = []
    registration_logs = []
    odometry_init = np.eye(4)
    odometry_candidates = [edge_candidate for edge_candidate in edge_candidates if edge_candidate.edge_type == "odometry"]
    loop_candidates = [edge_candidate for edge_candidate in edge_candidates if edge_candidate.edge_type == "loop"]

    for edge_candidate in tqdm(odometry_candidates, desc="odometry"):
        registration_result = register_pointclouds(fragment_data, edge_candidate, odometry_init, cfg)

        if registration_result is None and edge_candidate.edge_type == "odometry":
                raise RuntimeError("No correspondence found for the odometry case")

        if registration_result is not None and edge_candidate.edge_type == "odometry" and cfg.rolling_odometry_init:
            odometry_init = registration_result.transformation

        registration_results.append(registration_result)
        if registration_result is not None:
            registration_logs.append(PairwiseRegistrationLog(
                fragment_data[edge_candidate.source_id],
                fragment_data[edge_candidate.target_id],
                registration_result.transformation,
                edge_candidate.edge_type,
                registration_result.iteration_data,
            ))

    loop_registration_results, loop_registration_logs = register_loop_candidates(fragment_data, loop_candidates, cfg)
    registration_results.extend(loop_registration_results)
    registration_logs.extend(loop_registration_logs)

    return registration_results, registration_logs


def build_pose_graph(edge_candidates: List[EdgeCandidate], registration_results: List[Optional[RegistrationResult]]) -> o3d.pipelines.registration.PoseGraph:
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
                 save_dir: str,
                 config: PipelineConfig) -> None:
    # CONFIG
    json_file_name = 'transforms_train.json'
    transforms_train = os.path.join(dataset_dir, json_file_name)
    max_fragments = 50 # debug, set to None to disable
    n_cpus = multiprocessing.cpu_count()

    # ICP
    json_data = _load_transforms_json(dataset_dir, json_file_name)
    if max_fragments is not None:
        # debug
        frame_data = load_frame_data_from_dataset(dataset_dir, transforms_train, max_frames=max_fragments * config.frame_clustering.frames_per_cluster)
    else:
        frame_data = load_frame_data_from_dataset(dataset_dir, transforms_train)
    print("clustering frames...")
    fragment_data = cluster_frame_data(frame_data, config.frame_clustering)

    if config.edge_candidates.use_covisibility:
        tic = time.time()
        def _precompute_sift(fragment: ParentFrame):
            sift = cv2.SIFT_create()
            fragment.compute_descriptors(sift)
        print("pre-computing SIFT features...")
        with Parallel(n_jobs=n_cpus, verbose=1, require='sharedmem') as parallel:
            # can we use tqdm?
            parallel(delayed(_precompute_sift)(fragment) for fragment in fragment_data)

        # for fragment in tqdm(fragment_data):
        #     # TODO: multi-thread this
        #     sift = cv2.SIFT_create()
        #     fragment.compute_descriptors(sift)

        del _precompute_sift
        toc = time.time()
        print(f"pre-computing SIFT features took {toc-tic:.2f}s")

    print("pre-computing pointclouds...")
    tic = time.time()
    def _precompute_pointcloud(fragment: ParentFrame):
        if config.use_only_max_confidence_pointcloud:
            fragment.pointcloud = fragment.pointcloud.take_only_with_max_confidence()  # confidence pruning
        fragment.pointcloud.as_open3d_tensor(estimate_normals=True, device=device)

    with Parallel(n_jobs=n_cpus//2, verbose=1, require='sharedmem') as parallel:
        # paralel execution will most probably help only on a machine with GPU
        # TODO: can we use tqdm?
        parallel(delayed(_precompute_pointcloud)(fragment) for fragment in fragment_data)
    toc = time.time()
    del _precompute_pointcloud
    print(f'pre-computing pointclouds took {toc-tic:.2f}s')

    # get edge candidates
    # time build_edge_candidates
    print("building edge candidates...")
    tic = time.time()
    edge_candidates = build_edge_candidates(fragment_data, config.edge_candidates)
    toc = time.time()
    print(f'build_edge_candidates took {toc-tic:.2f}s')


    print('Registering frames ...')
    tic = time.time()
    registration_results, registration_logs = register_candidates(fragment_data, edge_candidates, config.registration)
    toc = time.time()
    print(f'register_candidates took {toc-tic:.2f}s')

    pairwise_registrations_logs = list(map(lambda x: x.as_dict(), registration_logs))
    registration_data = {
        "transforms_json_file": json_file_name,
        "pairwise_registrations": pairwise_registrations_logs,
    }
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'pairwise_registrations.json'), 'w') as f:
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
            max_correspondence_distance=config.registration.last_correspondence_distance(),
            edge_prune_threshold=0.25,
            reference_node=0)
    criteria = o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria()
    criteria.max_iteration = config.global_optimization.max_iterations
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        o3d.pipelines.registration.global_optimization(
            pose_graph,
            o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
            criteria,
            option)

    print("Done optimizing pose graph")

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
    transforms_train_shifted = os.path.join(save_dir, json_file_name)
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
    n_pcds = 200
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


def main(dataset_dir: str, save_dir: str, config_path):
    config = PipelineConfig.from_dict(json.load(open(config_path, 'r')))
    run_full_icp(dataset_dir, save_dir, config)

if __name__ == "__main__":
    Fire(main)
