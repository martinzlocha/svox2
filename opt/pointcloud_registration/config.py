from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Literal
import open3d as o3d
if o3d.__DEVICE_API__ == 'cuda':
    import open3d.cuda.pybind.t.pipelines.registration as treg
    device = o3d.core.Device("CUDA:0")
else:
    import open3d.cpu.pybind.t.pipelines.registration as treg
    device = o3d.core.Device("CPU:0")
import json


# FRAME CLUSTERING CONFIG
@dataclass
class FrameClusteringConfig:
    frames_per_cluster: int

    @classmethod
    def from_dict(cls, config_data: Dict) -> "FrameClusteringConfig":
        return cls(**config_data)

# EDGE CANDIDATES CONFIG
@dataclass
class EdgeCandidatesConfig:
    use_covisibility: bool
    use_iou: bool
    iou_threshold: float

    @classmethod
    def from_dict(cls, config_data: Dict) -> "EdgeCandidatesConfig":
        return cls(**config_data)

# REGISTRATION CONFIG
class EstimationKernel(ABC):
    @abstractmethod
    def get_kernel_args(self) -> List[Any]:
        pass

@dataclass
class RobustKernel(EstimationKernel):
    loss: Literal["tukey"]  # TODO: add more losses
    sigma: float

    def get_kernel_args(self):
        if self.loss == "tukey":
            return [treg.robust_kernel.RobustKernel(treg.robust_kernel.RobustKernelMethod.TukeyLoss, self.sigma)]
        else:
            raise ValueError(f"Unknown loss function {self.loss}")

class DefaultKernel(EstimationKernel):
    def get_kernel_args(self):
        return []

@dataclass
class Estimation:
    method: Literal["point_to_point", "point_to_plane", "colored"]
    kernel: EstimationKernel

    def get_method(self):
        kernel_args = self.kernel.get_kernel_args()
        if self.method == "point_to_point":
            return treg.TransformationEstimationPointToPoint(*kernel_args)
        elif self.method == "point_to_plane":
            return treg.TransformationEstimationPointToPlane(*kernel_args)
        elif self.method == "colored":
            return treg.TransformationEstimationForColoredICP(*kernel_args)
        else:
            raise ValueError(f"Unknown estimation method {self.method}")

    @classmethod
    def from_dict(cls, config_dict):
        if config_dict["kernel"]["type"] == "default":
            kernel = DefaultKernel()
        elif config_dict["kernel"]["type"] == "robust":
            kernel = RobustKernel(**config_dict["kernel"]["args"])
        else:
            raise ValueError(f"Unknown kernel type {config_dict['kernel']['type']}")
        return cls(config_dict["method"], kernel)

@dataclass
class ConvergenceCriterium:
    relative_fitness: float
    relative_rmse: float
    max_iteration: int

    def get_criterium(self):
        return treg.ICPConvergenceCriteria(self.relative_fitness, self.relative_rmse, self.max_iteration)

@dataclass
class ConvergenceCriteria:
    criteria: List[ConvergenceCriterium]

    def get_criteria(self):
        return [criterium.get_criterium() for criterium in self.criteria]

    @classmethod
    def from_dict(cls, config_dict):
        return cls([ConvergenceCriterium(**criterium) for criterium in config_dict])

@dataclass
class DoubleVector:
    values: List[float]

    def get_o3d_vector(self):
        return o3d.utility.DoubleVector(self.values)

@dataclass
class RegistrationConfig:
    register_odometry_edges: bool
    estimation: Estimation
    convergence_criteria: ConvergenceCriteria
    voxel_sizes: DoubleVector
    max_correspondence_distance: DoubleVector

    @classmethod
    def from_dict(cls, config_data: Dict) -> "RegistrationConfig":
        return cls(
            config_data["register_odometry_edges"],
            Estimation.from_dict(config_data["estimation"]),
            ConvergenceCriteria.from_dict(config_data["convergence_criteria"]),
            DoubleVector(config_data["voxel_sizes"]),
            DoubleVector(config_data["max_correspondence_distance"]),
        )

    def last_correspondence_distance(self):
        return self.max_correspondence_distance.values[-1]

# GLOBAL OPTIMIZATION CONFIG
@dataclass
class GlobalOptimizationConfig:
    pass

# PIPELINE CONFIG
@dataclass
class PipelineConfig:
    frame_clustering: FrameClusteringConfig
    use_only_max_confidence_pointcloud: bool
    edge_candidates: EdgeCandidatesConfig
    registration: RegistrationConfig
    global_optimization: GlobalOptimizationConfig

    @classmethod
    def from_dict(cls, config_data: Dict) -> "PipelineConfig":
        return cls(
            FrameClusteringConfig.from_dict(config_data["frame_clustering"]),
            config_data["use_only_max_confidence_pointcloud"],
            EdgeCandidatesConfig.from_dict(config_data["edge_candidates"]),
            RegistrationConfig.from_dict(config_data["registration"]),
            GlobalOptimizationConfig.from_dict(config_data["global_optimization"]),
        )
