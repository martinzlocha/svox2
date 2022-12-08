from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Literal
import open3d as o3d
if o3d.__DEVICE_API__ == 'cuda':
    import open3d.cuda.pybind.t.pipelines.registration as treg
    device = o3d.core.Device("CUDA:0")
else:
    import open3d.cpu.pybind.t.pipelines.registration as treg
    device = o3d.core.Device("CPU:0")
import json


@dataclass
class FrameClusteringConfig:
    pass

@dataclass
class EdgeCandidatesConfig:
    pass

@dataclass
class RegistrationConfig:
    pass

@dataclass
class PipelineConfig:
    frame_clustering: FrameClusteringConfig
    use_covisibility: bool
    use_only_max_confidence_pointcloud: bool
    edge_candidates: EdgeCandidatesConfig
    registration: RegistrationConfig
