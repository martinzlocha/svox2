import torch
from dataclasses import dataclass
import os
import numpy as np
import liblzfse
import cv2

def load_depth_file(fpath: str) -> torch.Tensor:
    os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

    depth = cv2.imread(fpath, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if depth.ndim == 3:
        depth = depth[:,:,2]
    # Quick hack: If more than 100 the units are likely mm not m.
    if np.max(depth) > 100:
        depth = depth.astype(np.float32)
        depth /= 1000

    return torch.from_numpy(depth)

def load_confidence_file(fpath: str) -> torch.Tensor:
    if '.conf' in fpath:
        with open(fpath, 'rb') as confidence_fh:
            raw_bytes = confidence_fh.read()
            decompressed_bytes = liblzfse.decompress(raw_bytes)
            confidence_img = np.frombuffer(decompressed_bytes, dtype=np.uint8)
    else:
        confidence_img = cv2.imread(fpath, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    return torch.from_numpy(confidence_img).to(dtype=torch.uint8)

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

def img_file_path_from_frame(frame: dict, dataset_dir: str) -> str:
    return os.path.join(dataset_dir, frame['file_path'])

def depth_file_path_from_frame(frame: dict, dataset_dir: str) -> str:
    return os.path.join(dataset_dir, frame['depth_path'])

def confidence_file_path_from_frame(frame: dict, dataset_dir: str) -> str:
    return os.path.join(dataset_dir, frame['confidence_path'])