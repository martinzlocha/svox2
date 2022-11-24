import torch
from dataclasses import dataclass
import os
import cv2

def load_depth_file(fpath: str) -> torch.Tensor:
    os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

    img = cv2.imread(fpath, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    depth = img[:,:,2]

    return torch.from_numpy(depth)

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

def img_file_path_from_frame(frame: dict, potential_images_dir: str, dataset_dir: str) -> str:
    img_path = frame['file_path']

    extensions = ['.jpg', '.png', '']
    parent_dirs = [potential_images_dir, dataset_dir, '']

    for extension in extensions:
        for parent_dir in parent_dirs:
            fpath = os.path.join(parent_dir, img_path + extension)
            if os.path.exists(fpath):
                return fpath

    raise FileNotFoundError(f'Could not find image file for frame {frame["frame_id"]}')

def depth_file_path_from_frame(frame: dict, depth_dir: str, dataset_dir: str) -> str:
    if 'depth_path' in frame:
        return os.path.join(dataset_dir, frame['depth_path'])

    return os.path.join(depth_dir, f"{frame['image_id']:04d}.exr")