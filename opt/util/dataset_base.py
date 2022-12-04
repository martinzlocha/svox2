import torch
import torch.nn.functional as F
from typing import Union, Optional, List
from .util import Rays, Intrin

class DatasetBase:
    split: str
    permutation: bool
    epoch_size: Optional[int]
    n_images: int
    h_full: int
    w_full: int
    intrins_full: Intrin
    c2w: torch.Tensor  # C2W OpenCV poses
    gt: Union[torch.Tensor, List[torch.Tensor]]   # RGB images
    depths: Union[torch.Tensor, List[torch.Tensor]]
    confidences: Union[torch.Tensor, List[torch.Tensor]]
    device : Union[str, torch.device]

    def __init__(self):
        self.ndc_coeffs = (-1, -1)
        self.use_sphere_bound = False
        self.should_use_background = True # a hint
        self.use_sphere_bound = True
        self.scene_center = [0.0, 0.0, 0.0]
        self.scene_radius = [1.0, 1.0, 1.0]
        self.permutation = False

    def get_ray_subset(self, indices):
        frame_size = self.h_full * self.w_full

        frames = indices // (frame_size)
        frame_offset = indices % frame_size

        xx = frame_offset % w
        yy = frame_offset // w
        zz = -torch.ones_like(xx)

        dirs = torch.stack((xx, yy, zz), dim=-1)  # OpenCV convention
        dirs_norm = torch.norm(dirs, dim=-1)
        dirs /= torch.norm(dirs, dim=-1, keepdim=True)
        dirs = (self.c2w[frames, :3, :3] @ dirs)[..., 0]
        origins = self.c2w[frames, :3, 3]

        gt = self.gt[frames, xx, yy, :]
        depths = self.depths[frames, xx, yy]
        depths /= dirs_norm
        if hasattr(self, 'confidences'):
            confidences = self.confidences[frames, xx, yy]
            depths[confidences != 2] = 0

        return Rays(origins=origins, dirs=dirs, gt=gt, depths=depths)

    def get_image_size(self, i : int):
        # H, W
        if hasattr(self, 'image_size'):
            return tuple(self.image_size[i])
        else:
            return self.h, self.w

    def get_number_of_rays(self):
        return self.gt.size(0) * self.gt.size(1) * self.gt.size(2)