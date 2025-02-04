# Standard NeRF Blender dataset loader
from .util import Rays, Intrin
from .dataset_base import DatasetBase
import torch
import liblzfse
import torch.nn.functional as F
from typing import Optional, Union
from os import path
from functools import partial
import imageio
from tqdm import tqdm
import cv2
import json
import numpy as np
import concurrent.futures
import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"


def load_image(fpath, scale=1):
    im_gt = imageio.imread(fpath)

    if scale < 1.0:
        full_size = list(im_gt.shape[:2])
        rsz_h, rsz_w = [round(hw * scale) for hw in full_size]
        im_gt = cv2.resize(im_gt, (rsz_w, rsz_h), interpolation=cv2.INTER_AREA)

    return torch.from_numpy(im_gt).to(dtype=torch.uint8)

def load_depth_file(fpath, width, height) -> torch.Tensor:
    depth = cv2.imread(fpath, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if depth.ndim == 3: 
        depth = depth[:,:,2]
    # Quick hack: If more than 100 the units are likely mm not m.
    if np.max(depth) > 100:
        depth = depth.astype(np.float32)
        depth /= 1000
    depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_NEAREST)

    return torch.from_numpy(depth).to(dtype=torch.float16)

def load_confidence_file(fpath, width, height) -> torch.Tensor:
    if '.conf' in fpath:
        with open(fpath, 'rb') as confidence_fh:
            raw_bytes = confidence_fh.read()
            decompressed_bytes = liblzfse.decompress(raw_bytes)
            confidence_img = np.frombuffer(decompressed_bytes, dtype=np.uint8)
            confidence_img = confidence_img.reshape((256, 192))
            confidence_img = cv2.resize(confidence_img, (width, height), interpolation=cv2.INTER_NEAREST)
    else:
        confidence_img = cv2.imread(fpath, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        confidence_img = cv2.resize(confidence_img, (width, height), interpolation=cv2.INTER_NEAREST)
    return torch.from_numpy(confidence_img).to(dtype=torch.uint8)


class NeRFDataset(DatasetBase):
    """
    NeRF dataset loader
    """

    focal: float
    c2w: torch.Tensor  # (n_images, 4, 4)
    gt: torch.Tensor  # (n_images, h, w, 3)
    h: int
    w: int
    n_images: int
    rays: Optional[Rays]
    split: str

    def __init__(
        self,
        root,
        split,
        epoch_size : Optional[int] = None,
        device: Union[str, torch.device] = "cpu",
        scene_scale: Optional[float] = None,
        scene_x_translate: float = 0,
        scene_y_translate: float = 0,
        scene_z_translate: float = 0,
        factor: int = 1,
        scale : Optional[float] = None,
        permutation: bool = True,
        white_bkgd: bool = True,
        n_images = None,
        use_depth: bool = False,
        **kwargs
    ):
        super().__init__()
        assert os.path.isdir(root), f"'{root}' is not a directory"

        if scale is None:
            scale = 1.0
        self.device = device
        self.permutation = permutation
        self.epoch_size = epoch_size
        all_c2w = []
        all_gt = []

        split_name = split if split != "test_train" else "train"
        data_json = os.path.join(root, "transforms_" + split_name + ".json")

        print("MODIFIED LOAD DATA", root)

        j = json.load(open(data_json, "r"))

        # OpenGL -> OpenCV
        cam_trans = torch.diag(torch.tensor([1, -1, 1, 1], dtype=torch.float32))

        paths = map(lambda frame: os.path.join(root, frame["file_path"]), j["frames"])
        with concurrent.futures.ThreadPoolExecutor() as executor:
            all_gt = list(tqdm(executor.map(partial(load_image, scale=scale), paths), total=len(j["frames"])))


        for frame in tqdm(j["frames"]):
            c2w = torch.tensor(frame["transform_matrix"], dtype=torch.float32)
            c2w = c2w @ cam_trans  # To OpenCV
            all_c2w.append(c2w)

        focal = float(
            0.5 * all_gt[0].shape[1] / np.tan(0.5 * j["camera_angle_x"])
        )
        self.c2w = torch.stack(all_c2w)

        self.c2w[:, :3, 3] += torch.tensor([scene_x_translate, scene_y_translate, scene_z_translate])
        self.c2w[:, :3, 3] *= scene_scale
        print(f'Scene scale: {scene_scale}')
        print(f'Scene bounds X: {torch.min(self.c2w[:, 0, 3])} - {torch.max(self.c2w[:, 0, 3])}')
        print(f'Scene bounds Y: {torch.min(self.c2w[:, 1, 3])} - {torch.max(self.c2w[:, 1, 3])}')
        print(f'Scene bounds Z: {torch.min(self.c2w[:, 2, 3])} - {torch.max(self.c2w[:, 2, 3])}')

        self.gt = torch.stack(all_gt).float() / 255.0
        if self.gt.size(-1) == 4:
            if white_bkgd:
                # Apply alpha channel
                self.gt = self.gt[..., :3] * self.gt[..., 3:] + (1.0 - self.gt[..., 3:])
            else:
                self.gt = self.gt[..., :3]

        self.n_images, self.h_full, self.w_full, _ = self.gt.shape
        # Choose a subset of training images
        if n_images is not None:
            if n_images > self.n_images:
                print(f'using {self.n_images} available training views instead of the requested {n_images}.')
                n_images = self.n_images
            self.n_images = n_images
            self.gt = self.gt[0:n_images,...]
            self.c2w = self.c2w[0:n_images,...]

        cx = self.w_full * 0.5
        cy = self.h_full * 0.5
        if "camera_intrinsics" in j:
            intrinsics = j["camera_intrinsics"]
            cx = intrinsics[6] * scale
            cy = intrinsics[7] * scale

        print("Using cx, cy:", cx, cy)

        self.intrins_full : Intrin = Intrin(focal, focal, cx, cy)

        if use_depth and split == 'train':
            depth_paths = map(lambda frame: os.path.join(root, frame["depth_path"]), j["frames"])
            with concurrent.futures.ThreadPoolExecutor() as executor:
                depths = list(tqdm(executor.map(partial(load_depth_file, width=self.w_full, height=self.h_full), depth_paths), total=len(j["frames"])))

            depths = torch.stack(depths).float()
            self.depths = depths * scene_scale

            if 'confidence_path' in j["frames"][0]:
                confidence_paths = map(lambda frame: os.path.join(root, frame["confidence_path"]), j["frames"])
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    confidences = list(tqdm(executor.map(partial(load_confidence_file, width=self.w_full, height=self.h_full), confidence_paths), total=len(j["frames"])))
                self.confidences = torch.stack(confidences).byte()

        self.split = split
        self.scene_scale = scene_scale
        if self.split == "train":
            self.gen_rays(factor=factor)
        else:
            # Rays are not needed for testing
            self.h, self.w = self.h_full, self.w_full
            self.intrins : Intrin = self.intrins_full

        self.should_use_background = False  # Give warning

