# Copyright 2021 Alex Yu
# Render 360 circle path

from typing import List

import torch
import svox2
import svox2.utils
import math
import argparse
import numpy as np
import os
from os import path
from util.dataset import datasets
from util.util import Timing, compute_ssim, viridis_cmap, pose_spherical
from util import config_util
from torch import nn
import imageio
import cv2
from tqdm import tqdm
from dataset_utils import point_cloud

def load_pointcloud(translation: torch.Tensor,
                    scaling: float,
                    dataset_path: str = '/root/svox2/data/livingroom/') -> point_cloud.Pointcloud:
    return point_cloud.Pointcloud.from_dataset(dataset_path,
                                               ['transforms_train.json'],
                                               translation=translation,
                                               scaling=scaling)

def set_harmonics(grid: svox2.svox2.SparseGrid,
                  idxs: torch.Tensor,
                  target_color: torch.Tensor,
                  mask: torch.Tensor,
                  n_harmonics: int = 1):
    optimal_colors = torch.zeros_like(grid.sh_data, dtype=grid.sh_data.dtype).cuda()
    color_capacity = grid.sh_data[0].shape[-1]
    if n_harmonics * 3 > color_capacity:
        raise ValueError(f'Trying to set too many harmonics! You can only set {color_capacity // 3}')
    target_color_masked = target_color[mask]
    target_color_masked = target_color_masked.repeat(1, n_harmonics)
    padding = torch.zeros([target_color_masked.shape[0], color_capacity - n_harmonics * 3]).cuda()

    optimal_colors[idxs] = torch.concat([target_color_masked, padding], axis=-1)
    grid.sh_data = nn.Parameter(optimal_colors)

def set_grid_colors(grid: svox2.svox2.SparseGrid,
                    locations: torch.Tensor,
                    target_color: torch.Tensor,
                    target_density: float = 1.,
                    n_harmonics=1):
    """Sets grid location to target color and density"""

    # Shape: 3 x [n]
    lx, ly, lz = locations.unbind(-1)
    optimal_density = torch.zeros_like(grid.density_data)
    for dx in [0, 1]:
        for dy in [0, 1]:
            for dz in [0, 1]:
                ldx, ldy, ldz = lx + dx, ly + dy, lz + dz
                links = grid.links[ldx, ldy, ldz]
                mask = links >= 0
                idxs = links[mask].long()
                optimal_density[idxs] = target_density
                set_harmonics(grid,
                              idxs,
                              target_color,
                              mask,
                              n_harmonics=n_harmonics)
    
    grid.density_data = nn.Parameter(optimal_density)


parser = argparse.ArgumentParser()

config_util.define_common_args(parser)

parser.add_argument('--n_eval', '-n', type=int, default=100000, help='images to evaluate (equal interval), at most evals every image')
parser.add_argument('--traj_type',
                    choices=['spiral', 'circle'],
                    default='spiral',
                    help="Render a spiral (doubles length, using 2 elevations), or just a cirle")
parser.add_argument('--fps',
                    type=int,
                    default=30,
                    help="FPS of video")
parser.add_argument(
                "--width", "-W", type=float, default=None, help="Rendering image width (only if not --traj)"
                        )
parser.add_argument(
                    "--height", "-H", type=float, default=None, help="Rendering image height (only if not --traj)"
                            )
parser.add_argument(
	"--num_views", "-N", type=int, default=600,
    help="Number of frames to render"
)

# Path adjustment
parser.add_argument(
    "--offset", type=str, default="0,0,0", help="Center point to rotate around (only if not --traj)"
)
parser.add_argument("--radius", type=float, default=0.85, help="Radius of orbit (only if not --traj)")
parser.add_argument(
    "--elevation",
    type=float,
    default=-45.0,
    help="Elevation of orbit in deg, negative is above",
)
parser.add_argument(
    "--elevation2",
    type=float,
    default=-12.0,
    help="Max elevation, only for spiral",
)
parser.add_argument(
    "--vec_up",
    type=str,
    default=None,
    help="up axis for camera views (only if not --traj);"
    "3 floats separated by ','; if not given automatically determined",
)
parser.add_argument(
    "--vert_shift",
    type=float,
    default=0.0,
    help="vertical shift by up axis"
)

# Camera adjustment
parser.add_argument('--crop',
                    type=float,
                    default=1.0,
                    help="Crop (0, 1], 1.0 = full image")

# Foreground/background only
parser.add_argument('--nofg',
                    action='store_true',
                    default=False,
                    help="Do not render foreground (if using BG model)")
parser.add_argument('--nobg',
                    action='store_true',
                    default=False,
                    help="Do not render background (if using BG model)")

# Random debugging features
parser.add_argument('--blackbg',
                    action='store_true',
                    default=False,
                    help="Force a black BG (behind BG model) color; useful for debugging 'clouds'")
args = parser.parse_args()
config_util.maybe_merge_config_file(args, allow_invalid=True)
device = 'cuda:0'


dset = datasets[args.dataset_type](args.data_dir, split="test",
                                    **config_util.build_data_options(args))

if args.vec_up is None:
    up_rot = dset.c2w[:, :3, :3].cpu().numpy()
    ups = np.matmul(up_rot, np.array([0, -1.0, 0])[None, :, None])[..., 0]
    args.vec_up = np.mean(ups, axis=0)
    args.vec_up /= np.linalg.norm(args.vec_up)
    print('  Auto vec_up', args.vec_up)
else:
    args.vec_up = np.array(list(map(float, args.vec_up.split(","))))


args.offset = np.array(list(map(float, args.offset.split(","))))
if args.traj_type == 'spiral':
    angles = np.linspace(-180, 180, args.num_views + 1)[:-1]
    elevations = np.linspace(args.elevation, args.elevation2, args.num_views)
    c2ws = [
        pose_spherical(
            angle,
            ele,
            args.radius,
            args.offset,
            vec_up=args.vec_up,
        )
        for ele, angle in zip(elevations, angles)
    ]
    c2ws += [
        pose_spherical(
            angle,
            ele,
            args.radius,
            args.offset,
            vec_up=args.vec_up,
        )
        for ele, angle in zip(reversed(elevations), angles)
    ]
else :
    c2ws = [
        pose_spherical(
            angle,
            args.elevation,
            args.radius,
            args.offset,
            vec_up=args.vec_up,
        )
        for angle in np.linspace(-180, 180, args.num_views + 1)[:-1]
    ]
c2ws = np.stack(c2ws, axis=0)
if args.vert_shift != 0.0:
    c2ws[:, :3, 3] += np.array(args.vec_up) * args.vert_shift
c2ws = torch.from_numpy(c2ws).to(device=device)

render_out_path = path.join('./manual_grid/', 'circle_renders')

# Handle various image transforms
if args.crop != 1.0:
    render_out_path += f'_crop{args.crop}'
if args.vert_shift != 0.0:
    render_out_path += f'_vshift{args.vert_shift}'

grid = svox2.SparseGrid(reso=(100, 100, 100),
                        center=0.0,
                        radius=1.,
                        use_sphere_bound=False,
                        basis_dim=1,
                        use_z_order=True,
                        device=device,
                        mlp_posenc_size=0,
                        mlp_width=16,
                        background_nlayers=0,
                        background_reso=0)
translation = torch.Tensor([1.2, 0, 1.4])
scaling = 0.25
point_cloud = load_pointcloud(translation=translation, scaling=scaling)
point_cloud = point_cloud.get_pruned_pointcloud(10000000)
points = point_cloud.points
colors = point_cloud.features.cuda().float() / 255.

# WARNING: Set colors to blue
colors = torch.concat([torch.zeros([colors.shape[0], 2]),
                       torch.ones([colors.shape[0], 1])], dim=-1).float().cuda()

locations = grid.world2grid(points)
locations.clamp_min_(0.0)
for i in range(3):
    locations[:, i].clamp_max_(grid.links.size(i) - 1)
l = locations.to(torch.long)
for i in range(3):
    l[:, i].clamp_max_(grid.links.size(i) - 2)

set_grid_colors(
    grid,
    l.cuda(),
    target_color=colors,
    n_harmonics=1   
)

config_util.setup_render_opts(grid.opt, args)

if args.blackbg:
    print('Forcing black bg')
    render_out_path += '_blackbg'
    grid.opt.background_brightness = 0.0

render_out_path += '.mp4'
print('Writing to', render_out_path)

# NOTE: no_grad enables the fast image-level rendering kernel for cuvol backend only
# other backends will manually generate rays per frame (slow)
with torch.no_grad():
    n_images = c2ws.size(0)
    img_eval_interval = max(n_images // args.n_eval, 1)
    avg_psnr = 0.0
    avg_ssim = 0.0
    avg_lpips = 0.0
    n_images_gen = 0
    frames = []
    #  if args.near_clip >= 0.0:
    grid.opt.near_clip = 0.0 #args.near_clip
    if args.width is None:
        args.width = dset.get_image_size(0)[1]
    if args.height is None:
        args.height = dset.get_image_size(0)[0]

    for img_id in tqdm(range(0, n_images, img_eval_interval)):
        dset_h, dset_w = args.height, args.width
        im_size = dset_h * dset_w
        w = dset_w if args.crop == 1.0 else int(dset_w * args.crop)
        h = dset_h if args.crop == 1.0 else int(dset_h * args.crop)

        cam = svox2.Camera(c2ws[img_id],
                           dset.intrins.get('fx', 0),
                           dset.intrins.get('fy', 0),
                           w * 0.5,
                           h * 0.5,
                           w, h,
                           ndc_coeffs=(-1.0, -1.0))
        torch.cuda.synchronize()
        im = grid.volume_render_image(cam, use_kernel=True)
        torch.cuda.synchronize()
        im.clamp_(0.0, 1.0)

        im = im.cpu().numpy()
        im = (im * 255).astype(np.uint8)
        frames.append(im)
        im = None
        n_images_gen += 1
    if len(frames):
        vid_path = render_out_path
        imageio.mimwrite(vid_path, frames, fps=args.fps, macro_block_size=8)  # pip install imageio-ffmpeg


