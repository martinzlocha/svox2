# Copyright 2021 Alex Yu

# First, install svox2
# Then, python opt.py <path_to>/nerf_synthetic/<scene> -t ckpt/<some_name>
# or use launching script:   sh launch.sh <EXP_NAME> <GPU> <DATA_DIR>
import torch
import torch.cuda
import torch.optim
import torch.nn.functional as F
import svox2
import json
import imageio
import os
from os import path
import shutil
import gc
import numpy as np
import math
import argparse
import cv2
from util.dataset import datasets
from util.util import Timing, get_expon_lr_func, generate_dirs_equirect, viridis_cmap, compute_ssim
from util import config_util
from dataset_utils import point_cloud

from warnings import warn
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

WANDB_ON=False

from tqdm import tqdm
from typing import NamedTuple, Optional, Union


def log_image(image_name, image, global_step):
    summary_writer.add_image(image_name, image, global_step=gstep_id_base, dataformats='HWC')

    if WANDB_ON:
        wandb.log({image_name: wandb.Image(image.numpy())}, step=gstep_id_base)

device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
config_util.define_common_args(parser)


group = parser.add_argument_group("general")
group.add_argument('--train_dir', '-t', type=str, default='ckpt',
                     help='checkpoint and logging directory')

group.add_argument('--reso',
                        type=str,
                        default=
                        "[[256, 256, 256], [512, 512, 512]]",
                       help='List of grid resolution (will be evaled as json);'
                            'resamples to the next one every upsamp_every iters, then ' +
                            'stays at the last one; ' +
                            'should be a list where each item is a list of 3 ints or an int')
group.add_argument('--upsamp_every', type=int, default=
                     3 * 12800,
                    help='upsample the grid every x iters')
group.add_argument('--init_iters', type=int, default=
                     0,
                    help='do not upsample for first x iters')
group.add_argument('--upsample_density_add', type=float, default=
                    0.0,
                    help='add the remaining density by this amount when upsampling')

group.add_argument('--basis_type',
                    choices=['sh', '3d_texture', 'mlp'],
                    default='sh',
                    help='Basis function type')

group.add_argument('--basis_reso', type=int, default=32,
                   help='basis grid resolution (only for learned texture)')
group.add_argument('--sh_dim', type=int, default=9, help='SH/learned basis dimensions (at most 10)')

group.add_argument('--mlp_posenc_size', type=int, default=4, help='Positional encoding size if using MLP basis; 0 to disable')
group.add_argument('--mlp_width', type=int, default=32, help='MLP width if using MLP basis')

group.add_argument('--background_nlayers', type=int, default=0,#32,
                   help='Number of background layers (0=disable BG model)')
group.add_argument('--background_reso', type=int, default=512, help='Background resolution')



group = parser.add_argument_group("optimization")
group.add_argument('--n_iters', type=int, default=10 * 12800, help='total number of iters to optimize for')
group.add_argument('--batch_size', type=int, default=
                     5000,
                     #100000,
                     #  2000,
                   help='batch size')


# TODO: make the lr higher near the end
group.add_argument('--sigma_optim', choices=['sgd', 'rmsprop'], default='rmsprop', help="Density optimizer")
group.add_argument('--lr_sigma', type=float, default=3e1, help='SGD/rmsprop lr for sigma')
group.add_argument('--lr_sigma_final', type=float, default=5e-2)
group.add_argument('--lr_sigma_decay_steps', type=int, default=250000)
group.add_argument('--lr_sigma_delay_steps', type=int, default=15000,
                   help="Reverse cosine steps (0 means disable)")
group.add_argument('--lr_sigma_delay_mult', type=float, default=1e-2)#1e-4)#1e-4)


group.add_argument('--sh_optim', choices=['sgd', 'rmsprop'], default='rmsprop', help="SH optimizer")
group.add_argument('--lr_sh', type=float, default=
                    1e-2,
                   help='SGD/rmsprop lr for SH')
group.add_argument('--lr_sh_final', type=float,
                      default=
                    5e-6
                    )
group.add_argument('--lr_sh_decay_steps', type=int, default=250000)
group.add_argument('--lr_sh_delay_steps', type=int, default=0, help="Reverse cosine steps (0 means disable)")
group.add_argument('--lr_sh_delay_mult', type=float, default=1e-2)

group.add_argument('--lr_fg_begin_step', type=int, default=0, help="Foreground begins training at given step number")

# BG LRs
group.add_argument('--bg_optim', choices=['sgd', 'rmsprop'], default='rmsprop', help="Background optimizer")
group.add_argument('--lr_sigma_bg', type=float, default=3e0,
                    help='SGD/rmsprop lr for background')
group.add_argument('--lr_sigma_bg_final', type=float, default=3e-3,
                    help='SGD/rmsprop lr for background')
group.add_argument('--lr_sigma_bg_decay_steps', type=int, default=250000)
group.add_argument('--lr_sigma_bg_delay_steps', type=int, default=0, help="Reverse cosine steps (0 means disable)")
group.add_argument('--lr_sigma_bg_delay_mult', type=float, default=1e-2)

group.add_argument('--lr_color_bg', type=float, default=1e-1,
                    help='SGD/rmsprop lr for background')
group.add_argument('--lr_color_bg_final', type=float, default=5e-6,#1e-4,
                    help='SGD/rmsprop lr for background')
group.add_argument('--lr_color_bg_decay_steps', type=int, default=250000)
group.add_argument('--lr_color_bg_delay_steps', type=int, default=0, help="Reverse cosine steps (0 means disable)")
group.add_argument('--lr_color_bg_delay_mult', type=float, default=1e-2)
# END BG LRs

group.add_argument('--basis_optim', choices=['sgd', 'rmsprop'], default='rmsprop', help="Learned basis optimizer")
group.add_argument('--lr_basis', type=float, default=#2e6,
                      1e-6,
                   help='SGD/rmsprop lr for SH')
group.add_argument('--lr_basis_final', type=float,
                      default=
                      1e-6
                    )
group.add_argument('--lr_basis_decay_steps', type=int, default=250000)
group.add_argument('--lr_basis_delay_steps', type=int, default=0,#15000,
                   help="Reverse cosine steps (0 means disable)")
group.add_argument('--lr_basis_begin_step', type=int, default=0)#4 * 12800)
group.add_argument('--lr_basis_delay_mult', type=float, default=1e-2)

group.add_argument('--rms_beta', type=float, default=0.95, help="RMSProp exponential averaging factor")

group.add_argument('--print_every', type=int, default=20, help='print every')
group.add_argument('--save_every', type=int, default=5,
                   help='save every x epochs')
group.add_argument('--eval_every', type=int, default=1,
                   help='evaluate every x epochs')

group.add_argument('--init_sigma', type=float,
                   default=0.1,
                   help='initialization sigma')
group.add_argument('--init_sigma_bg', type=float,
                   default=0.1,
                   help='initialization sigma (for BG)')

# Extra logging
group.add_argument('--log_mse_image', action='store_true', default=False)
group.add_argument('--log_depth_map', action='store_true', default=False)
group.add_argument('--log_depth_map_use_thresh', type=float, default=None,
        help="If specified, uses the Dex-neRF version of depth with given thresh; else returns expected term")


group = parser.add_argument_group("misc experiments")
group.add_argument('--thresh_type',
                    choices=["weight", "sigma"],
                    default="weight",
                   help='Upsample threshold type')
group.add_argument('--weight_thresh', type=float,
                    default=0.0005 * 512,
                    #  default=0.025 * 512,
                   help='Upsample weight threshold; will be divided by resulting z-resolution')
group.add_argument('--density_thresh', type=float,
                    default=5.0,
                   help='Upsample sigma threshold')
group.add_argument('--background_density_thresh', type=float,
                    default=1.0+1e-9,
                   help='Background sigma threshold for sparsification')
group.add_argument('--max_grid_elements', type=int,
                    default=44_000_000,
                   help='Max items to store after upsampling '
                        '(the number here is given for 22GB memory)')

group.add_argument('--tune_mode', action='store_true', default=False,
                   help='hypertuning mode (do not save, for speed)')
group.add_argument('--tune_nosave', action='store_true', default=False,
                   help='do not save any checkpoint even at the end')



group = parser.add_argument_group("losses")
# Foreground TV
group.add_argument('--lambda_tv', type=float, default=1e-5)
group.add_argument('--tv_sparsity', type=float, default=0.01)
group.add_argument('--tv_logalpha', action='store_true', default=False,
                   help='Use log(1-exp(-delta * sigma)) as in neural volumes')

group.add_argument('--lambda_tv_sh', type=float, default=1e-3)
group.add_argument('--tv_sh_sparsity', type=float, default=0.01)

group.add_argument('--lambda_tv_lumisphere', type=float, default=0.0)#1e-2)#1e-3)
group.add_argument('--tv_lumisphere_sparsity', type=float, default=0.01)
group.add_argument('--tv_lumisphere_dir_factor', type=float, default=0.0)

group.add_argument('--tv_decay', type=float, default=1.0)

group.add_argument('--lambda_l2_sh', type=float, default=0.0)#1e-4)
group.add_argument('--tv_early_only', type=int, default=1, help="Turn off TV regularization after the first split/prune")

group.add_argument('--tv_contiguous', type=int, default=1,
                        help="Apply TV only on contiguous link chunks, which is faster")
# End Foreground TV

group.add_argument('--lambda_sparsity', type=float, default=
                    0.0,
                    help="Weight for sparsity loss as in SNeRG/PlenOctrees " +
                         "(but applied on the ray)")
group.add_argument('--lambda_beta', type=float, default=
                    0.0,
                    help="Weight for beta distribution sparsity loss as in neural volumes")


# Background TV
group.add_argument('--lambda_tv_background_sigma', type=float, default=1e-2)
group.add_argument('--lambda_tv_background_color', type=float, default=1e-2)

group.add_argument('--tv_background_sparsity', type=float, default=0.01)
# End Background TV

# Basis TV
group.add_argument('--lambda_tv_basis', type=float, default=0.0,
                   help='Learned basis total variation loss')
# End Basis TV

group.add_argument('--weight_decay_sigma', type=float, default=1.0)
group.add_argument('--weight_decay_sh', type=float, default=1.0)

group.add_argument('--lr_decay', action='store_true', default=True)

group.add_argument('--n_train', type=int, default=None, help='Number of training images. Defaults to use all avaiable.')

group.add_argument('--nosphereinit', action='store_true', default=False,
                     help='do not start with sphere bounds (please do not use for 360)')

args = parser.parse_args()
config_util.maybe_merge_config_file(args)

if WANDB_ON:
    import wandb
    wandb.init(project="roomoxels", config=vars(args), anonymous="allow", tags=[args.data_dir])
    args = wandb.config
    args_as_dict = wandb.config.as_dict()
else:
    args_as_dict = vars(args)


assert args.lr_sigma_final <= args.lr_sigma, "lr_sigma must be >= lr_sigma_final"
assert args.lr_sh_final <= args.lr_sh, "lr_sh must be >= lr_sh_final"
assert args.lr_basis_final <= args.lr_basis, "lr_basis must be >= lr_basis_final"

os.makedirs(args.train_dir, exist_ok=True)
summary_writer = SummaryWriter(args.train_dir)

reso_list = json.loads(args.reso)
reso_id = 0

def load_pointcloud(dataset_path: str = '/root/svox2/data/livingroom/') -> point_cloud.Pointcloud:
    translation = torch.Tensor([1.2, 0, 1.4])
    scaling = 0.2
    return point_cloud.Pointcloud.from_dataset(dataset_path,
                                               ['transforms_train.json'],
                                               translation=translation,
                                               scaling=scaling)

with open(path.join(args.train_dir, 'args.json'), 'w') as f:
    json.dump(args_as_dict, f, indent=2)
    # Changed name to prevent errors
    shutil.copyfile(__file__, path.join(args.train_dir, 'opt_frozen.py'))

torch.manual_seed(20200823)
np.random.seed(20200823)

factor = 1
dset = datasets[args.dataset_type](
               args.data_dir,
               split="train",
               device=device,
               factor=factor,
               n_images=args.n_train,
               **config_util.build_data_options(args))

if args.background_nlayers > 0 and not dset.should_use_background:
    warn('Using a background model for dataset type ' + str(type(dset)) + ' which typically does not use background')

dset_test = datasets[args.dataset_type](
        args.data_dir, split="test", **config_util.build_data_options(args))

global_start_time = datetime.now()

grid = svox2.SparseGrid(reso=reso_list[reso_id],
                        center=dset.scene_center,
                        radius=dset.scene_radius,
                        use_sphere_bound=dset.use_sphere_bound and not args.nosphereinit,
                        basis_dim=args.sh_dim,
                        use_z_order=True,
                        device=device,
                        basis_reso=args.basis_reso,
                        basis_type=svox2.__dict__['BASIS_TYPE_' + args.basis_type.upper()],
                        mlp_posenc_size=args.mlp_posenc_size,
                        mlp_width=args.mlp_width,
                        background_nlayers=args.background_nlayers,
                        background_reso=args.background_reso)

# DC -> gray; mind the SH scaling!
grid.sh_data.data[:] = 0.0
grid.density_data.data[:] = 0.0 if args.lr_fg_begin_step > 0 else args.init_sigma

if grid.use_background:
    grid.background_data.data[..., -1] = args.init_sigma_bg
    #  grid.background_data.data[..., :-1] = 0.5 / svox2.utils.SH_C0

#  grid.sh_data.data[:, 0] = 4.0
#  osh = grid.density_data.data.shape
#  den = grid.density_data.data.view(grid.links.shape)
#  #  den[:] = 0.00
#  #  den[:, :256, :] = 1e9
#  #  den[:, :, 0] = 1e9
#  grid.density_data.data = den.view(osh)

optim_basis_mlp = None

if grid.basis_type == svox2.BASIS_TYPE_3D_TEXTURE:
    grid.reinit_learned_bases(init_type='sh')
    #  grid.reinit_learned_bases(init_type='fourier')
    #  grid.reinit_learned_bases(init_type='sg', upper_hemi=True)
    #  grid.basis_data.data.normal_(mean=0.28209479177387814, std=0.001)

elif grid.basis_type == svox2.BASIS_TYPE_MLP:
    # MLP!
    optim_basis_mlp = torch.optim.Adam(
                    grid.basis_mlp.parameters(),
                    lr=args.lr_basis
                )


grid.requires_grad_(True)
config_util.setup_render_opts(grid.opt, args)
print('Render options', grid.opt)

gstep_id_base = 0

resample_cameras = [
        svox2.Camera(c2w.to(device=device),
                     dset.intrins.get('fx', i),
                     dset.intrins.get('fy', i),
                     dset.intrins.get('cx', i),
                     dset.intrins.get('cy', i),
                     width=dset.get_image_size(i)[1],
                     height=dset.get_image_size(i)[0],
                     ndc_coeffs=dset.ndc_coeffs) for i, c2w in enumerate(dset.c2w)
    ]
ckpt_path = path.join(args.train_dir, 'ckpt.npz')

lr_sigma_func = get_expon_lr_func(args.lr_sigma, args.lr_sigma_final, args.lr_sigma_delay_steps,
                                  args.lr_sigma_delay_mult, args.lr_sigma_decay_steps)
lr_sh_func = get_expon_lr_func(args.lr_sh, args.lr_sh_final, args.lr_sh_delay_steps,
                               args.lr_sh_delay_mult, args.lr_sh_decay_steps)
lr_basis_func = get_expon_lr_func(args.lr_basis, args.lr_basis_final, args.lr_basis_delay_steps,
                               args.lr_basis_delay_mult, args.lr_basis_decay_steps)
lr_sigma_bg_func = get_expon_lr_func(args.lr_sigma_bg, args.lr_sigma_bg_final, args.lr_sigma_bg_delay_steps,
                               args.lr_sigma_bg_delay_mult, args.lr_sigma_bg_decay_steps)
lr_color_bg_func = get_expon_lr_func(args.lr_color_bg, args.lr_color_bg_final, args.lr_color_bg_delay_steps,
                               args.lr_color_bg_delay_mult, args.lr_color_bg_decay_steps)
lr_sigma_factor = 1.0
lr_sh_factor = 1.0
lr_basis_factor = 1.0

last_upsamp_step = args.init_iters
pc_point_count = 400000
pc_orig = load_pointcloud()
pc_large = pc_orig.get_pruned_pointcloud(pc_point_count * 2)
pc = pc_orig.get_pruned_pointcloud(pc_point_count)
pc_points = pc.points.cuda()
negative_points = torch.rand([pc_point_count, 3]).cuda() * 0.2 - 0.1
pc_keep_points = pc_large.points.cuda()

def set_grid_density(grid: svox2.svox2.SparseGrid,
                    points: torch.Tensor,
                    target_density: float = 10.):
    """Sets grid location to target color and density"""

    # Shape: 3 x [n]
    points = grid.world2grid(points)
    points.clamp_min_(0.0)
    for i in range(3):
        points[:, i].clamp_max_(grid.links.size(i) - 1)
    l = points.to(torch.long)
    for i in range(3):
        l[:, i].clamp_max_(grid.links.size(i) - 2)
    lx, ly, lz = l.unbind(-1)
    optimal_density = torch.zeros_like(grid.density_data)
    for dx in [0, 1]:
        for dy in [0, 1]:
            for dz in [0, 1]:
                ldx, ldy, ldz = lx + dx, ly + dy, lz + dz
                links = grid.links[ldx, ldy, ldz]
                mask = links >= 0
                links = links[mask].long()
                link_bincount = torch.bincount(links)
                idxs = torch.argsort(link_bincount, descending=True)[:25000]
                optimal_density[idxs] = target_density
    
    grid.density_data = torch.nn.Parameter(optimal_density)

def get_links(points):
    points = grid.world2grid(points)
    points.clamp_min_(0.0)
    for i in range(3):
        points[:, i].clamp_max_(grid.links.size(i) - 1)
    l = points.to(torch.long)
    for i in range(3):
        l[:, i].clamp_max_(grid.links.size(i) - 2)
    wb = points - l
    wa = 1.0 - wb
    lx, ly, lz = l.unbind(-1)
    links000 = grid.links[lx, ly, lz]
    links001 = grid.links[lx, ly, lz + 1]
    links010 = grid.links[lx, ly + 1, lz]
    links011 = grid.links[lx, ly + 1, lz + 1]
    links100 = grid.links[lx + 1, ly, lz]
    links101 = grid.links[lx + 1, ly, lz + 1]
    links110 = grid.links[lx + 1, ly + 1, lz]
    links111 = grid.links[lx + 1, ly + 1, lz + 1]
    
    return (links000, links001, links010, links011, links100, links101, links110, links111), (wa, wb)

#optimal_density = get_links(pc_points)
#neg_optimal_density = get_links(negative_points)
set_grid_density(grid, pc_keep_points)
grid.save_voxels_to_dict(ckpt_path)


if args.enable_random:
    warn("Randomness is enabled for training (normal for LLFF & scenes with background)")

epoch_id = -1
grid.resample(reso=reso_list[0],
                sigma_thresh=args.density_thresh,
                weight_thresh=args.weight_thresh / reso_list[0][2],
                dilate=1,
                cameras=resample_cameras if args.thresh_type == 'weight' else None,
                max_elements=args.max_grid_elements)
while True:
    dset.shuffle_rays()
    epoch_id += 1
    epoch_size = dset.rays.origins.size(0)
    batches_per_epoch = (epoch_size-1)//args.batch_size+1
    # Test
    def eval_step():
        # Put in a function to avoid memory leak
        print('Eval step')
        with torch.no_grad():
            stats_test = {'psnr' : 0.0, 'mse' : 0.0}

            # Standard set
            N_IMGS_TO_EVAL = min(20 if epoch_id > 0 else 5, dset_test.n_images)
            N_IMGS_TO_SAVE = N_IMGS_TO_EVAL # if not args.tune_mode else 1
            img_eval_interval = dset_test.n_images // N_IMGS_TO_EVAL
            img_save_interval = (N_IMGS_TO_EVAL // N_IMGS_TO_SAVE)
            img_ids = range(0, dset_test.n_images, img_eval_interval)

            # Special 'very hard' specular + fuzz set
            #  img_ids = [2, 5, 7, 9, 21,
            #             44, 45, 47, 49, 56,
            #             80, 88, 99, 115, 120,
            #             154]
            #  img_save_interval = 1

            n_images_gen = 0
            for i, img_id in tqdm(enumerate(img_ids), total=len(img_ids)):
                c2w = dset_test.c2w[img_id].to(device=device)
                cam = svox2.Camera(c2w,
                                   dset_test.intrins.get('fx', img_id),
                                   dset_test.intrins.get('fy', img_id),
                                   dset_test.intrins.get('cx', img_id),
                                   dset_test.intrins.get('cy', img_id),
                                   width=dset_test.get_image_size(img_id)[1],
                                   height=dset_test.get_image_size(img_id)[0],
                                   ndc_coeffs=dset_test.ndc_coeffs)
                rgb_pred_test = grid.volume_render_image(cam, use_kernel=True)
                rgb_gt_test = dset_test.gt[img_id].to(device=device)
                all_mses = ((rgb_gt_test - rgb_pred_test) ** 2).cpu()
                if i % img_save_interval == 0:
                    img_pred = rgb_pred_test.cpu()
                    img_pred.clamp_max_(1.0)
                    log_image(f'test/image_{img_id:04d}', img_pred, gstep_id_base)
                    if args.log_mse_image:
                        mse_img = all_mses / all_mses.max()
                        log_image(f'test/mse_map_{img_id:04d}', mse_img, gstep_id_base)
                    if args.log_depth_map:
                        depth_img = grid.volume_render_depth_image(cam,
                                    args.log_depth_map_use_thresh if
                                    args.log_depth_map_use_thresh else None
                                )
                        depth_img = viridis_cmap(depth_img.cpu())
                        log_image(f'test/depth_map_{img_id:04d}', depth_img, gstep_id_base)

                rgb_pred_test = rgb_gt_test = None
                mse_num : float = all_mses.mean().item()
                psnr = -10.0 * math.log10(mse_num)
                if math.isnan(psnr):
                    print('NAN PSNR', i, img_id, mse_num)
                    assert False
                stats_test['mse'] += mse_num
                stats_test['psnr'] += psnr
                n_images_gen += 1

            if grid.basis_type == svox2.BASIS_TYPE_3D_TEXTURE or \
               grid.basis_type == svox2.BASIS_TYPE_MLP:
                 # Add spherical map visualization
                EQ_RESO = 256
                eq_dirs = generate_dirs_equirect(EQ_RESO * 2, EQ_RESO)
                eq_dirs = torch.from_numpy(eq_dirs).to(device=device).view(-1, 3)

                if grid.basis_type == svox2.BASIS_TYPE_MLP:
                    sphfuncs = grid._eval_basis_mlp(eq_dirs)
                else:
                    sphfuncs = grid._eval_learned_bases(eq_dirs)
                sphfuncs = sphfuncs.view(EQ_RESO, EQ_RESO*2, -1).permute([2, 0, 1]).cpu().numpy()

                stats = [(sphfunc.min(), sphfunc.mean(), sphfunc.max())
                        for sphfunc in sphfuncs]
                sphfuncs_cmapped = [viridis_cmap(sphfunc) for sphfunc in sphfuncs]
                for im, (minv, meanv, maxv) in zip(sphfuncs_cmapped, stats):
                    cv2.putText(im, f"{minv:.4f} {meanv:.4f} {maxv:.4f}", (10, 20),
                                0, 0.5, [255, 0, 0])
                sphfuncs_cmapped = np.concatenate(sphfuncs_cmapped, axis=0)
                summary_writer.add_image(f'test/spheric',
                        sphfuncs_cmapped, global_step=gstep_id_base, dataformats='HWC')
                # END add spherical map visualization

            stats_test['mse'] /= n_images_gen
            stats_test['psnr'] /= n_images_gen
            for stat_name in stats_test:
                summary_writer.add_scalar('test/' + stat_name,
                        stats_test[stat_name], global_step=gstep_id_base)
            summary_writer.add_scalar('epoch_id', float(epoch_id), global_step=gstep_id_base)
            print('eval stats:', stats_test)
            if WANDB_ON:
                wandb.log({
                    "eval/mse": stats_test['mse'],
                    "eval/psnr": stats_test['psnr'],
                }, step=gstep_id_base)
    if epoch_id % max(factor, args.eval_every) == 0: #and (epoch_id > 0 or not args.tune_mode):
        # NOTE: we do an eval sanity check, if not in tune_mode
        eval_step()
        gc.collect()
    
    def train_step():
        print('Train step')
        pbar = tqdm(enumerate(range(0, epoch_size, args.batch_size)), total=batches_per_epoch)
        stats = {"mse" : 0.0, "psnr" : 0.0, "invsqr_mse" : 0.0}

        for iter_id, batch_begin in pbar:
            gstep_id = iter_id + gstep_id_base
            if args.lr_fg_begin_step > 0 and gstep_id == args.lr_fg_begin_step:
                grid.density_data.data[:] = args.init_sigma
            lr_sigma = lr_sigma_func(gstep_id) * lr_sigma_factor
            lr_sh = lr_sh_func(gstep_id) * lr_sh_factor
            lr_basis = lr_basis_func(gstep_id - args.lr_basis_begin_step) * lr_basis_factor
            lr_sigma_bg = lr_sigma_bg_func(gstep_id - args.lr_basis_begin_step) * lr_basis_factor
            lr_color_bg = lr_color_bg_func(gstep_id - args.lr_basis_begin_step) * lr_basis_factor
            if not args.lr_decay:
                lr_sigma = args.lr_sigma * lr_sigma_factor
                lr_sh = args.lr_sh * lr_sh_factor
                lr_basis = args.lr_basis * lr_basis_factor

            batch_end = min(batch_begin + args.batch_size, epoch_size)
            batch_origins = dset.rays.origins[batch_begin: batch_end]
            batch_dirs = dset.rays.dirs[batch_begin: batch_end]
            rgb_gt = dset.rays.gt[batch_begin: batch_end]
            rays = svox2.Rays(batch_origins, batch_dirs)

            """
            if iter_id % 500 == 0 and iter_id > 0:
                pc = pc_orig.get_pruned_pointcloud(pc_point_count)
                pc_points = pc.points.cuda()
                negative_points = torch.rand([pc_point_count, 3]).cuda() * 0.2 - 0.1
                optimal_density = get_links(pc_points)
                neg_optimal_density = get_links(negative_points)

            sigmas = []
            for links in optimal_density[0]:
                sigma, _ = grid._fetch_links(links)
                sigmas.append(sigma)
            
            sigma000 = sigmas[0]
            sigma001 = sigmas[1]
            sigma010 = sigmas[2]
            sigma011 = sigmas[3]
            sigma100 = sigmas[4]
            sigma101 = sigmas[5]
            sigma110 = sigmas[6]
            sigma111 = sigmas[7]
            wa, wb = optimal_density[-1]

            c00 = sigma000 * wa[:, 2:] + sigma001 * wb[:, 2:]
            c01 = sigma010 * wa[:, 2:] + sigma011 * wb[:, 2:]
            c10 = sigma100 * wa[:, 2:] + sigma101 * wb[:, 2:]
            c11 = sigma110 * wa[:, 2:] + sigma111 * wb[:, 2:]
            c0 = c00 * wa[:, 1:2] + c01 * wb[:, 1:2]
            c1 = c10 * wa[:, 1:2] + c11 * wb[:, 1:2]
            samples_sigma = c0 * wa[:, :1] + c1 * wb[:, :1]
            (1e-2 * -samples_sigma.mean()).backward()

            sigmas = []
            for links in neg_optimal_density[0]:
                sigma, _ = grid._fetch_links(links)
                sigmas.append(sigma)
            
            sigma000 = sigmas[0]
            sigma001 = sigmas[1]
            sigma010 = sigmas[2]
            sigma011 = sigmas[3]
            sigma100 = sigmas[4]
            sigma101 = sigmas[5]
            sigma110 = sigmas[6]
            sigma111 = sigmas[7]
            wa, wb = neg_optimal_density[-1]

            c00 = sigma000 * wa[:, 2:] + sigma001 * wb[:, 2:]
            c01 = sigma010 * wa[:, 2:] + sigma011 * wb[:, 2:]
            c10 = sigma100 * wa[:, 2:] + sigma101 * wb[:, 2:]
            c11 = sigma110 * wa[:, 2:] + sigma111 * wb[:, 2:]
            c0 = c00 * wa[:, 1:2] + c01 * wb[:, 1:2]
            c1 = c10 * wa[:, 1:2] + c11 * wb[:, 1:2]
            samples_sigma = c0 * wa[:, :1] + c1 * wb[:, :1]
            (1e-2 * samples_sigma.mean()).backward()
            """
            ### ADD DEPTH LOSS ###
            #total_loss = 0.
            #aggregate = torch.sum(grid.density_data)
            #total_pos = 0.
            #total_cnt = 0
            #for idx in optimal_density:
            #    pos_sum = torch.sum(grid.density_data[idx])
            #    total_loss = total_loss + (pos_sum / idx.shape[0])
            #    total_pos = total_pos + pos_sum
            #    total_cnt = total_cnt + idx.shape[0]
            #total_loss -= (aggregate - total_pos) / (grid.density_data.shape[0] - total_cnt)


            #  with Timing("volrend_fused"):
            rgb_pred = grid.volume_render_fused(rays, rgb_gt,
                    beta_loss=args.lambda_beta,
                    sparsity_loss=args.lambda_sparsity,
                    randomize=args.enable_random)

            #  with Timing("loss_comp"):
            mse = F.mse_loss(rgb_gt, rgb_pred)

            # Stats
            mse_num : float = mse.detach().item()
            psnr = -10.0 * math.log10(mse_num)
            stats['mse'] += mse_num
            stats['psnr'] += psnr
            stats['invsqr_mse'] += 1.0 / mse_num ** 2

            if (iter_id + 1) % args.print_every == 0:
                # Print averaged stats
                pbar.set_description(f'epoch {epoch_id} psnr={psnr:.2f}')
                for stat_name in stats:
                    stat_val = stats[stat_name] / args.print_every
                    summary_writer.add_scalar(stat_name, stat_val, global_step=gstep_id)
                    stats[stat_name] = 0.0
                    if WANDB_ON:
                        wandb.log({"train/" + stat_name: stat_val}, step=gstep_id)
                #  if args.lambda_tv > 0.0:
                #      with torch.no_grad():
                #          tv = grid.tv(logalpha=args.tv_logalpha, ndc_coeffs=dset.ndc_coeffs)
                #      summary_writer.add_scalar("loss_tv", tv, global_step=gstep_id)
                #  if args.lambda_tv_sh > 0.0:
                #      with torch.no_grad():
                #          tv_sh = grid.tv_color()
                #      summary_writer.add_scalar("loss_tv_sh", tv_sh, global_step=gstep_id)
                #  with torch.no_grad():
                #      tv_basis = grid.tv_basis() #  summary_writer.add_scalar("loss_tv_basis", tv_basis, global_step=gstep_id)
                summary_writer.add_scalar("lr_sh", lr_sh, global_step=gstep_id)
                summary_writer.add_scalar("lr_sigma", lr_sigma, global_step=gstep_id)
                if grid.basis_type == svox2.BASIS_TYPE_3D_TEXTURE:
                    summary_writer.add_scalar("lr_basis", lr_basis, global_step=gstep_id)
                if grid.use_background:
                    summary_writer.add_scalar("lr_sigma_bg", lr_sigma_bg, global_step=gstep_id)
                    summary_writer.add_scalar("lr_color_bg", lr_color_bg, global_step=gstep_id)

                if args.weight_decay_sh < 1.0:
                    grid.sh_data.data *= args.weight_decay_sigma
                if args.weight_decay_sigma < 1.0:
                    grid.density_data.data *= args.weight_decay_sh

                if WANDB_ON:
                    wandb.log({
                        "lr_sh": lr_sh,
                        "lr_sigma": lr_sigma,
                        "lr_basis": lr_basis,

                    }, step=gstep_id)

            #  # For outputting the % sparsity of the gradient
            #  indexer = grid.sparse_sh_grad_indexer
            #  if indexer is not None:
            #      if indexer.dtype == torch.bool:
            #          nz = torch.count_nonzero(indexer)
            #      else:
            #          nz = indexer.size()
            #      with open(os.path.join(args.train_dir, 'grad_sparsity.txt'), 'a') as sparsity_file:
            #          sparsity_file.write(f"{gstep_id} {nz}\n")

            # Apply TV/Sparsity regularizers
            if args.lambda_tv > 0.0:
                #  with Timing("tv_inpl"):
                grid.inplace_tv_grad(grid.density_data.grad,
                        scaling=args.lambda_tv,
                        sparse_frac=args.tv_sparsity,
                        logalpha=args.tv_logalpha,
                        ndc_coeffs=dset.ndc_coeffs,
                        contiguous=args.tv_contiguous)
            if args.lambda_tv_sh > 0.0:
                #  with Timing("tv_color_inpl"):
                grid.inplace_tv_color_grad(grid.sh_data.grad,
                        scaling=args.lambda_tv_sh,
                        sparse_frac=args.tv_sh_sparsity,
                        ndc_coeffs=dset.ndc_coeffs,
                        contiguous=args.tv_contiguous)
            if args.lambda_tv_lumisphere > 0.0:
                grid.inplace_tv_lumisphere_grad(grid.sh_data.grad,
                        scaling=args.lambda_tv_lumisphere,
                        dir_factor=args.tv_lumisphere_dir_factor,
                        sparse_frac=args.tv_lumisphere_sparsity,
                        ndc_coeffs=dset.ndc_coeffs)
            if args.lambda_l2_sh > 0.0:
                grid.inplace_l2_color_grad(grid.sh_data.grad,
                        scaling=args.lambda_l2_sh)
            if grid.use_background and (args.lambda_tv_background_sigma > 0.0 or args.lambda_tv_background_color > 0.0):
                grid.inplace_tv_background_grad(grid.background_data.grad,
                        scaling=args.lambda_tv_background_color,
                        scaling_density=args.lambda_tv_background_sigma,
                        sparse_frac=args.tv_background_sparsity,
                        contiguous=args.tv_contiguous)
            if args.lambda_tv_basis > 0.0:
                tv_basis = grid.tv_basis()
                loss_tv_basis = tv_basis * args.lambda_tv_basis
                loss_tv_basis.backward()
            #  print('nz density', torch.count_nonzero(grid.sparse_grad_indexer).item(),
            #        ' sh', torch.count_nonzero(grid.sparse_sh_grad_indexer).item())

            # Manual SGD/rmsprop step
            if gstep_id >= args.lr_fg_begin_step:
                grid.optim_density_step(lr_sigma, beta=args.rms_beta, optim=args.sigma_optim)
                grid.optim_sh_step(lr_sh, beta=args.rms_beta, optim=args.sh_optim)
            if grid.use_background:
                grid.optim_background_step(lr_sigma_bg, lr_color_bg, beta=args.rms_beta, optim=args.bg_optim)
            if gstep_id >= args.lr_basis_begin_step:
                if grid.basis_type == svox2.BASIS_TYPE_3D_TEXTURE:
                    grid.optim_basis_step(lr_basis, beta=args.rms_beta, optim=args.basis_optim)
                elif grid.basis_type == svox2.BASIS_TYPE_MLP:
                    optim_basis_mlp.step()
                    optim_basis_mlp.zero_grad()
    train_step()
    gc.collect()
    gstep_id_base += batches_per_epoch

    #  ckpt_path = path.join(args.train_dir, f'ckpt_{epoch_id:05d}.npz')
    # Overwrite prev checkpoints since they are very huge
    if (args.save_every > 0 and (epoch_id + 1) % max(
            factor, args.save_every) == 0 and not args.tune_mode):
        print('Saving', ckpt_path)
        grid.save(ckpt_path)
        grid.save_voxels_to_dict(ckpt_path)

    if (gstep_id_base - last_upsamp_step) >= args.upsamp_every:
        last_upsamp_step = gstep_id_base
        if reso_id < len(reso_list) - 1:
            print('* Upsampling from', reso_list[reso_id], 'to', reso_list[reso_id + 1])
            if args.tv_early_only > 0:
                print('turning off TV regularization')
                args.lambda_tv = 0.0
                args.lambda_tv_sh = 0.0
            elif args.tv_decay != 1.0:
                args.lambda_tv *= args.tv_decay
                args.lambda_tv_sh *= args.tv_decay

            reso_id += 1
            use_sparsify = True
            z_reso = reso_list[reso_id] if isinstance(reso_list[reso_id], int) else reso_list[reso_id][2]
            grid.resample(reso=reso_list[reso_id],
                    sigma_thresh=args.density_thresh,
                    weight_thresh=args.weight_thresh / z_reso if use_sparsify else 0.0,
                    dilate=2, #use_sparsify,
                    cameras=resample_cameras if args.thresh_type == 'weight' else None,
                    max_elements=args.max_grid_elements)
            #optimal_density = get_links(pc_points)
            #neg_optimal_density = get_links(negative_points)

            if grid.use_background and reso_id <= 1:
                grid.sparsify_background(args.background_density_thresh)

            if args.upsample_density_add:
                grid.density_data.data[:] += args.upsample_density_add

        if factor > 1 and reso_id < len(reso_list) - 1:
            print('* Using higher resolution images due to large grid; new factor', factor)
            factor //= 2
            dset.gen_rays(factor=factor)
            dset.shuffle_rays()

    if gstep_id_base >= args.n_iters:
        print('* Final eval and save')
        eval_step()
        global_stop_time = datetime.now()
        secs = (global_stop_time - global_start_time).total_seconds()
        timings_file = open(os.path.join(args.train_dir, 'time_mins.txt'), 'a')
        timings_file.write(f"{secs / 60}\n")
        if not args.tune_nosave:
            grid.save(ckpt_path)
            grid.save_voxels_to_dict(ckpt_path)
        break


if WANDB_ON:
    dset = datasets[args.dataset_type](args.data_dir, split="test",
                                **config_util.build_data_options(args))
    import lpips
    lpips_vgg = lpips.LPIPS(net="vgg").eval().to(device)

    config_util.setup_render_opts(grid.opt, args)

    # NOTE: no_grad enables the fast image-level rendering kernel for cuvol backend only
    # other backends will manually generate rays per frame (slow)
    with torch.no_grad():
        n_images = dset.n_images
        img_eval_interval = 1
        avg_psnr = 0.0
        avg_ssim = 0.0
        avg_lpips = 0.0
        n_images_gen = 0
        c2ws = dset.c2w.to(device=device)

        frames = []
        for img_id in tqdm(range(0, n_images, img_eval_interval)):
            dset_h, dset_w = dset.get_image_size(img_id)
            im_size = dset_h * dset_w
            w = dset_w
            h = dset_h

            cam = svox2.Camera(c2ws[img_id],
                            dset.intrins.get('fx', img_id),
                            dset.intrins.get('fy', img_id),
                            dset.intrins.get('cx', img_id) + (w - dset_w) * 0.5,
                            dset.intrins.get('cy', img_id) + (h - dset_h) * 0.5,
                            w, h,
                            ndc_coeffs=dset.ndc_coeffs)
            im = grid.volume_render_image(cam, use_kernel=True, return_raylen=False)
            im.clamp_(0.0, 1.0)

            # not render path
            im_gt = dset.gt[img_id].to(device=device)
            mse = (im - im_gt) ** 2
            mse_num : float = mse.mean().item()
            psnr = -10.0 * math.log10(mse_num)
            avg_psnr += psnr
            #timing
            ssim = compute_ssim(im_gt, im).item()
            avg_ssim += ssim

            lpips_i = lpips_vgg(im_gt.permute([2, 0, 1]).contiguous(),
                    im.permute([2, 0, 1]).contiguous(), normalize=True).item()
            avg_lpips += lpips_i
            print(img_id, 'PSNR', psnr, 'SSIM', ssim, 'LPIPS', lpips_i)

            im = im.cpu().numpy()

            im_gt = dset.gt[img_id].numpy()
            im = np.concatenate([im_gt, im], axis=1)

            im = (im * 255).astype(np.uint8)
            frames.append(im)
            im = None
            n_images_gen += 1

        print('AVERAGES')

        avg_psnr /= n_images_gen
        avg_ssim /= n_images_gen
        avg_lpips /= n_images_gen

        vid = np.transpose(np.array(frames), (0, 3, 1, 2))

        vid_path = os.path.join("/root/videos/", wandb.run.name + ".mp4")
        print(f"saving the video {vid_path}")
        imageio.mimwrite(vid_path, frames, fps=24, macro_block_size=8)

        wandb.log({
            'final/psnr': avg_psnr,
            'final/ssim': avg_ssim,
            'final/lpips': avg_lpips,
            'final/video': wandb.Video(vid_path, format="mp4")
        })