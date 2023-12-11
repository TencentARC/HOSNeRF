# ------------------------------------------------------------------------------------
# HOSNeRF
# Copyright (c) 2023 Show Lab, National University of Singapore. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# Modified from NeRF-Factory (https://github.com/kakaobrain/nerf-factory)
# Copyright (c) 2022 POSTECH, KAIST, Kakao Brain Corp. All Rights Reserved.
# ------------------------------------------------------------------------------------

import os
from typing import *

import gin
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from PIL import Image
from tqdm import tqdm
import skimage
import json

from core.nets import create_network
import src.model.mipnerf360.helper as helper
import utils.store_image as store_image
from src.model.interface import LitModel
from third_parties.lpips import LPIPS
from core.utils.network_util import set_requires_grad
from core.train import create_optimizer
from core.data import create_dataloader
from core.utils.train_util import cpu_data_to_gpu
from core.utils.image_util import tile_images, to_8b_image

img2mse = lambda x, y : torch.mean((x - y) ** 2)
img2l1 = lambda x, y : torch.mean(torch.abs(x-y))
to8b = lambda x : (255.*np.clip(x,0.,1.)).astype(np.uint8)

EXCLUDE_KEYS_TO_GPU = ['frame_name', 'img_width', 'img_height']

def _unpack_imgs(rgbs, patch_masks, bgcolor, targets, div_indices):

    N_patch = len(div_indices) - 1
    assert patch_masks.shape[0] == N_patch
    assert targets.shape[0] == N_patch

    [b, w, h] = patch_masks.shape
    patch_imgs = rgbs.reshape([b, w, h, 3])

    return patch_imgs

def get_customized_lr_names(cfg):
    return [k[3:] for k in cfg.train.keys() if k.startswith('lr_')]   

def to_homogeneous(pts):
    if isinstance(pts, torch.Tensor):
        return torch.cat([pts, torch.ones_like(pts[..., 0:1])], axis=-1)
    elif isinstance(pts, np.ndarray):
        return np.concatenate([pts, np.ones_like(pts[..., 0:1])], axis=-1)    

def img2mae(x, y, weights=None, M=None):
    if weights == None:
        if M == None:
            return torch.mean(torch.abs(x - y))
        else:
            return torch.sum(torch.abs(x - y) * M) / (torch.sum(M) + 1e-8) / x.shape[-1]   
    else:
        if M == None:
            return torch.mean(torch.abs(x - y) * weights[..., None])
        else:
            return torch.sum(torch.abs(x - y) * weights[..., None] * M) / (torch.sum(M) + 1e-8) / x.shape[-1]   

def _raw2outputs(raw, z_vals, rays_d, pts_mask=None, bgcolor=None):
    def _raw2alpha(raw, dists):
        return 1.0 - torch.exp(-raw*dists)

    dists = z_vals[...,1:] - z_vals[...,:-1]
    infinity_dists = torch.Tensor([1e10])
    infinity_dists = infinity_dists.expand(dists[...,:1].shape).to(dists)
    dists = torch.cat([dists, infinity_dists], dim=-1) 
    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

    rgb = raw[...,:3]  # [N_rays, N_samples, 3]
    alpha = _raw2alpha(raw[...,3], dists)  # [N_rays, N_samples]
    if pts_mask != None:
        alpha = alpha * pts_mask[..., 0]

    weights = alpha * torch.cumprod(
        torch.cat([torch.ones((alpha.shape[0], 1)).to(alpha), 
                    1.-alpha + 1e-10], dim=-1), dim=-1)[:, :-1]
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    acc_map = torch.sum(weights, -1)

    if bgcolor != None:
        rgb_map = rgb_map + (1.-acc_map[...,None]) * bgcolor[None, :]/255.

    return rgb_map, acc_map, weights, depth_map

def psnr_metric(img_pred, img_gt):
    ''' Caculate psnr metric
        Args:
            img_pred: ndarray, W*H*3, range 0-1
            img_gt: ndarray, W*H*3, range 0-1

        Returns:
            psnr metric: scalar
    '''
    mse = np.mean((img_pred - img_gt) ** 2)
    psnr = -10 * np.log(mse) / np.log(10)
    return psnr.item()    

@gin.configurable()
class MipNeRF360MLP(nn.Module):
    def __init__(
        self,
        basedir,
        netdepth: int = 8,
        netwidth: int = 256,
        bottleneck_width: int = 256,
        netdepth_condition: int = 1,
        netwidth_condition: int = 128,
        min_deg_point: int = 0,
        max_deg_point: int = 12,
        skip_layer: int = 4,
        skip_layer_dir: int = 4,
        num_rgb_channels: int = 3,
        num_density_channels: int = 1,
        deg_view: int = 4,
        bottleneck_noise: float = 0.0,
        density_bias: float = -1.0,
        density_noise: float = 0.0,
        rgb_premultiplier: float = 1.0,
        rgb_bias: float = 0.0,
        rgb_padding: float = 0.001,
        basis_shape: str = "icosahedron",
        basis_subdivision: int = 2,
        disable_rgb: bool = False,
    ):

        for name, value in vars().items():
            if name not in ["self", "__class__"]:
                setattr(self, name, value)

        super(MipNeRF360MLP, self).__init__()

        self.net_activation = nn.ReLU()
        self.density_activation = nn.Softplus()
        self.rgb_activation = nn.Sigmoid()
        self.warp_fn = helper.contract
        self.register_buffer(
            "pos_basis_t", helper.generate_basis(basis_shape, basis_subdivision)
        )

        pos_size = ((max_deg_point - min_deg_point) * 2) * self.pos_basis_t.shape[-1]
        view_pos_size = (deg_view * 2 + 1) * 3

        # gt transition times
        transitions_times = []
        embedding_size = 64  
        pos_size = pos_size + embedding_size  
        if os.path.exists(os.path.join(basedir, "transitions_times.json")):
            with open(os.path.join(basedir, "transitions_times.json"), 'r') as f:
                frame_infos = json.load(f)    
            for frame_base_name in frame_infos:
                time_info = frame_infos[frame_base_name] 
                transitions_times.append(np.array(time_info['time'], dtype=np.float32))
            self.transitions_times = np.stack(transitions_times, axis=0) 
            self.bkgd_stateembeds = nn.ParameterList([nn.Parameter(torch.randn(embedding_size), requires_grad=True) for i in range(self.transitions_times.shape[0]+1)])                 
        else:
            self.bkgd_stateembeds = nn.ParameterList([nn.Parameter(torch.randn(embedding_size), requires_grad=True)])

        module = nn.Linear(pos_size, netwidth)
        init.kaiming_uniform_(module.weight)
        pts_linear = [module]

        for idx in range(netdepth - 1):
            if idx % skip_layer == 0 and idx > 0:
                module = nn.Linear(netwidth + pos_size, netwidth)
            else:
                module = nn.Linear(netwidth, netwidth)
            init.kaiming_uniform_(module.weight)
            pts_linear.append(module)

        self.pts_linear = nn.ModuleList(pts_linear)
        self.density_layer = nn.Linear(netwidth, num_density_channels)
        init.kaiming_uniform_(self.density_layer.weight)

        if not disable_rgb:
            self.bottleneck_layer = nn.Linear(netwidth, bottleneck_width)
            layer = nn.Linear(bottleneck_width + view_pos_size, netwidth_condition)
            init.kaiming_uniform_(layer.weight)
            views_linear = [layer]
            for idx in range(netdepth_condition - 1):
                if idx % skip_layer_dir == 0 and idx > 0:
                    layer = nn.Linear(
                        netwidth_condition + view_pos_size, netwidth_condition
                    )
                else:
                    layer = nn.Linear(netwidth_condition, netwidth_condition)
                init.kaiming_uniform_(layer.weight)
                views_linear.append(layer)
            self.views_linear = nn.ModuleList(views_linear)

            self.rgb_layer = nn.Linear(netwidth_condition, num_rgb_channels)

            init.kaiming_uniform_(self.bottleneck_layer.weight)
            init.kaiming_uniform_(self.rgb_layer.weight)

        self.dir_enc_fn = helper.pos_enc

    def predict_density(self, means, covs, randomized, is_train, time):

        means, covs = self.warp_fn(means, covs, is_train)

        lifted_means, lifted_vars = helper.lift_and_diagonalize(
            means, covs, self.pos_basis_t
        )
        x = helper.integrated_pos_enc(
            lifted_means, lifted_vars, self.min_deg_point, self.max_deg_point
        )
      
        batch_size, ray_size, _ = x.shape
        eps = 1e-5
        if len(self.bkgd_stateembeds) == 1:
            embed_state_ = self.bkgd_stateembeds[0]         

        if len(self.bkgd_stateembeds) == 2:
            if time < self.transitions_times[0]-eps:
                embed_state_ = self.bkgd_stateembeds[0]
            else:
                embed_state_ = self.bkgd_stateembeds[1]                      

        if len(self.bkgd_stateembeds) == 3:
            if time < self.transitions_times[0]-eps:
                embed_state_ = self.bkgd_stateembeds[0]
            elif time <= self.transitions_times[1]+eps:
                embed_state_ = self.bkgd_stateembeds[1]
            else:
                embed_state_ = self.bkgd_stateembeds[2]    

        if len(self.bkgd_stateembeds) == 4:
            if time < self.transitions_times[0]-eps:
                embed_state_ = self.bkgd_stateembeds[0]
            elif time <= self.transitions_times[1]+eps:
                embed_state_ = self.bkgd_stateembeds[1]
            elif time <= self.transitions_times[2]+eps:
                embed_state_ = self.bkgd_stateembeds[2]
            else:
                embed_state_ = self.bkgd_stateembeds[3]  

        if len(self.bkgd_stateembeds) == 5:
            if time < self.transitions_times[0]-eps:
                embed_state_ = self.bkgd_stateembeds[0]
            elif time <= self.transitions_times[1]+eps:
                embed_state_ = self.bkgd_stateembeds[1]
            elif time <= self.transitions_times[2]+eps:
                embed_state_ = self.bkgd_stateembeds[2]
            elif time <= self.transitions_times[3]+eps:
                embed_state_ = self.bkgd_stateembeds[3]
            else:
                embed_state_ = self.bkgd_stateembeds[4]     

        if len(self.bkgd_stateembeds) == 6:
            if time < self.transitions_times[0]-eps:
                embed_state_ = self.bkgd_stateembeds[0]
            elif time <= self.transitions_times[1]+eps:
                embed_state_ = self.bkgd_stateembeds[1]
            elif time <= self.transitions_times[2]+eps:
                embed_state_ = self.bkgd_stateembeds[2]
            elif time <= self.transitions_times[3]+eps:
                embed_state_ = self.bkgd_stateembeds[3]
            elif time <= self.transitions_times[4]+eps:
                embed_state_ = self.bkgd_stateembeds[4]
            else:
                embed_state_ = self.bkgd_stateembeds[5]

        if len(self.bkgd_stateembeds) == 7:
            if time < self.transitions_times[0]-eps:
                embed_state_ = self.bkgd_stateembeds[0]
            elif time <= self.transitions_times[1]+eps:
                embed_state_ = self.bkgd_stateembeds[1]
            elif time <= self.transitions_times[2]+eps:
                embed_state_ = self.bkgd_stateembeds[2]
            elif time <= self.transitions_times[3]+eps:
                embed_state_ = self.bkgd_stateembeds[3]
            elif time <= self.transitions_times[4]+eps:
                embed_state_ = self.bkgd_stateembeds[4]
            elif time <= self.transitions_times[5]+eps:
                embed_state_ = self.bkgd_stateembeds[5]
            else:
                embed_state_ = self.bkgd_stateembeds[6]                            

        embed_state = embed_state_.repeat(batch_size, ray_size, 1)
        x = torch.cat([x, embed_state], dim=-1)   

        inputs = x
        for idx in range(self.netdepth):
            x = self.pts_linear[idx](x)
            x = self.net_activation(x)
            if idx % self.skip_layer == 0 and idx > 0:
                x = torch.cat([x, inputs], dim=-1)

        raw_density = self.density_layer(x)[..., 0]
        if self.density_noise > 0.0 and randomized:
            raw_density += self.density_noise * torch.rand_like(raw_density)

        return raw_density, x

    def forward(self, gaussians, viewdirs, randomized, is_train, time):

        means, covs = gaussians

        raw_density, x = self.predict_density(means, covs, randomized, is_train, time)
        density = self.density_activation(raw_density + self.density_bias)

        if self.disable_rgb:
            rgb = torch.zeros_like(means)
            return {
                "density": density,
                "rgb": rgb,
            }

        bottleneck = self.bottleneck_layer(x)
        if self.bottleneck_noise > 0.0 and randomized:
            bottleneck += torch.rand_like(bottleneck) * self.bottleneck_noise
        x = [bottleneck]

        dir_enc = self.dir_enc_fn(viewdirs, 0, self.deg_view, True)
        dir_enc = torch.broadcast_to(
            dir_enc[..., None, :], bottleneck.shape[:-1] + (dir_enc.shape[-1],)
        )
        x.append(dir_enc)
        x = torch.cat(x, dim=-1)

        inputs = x
        for idx in range(self.netdepth_condition):
            x = self.views_linear[idx](x)
            x = self.net_activation(x)
            if idx % self.skip_layer_dir == 0 and idx > 0:
                x = torch.cat([x, inputs], dim=-1)

        x = self.rgb_layer(x)
        rgb = self.rgb_activation(self.rgb_premultiplier * x + self.rgb_bias)
        rgb = rgb * (1 + 2 * self.rgb_padding) - self.rgb_padding

        return {
            "density": density,
            "rgb": rgb,
        }


@gin.configurable()
class NeRFMLP(MipNeRF360MLP):
    def __init__(
        self,
        basedir,
        netdepth: int = 8,
        netwidth: int = 1024,
    ):
        super(NeRFMLP, self).__init__(basedir, netdepth=netdepth, netwidth=netwidth)


@gin.configurable()
class PropMLP(MipNeRF360MLP):
    def __init__(
        self,
        basedir,
        netdepth: int = 4,
        netwidth: int = 256,
    ):
        super(PropMLP, self).__init__(
            basedir, netdepth=netdepth, netwidth=netwidth, disable_rgb=True
        )


@gin.configurable()
class MipNeRF360(nn.Module):
    def __init__(
        self,
        basedir,
        num_prop_samples: int = 64,
        num_nerf_samples: int = 32,
        num_levels: int = 3,
        bg_intensity_range: Tuple[float] = (1.0, 1.0),
        anneal_slope: int = 10,
        stop_level_grad: bool = True,
        use_viewdirs: bool = True,
        ray_shape: str = "cone",
        disable_integration: bool = False,
        single_jitter: bool = True,
        dilation_multiplier: float = 0.5,
        dilation_bias: float = 0.0025,
        num_glo_features: int = 0,
        num_glo_embeddings: int = 1000,
        learned_exposure_scaling: bool = False,
        near_anneal_rate: Optional[float] = None,
        near_anneal_init: float = 0.95,
        single_mlp: bool = False,
        resample_padding: float = 0.0,
        use_gpu_resampling: bool = False,
        opaque_background: bool = False,
    ):

        for name, value in vars().items():
            if name not in ["self", "__class__"]:
                setattr(self, name, value)

        super(MipNeRF360, self).__init__()
        self.mlps = nn.ModuleList(
            [PropMLP(basedir) for _ in range(num_levels - 1)]
            + [
                NeRFMLP(basedir),
            ]
        )

    def forward(self, batch, train_frac, randomized, is_train, near, far):

        bsz, _ = batch["rays_o"].shape
        device = batch["rays_o"].device
        time = batch["times"]

        _, s_to_t = helper.construct_ray_warps(near, far)
        if self.near_anneal_rate is None:
            init_s_near = 0.0
        else:
            init_s_near = 1 - train_frac / self.near_anneal_rate
            init_s_near = max(min(init_s_near, 1), 0)
        init_s_far = 1.0

        sdist = torch.cat(
            [
                torch.full((bsz, 1), init_s_near, device=device),
                torch.full((bsz, 1), init_s_far, device=device),
            ],
            dim=-1,
        )

        weights = torch.ones(bsz, 1, device=device)
        prod_num_samples = 1

        ray_history = []
        renderings = []

        for i_level in range(self.num_levels):
            is_prop = i_level < (self.num_levels - 1)
            num_samples = self.num_prop_samples if is_prop else self.num_nerf_samples

            dilation = (
                self.dilation_bias
                + self.dilation_multiplier
                * (init_s_far - init_s_near)
                / prod_num_samples
            )

            prod_num_samples *= num_samples

            use_dilation = self.dilation_bias > 0 or self.dilation_multiplier > 0

            if i_level > 0 and use_dilation:
                sdist, weights = helper.max_dilate_weights(
                    sdist,
                    weights,
                    dilation,
                    domain=(init_s_near, init_s_far),
                    renormalize=True,
                )
                sdist = sdist[..., 1:-1]
                weights = weights[..., 1:-1]

            if self.anneal_slope > 0:
                bias = lambda x, s: (s * x) / ((s - 1) * x + 1)
                anneal = bias(train_frac, self.anneal_slope)
            else:
                anneal = 1.0

            logits_resample = torch.where(
                sdist[..., 1:] > sdist[..., :-1],
                anneal * torch.log(weights + self.resample_padding),
                torch.full_like(weights, -torch.inf),
            )

            sdist = helper.sample_intervals(
                randomized,
                sdist,
                logits_resample,
                num_samples,
                single_jitter=self.single_jitter,
                domain=(init_s_near, init_s_far),
            )

            if self.stop_level_grad:
                sdist = sdist.detach()

            tdist = s_to_t(sdist)

            gaussians = helper.cast_rays(
                tdist,
                batch["rays_o"],
                batch["rays_d"],
                batch["radii"],
                self.ray_shape,
                diag=False,
            )

            if self.disable_integration:
                gaussians = (gaussians[0], torch.zeros_like(gaussians[1]))

            ray_results = self.mlps[i_level](
                gaussians, batch["viewdirs"], randomized, is_train, time
            )

            weights = helper.compute_alpha_weights(
                ray_results["density"],
                tdist,
                batch["rays_d"],
                opaque_background=self.opaque_background,
            )[0]

            if self.bg_intensity_range[0] == self.bg_intensity_range[1]:
                bg_rgbs = self.bg_intensity_range[0]
            elif not randomized:
                bg_rgbs = (
                    self.bg_intensity_range[0] + self.bg_intensity_range[1]
                ) / 2.0
            else:
                bg_rgbs = (
                    torch.rand(3)
                    * (self.bg_intensity_range[1] - self.bg_intensity_range[0])
                    + self.bg_intensity_range[0]
                )

            ray_results["sdist"] = sdist
            ray_results["tdist"] = tdist
            ray_results["weights"] = weights

            ray_history.append(ray_results)

        return renderings, ray_history


class LitMipNeRF360(LitModel):
    def __init__(
        self,
        cfg,
        basedir,
        lr_init: float = 2.0e-3,
        lr_final: float = 2.0e-5,
        lr_delay_steps: int = 512,
        lr_delay_mult: float = 0.01,
        data_loss_mult: float = 1.0,
        interlevel_loss_mult: float = 1.0,
        distortion_loss_mult: float = 0.01,
        use_multiscale: bool = False,
        charb_padding: float = 0.001,
    ):

        for name, value in vars().items():
            if name not in ["self", "__class__"]:
                setattr(self, name, value)

        super(LitMipNeRF360, self).__init__()

        # gt transition times
        transitions_times = []
        with open(os.path.join(basedir, "transitions_times.json"), 'r') as f:
            frame_infos = json.load(f)    
        for frame_base_name in frame_infos:
            time_info = frame_infos[frame_base_name] 
            transitions_times.append(np.array(time_info['time'], dtype=np.float32))
        self.transitions_times = np.stack(transitions_times, axis=0)          

        self.model = MipNeRF360(basedir)
        self.human = create_network(cfg)
        self.cfg = cfg
        self.prog_dataloader = create_dataloader(self.cfg, data_type='progress')
        self.movement_dataloader = create_dataloader(self.cfg, data_type='movement')
        self.test_dataloader = create_dataloader(self.cfg, data_type='test')
        self.freeview_dataloader = create_dataloader(self.cfg, data_type='freeview')
        self.tpose_dataloader = create_dataloader(self.cfg, data_type='tpose')
        if "lpips" in cfg.train.lossweights.keys():
            self.lpips_func = LPIPS(net='vgg')
            set_requires_grad(self.lpips_func, requires_grad=False)


    def setup(self, stage):
        self.near_bkg = self.trainer.datamodule.near_bkg
        self.far_bkg = self.trainer.datamodule.far_bkg

    def test_tpose(self, time):
        self.progress_begin()

        print('Evaluate Tpose Images ...')

        images = []
        psnrs, ssims, lpipss = [], [], []
        is_empty_img = False
        idx_tpose = 0
        for _, batch in enumerate(tqdm(self.tpose_dataloader)):

            # only access the first batch as we process one image one time
            for k, v in batch.items():
                batch[k] = v[0]
              
            width = batch['img_width']
            height = batch['img_height']
            ray_mask = batch['ray_mask']

            rendered = np.full(
                        (height * width, 3), np.array(self.cfg.bgcolor)/255., 
                        dtype='float32')

            batch['time'] = torch.tensor(time)
            batch['iter_val'] = torch.full((1,), self.trainer.global_step)
            data = cpu_data_to_gpu(
                    batch, exclude_keys=EXCLUDE_KEYS_TO_GPU + ['target_rgbs'])
            with torch.no_grad():
                net_output = self.human(**data)

            human_out = torch.cat([net_output['human_rgb'], net_output['human_density'][..., None]], -1)

            rgb_, alpha_, _, depth_ = _raw2outputs(human_out, net_output['z_vals'], net_output['rays_d'], net_output['pts_mask'][..., None], net_output['bgcolor'])  # net_output['bgcolor'] 

            rgb = rgb_.data.to("cpu").numpy()

            rendered[ray_mask] = rgb

            image_vis = to_8b_image(rendered.reshape((height, width, -1)))
            
            tpose_dir = self.logdir + "/tpose_vis/" + "time_{:06}".format(time)
            if not os.path.exists(tpose_dir):
                os.makedirs(tpose_dir)
            Image.fromarray(image_vis).save(
                os.path.join(tpose_dir, "image-{:05}.jpg".format(idx_tpose)))

            idx_tpose += 1
            
        self.progress_end()

        return is_empty_img                    

    def test_step(self, batch, batch_idx):

        self.test_metrics()
        self.allimgs_metrics()
        self.free_view()
        if len(self.transitions_times) > 0:
            for i in range(len(self.transitions_times)):
                if i == 0:
                    time = (0.0 + self.transitions_times[i]) / 2
                else:
                    time = (self.transitions_times[i-1] + self.transitions_times[i]) / 2
                self.test_tpose(time=time)
            time = (self.transitions_times[-1] + 1.0) / 2
            self.test_tpose(time=time)
        else:
            self.test_tpose(time=0.5)             

    def progress_begin(self):
        self.model.eval()
        self.human.eval()
        self.cfg.perturb = 0.  

    def progress_end(self):
        self.model.train()
        self.human.train()
        self.cfg.perturb = self.cfg.train.perturb          

    def test_begin(self):
        self.model.eval()
        self.human.eval()
        self.cfg.perturb = 0.  

    def test_end(self):
        self.model.train()
        self.human.train()
        self.cfg.perturb = self.cfg.train.perturb               

    def progress(self):
        self.progress_begin()

        print('Evaluate Progress Images ...')

        is_empty_img = False
        psnrs, ssims, lpipss = [], [], []

        for _, batch in enumerate(tqdm(self.prog_dataloader)):
            
            # only access the first batch as we process one image one time
            for k, v in batch.items():
                batch[k] = v[0]
            width = batch['img_width']
            height = batch['img_height']
            ray_mask = batch['ray_mask']
            ray_mask_bkg = batch['ray_mask_bkg']
            frame_name = batch['frame_name']

            rendered = np.full(
                        (height * width, 3), np.array(self.cfg.bgcolor)/255., 
                        dtype='float32')
            truth = np.full(
                        (height * width, 3), np.array(self.cfg.bgcolor)/255., 
                        dtype='float32')

            batch['iter_val'] = torch.full((1,), self.trainer.global_step)
            batch = cpu_data_to_gpu(
                    batch, exclude_keys=EXCLUDE_KEYS_TO_GPU)
            with torch.no_grad():

                rgb = []
                for i in range(0, batch['rays_o_bkg'].shape[0], self.cfg.chunk_bkg):
                    batch_bkg = {
                        "rays_o": batch['rays_o_bkg'][i:i+self.cfg.chunk_bkg],
                        "rays_d": batch['rays_d_bkg'][i:i+self.cfg.chunk_bkg],
                        "viewdirs": batch['viewdirs_bkg'][i:i+self.cfg.chunk_bkg],
                        "radii": batch['radii'][i:i+self.cfg.chunk_bkg],
                        "times": batch['time']
                    }
                    train_frac = 1.0
                    rendered_results, ray_history = self.model(
                        batch_bkg, train_frac, True, True, self.near_bkg, self.far_bkg
                    )

                    batch_human = {
                        "rays": batch['rays'][:, i:i+self.cfg.chunk_bkg, :],
                        "near": batch['near'][i:i+self.cfg.chunk_bkg, :],
                        "far": batch['far'][i:i+self.cfg.chunk_bkg, :],
                        "bgcolor": batch['bgcolor'],
                        "target_rgbs": batch['target_rgbs'][i:i+self.cfg.chunk_bkg, :],
                        "dst_Rs": batch['dst_Rs'],
                        "dst_Ts": batch['dst_Ts'],
                        "cnl_gtfms": batch['cnl_gtfms'],
                        "canonical_joints": batch['canonical_joints'],
                        "motion_weights_priors": batch['motion_weights_priors'],
                        "cnl_bbox_min_xyz": batch['cnl_bbox_min_xyz'],
                        "cnl_bbox_max_xyz": batch['cnl_bbox_max_xyz'],
                        "cnl_bbox_scale_xyz": batch['cnl_bbox_scale_xyz'],
                        "dst_posevec": batch['dst_posevec'],
                        "iter_val": batch['iter_val'],
                        "time": batch['time'],
                        "is_train": batch['is_train']
                    }                

                    net_output = self.human(**batch_human)
                    scaleworld_pts = torch.einsum('ji, bni->bnj', batch['newsmpl_to_scale_world'], to_homogeneous(net_output['newsmpl_pts']))[..., :3]

                    # to handle the cases where the rays_d is too small. In this case, only one dimension is used to compute the z_vals.
                    if torch.any(torch.abs(batch_bkg["rays_d"][..., None, :]) < 1e-5):
                        idx = torch.abs(batch_bkg["rays_d"][..., None, :]) > 1e-5
                        idx_new = torch.zeros_like(batch_bkg["rays_d"][..., None, :], dtype=torch.bool).cuda()
                        idx_new[..., 0] = idx[..., 0]
                        if torch.sum(idx_new) < idx_new.shape[0]:
                            idx_col = idx_new[..., 0] == False
                            idx_new[..., 1][idx_col] = idx[..., 1][idx_col]
                            if torch.sum(idx_new) < idx_new.shape[0]:
                                idx_col = idx_new[..., 1] == False
                                idx_new[..., 2][idx_col] = idx[..., 2][idx_col]
                        if torch.sum(idx_new) == idx_new.shape[0]:
                            idx_new_pts = idx_new.repeat(1, scaleworld_pts.shape[1], 1)
                            z_vals_human_all = (scaleworld_pts - batch_bkg["rays_o"][..., None, :])[idx_new_pts].reshape(scaleworld_pts.shape[:2]) / (batch_bkg["rays_d"][..., None, :]+1e-10)[idx_new][..., None]
                        else:
                            print("There are points with very small rays_d!")
                            import pdb
                            pdb.set_trace()
                    else:
                        z_vals_human_ = (scaleworld_pts - batch_bkg["rays_o"][..., None, :]) / (batch_bkg["rays_d"][..., None, :]+1e-10)
                        z_vals_human_all = torch.mean(z_vals_human_, dim=-1)

                    thre_fg = 5e-3     # define the threshold for human fg.
                    pts_mask_human_all = net_output['pts_mask']

                    val = torch.sum(pts_mask_human_all, dim=-1)
                    idx_fg = val > thre_fg    # select the fg human, and leave the bg
                    idx_bg = ~idx_fg

                    device = pts_mask_human_all.device
                    rgb_batch = torch.full((pts_mask_human_all.shape[0], 3), 0, dtype=torch.float32, device=device)

                    z_vals_bkg = ray_history[-1]['tdist'][..., :-1][idx_fg]
                    z_vals_bkg_onlybg = ray_history[-1]['tdist'][..., :-1][idx_bg]
                    z_vals_human = z_vals_human_all[idx_fg]
                    human_out = torch.cat([net_output['human_rgb'][idx_fg], net_output['human_density'][..., None][idx_fg]], -1)

                    bkg_out_all = torch.cat([ray_history[-1]['rgb'], ray_history[-1]['density'][..., None]], -1)
                    bkg_out = bkg_out_all[idx_fg]
                    bkg_out_onlybg = bkg_out_all[idx_bg]

                    total_zvals, total_order = torch.sort(torch.cat([z_vals_bkg, z_vals_human], -1), -1)
                    total_out = torch.cat([bkg_out, human_out], 1)
                    _b, _n, _c = total_out.shape
                    total_out = total_out[
                        torch.arange(_b).view(_b, 1, 1).repeat(1, _n, _c),
                        total_order.view(_b, _n, 1).repeat(1, 1, _c),
                        torch.arange(_c).view(1, 1, _c).repeat(_b, _n, 1),
                    ]     

                    pts_mask_human = pts_mask_human_all[idx_fg]

                    pts_mask_bkg = torch.ones_like(z_vals_bkg).cuda()
                    pts_mask = torch.cat([pts_mask_bkg, pts_mask_human], -1)[..., None]
                    _b, _n, _c = pts_mask.shape
                    pts_mask = pts_mask[
                        torch.arange(_b).view(_b, 1, 1).repeat(1, _n, _c),
                        total_order.view(_b, _n, 1).repeat(1, 1, _c),
                        torch.arange(_c).view(1, 1, _c).repeat(_b, _n, 1),
                    ]   

                    rgb_onlyfg, alpha_onlyfg, _, depth_onlyfg = _raw2outputs(total_out, total_zvals, batch_bkg["rays_d"][idx_fg], pts_mask) 
                    rgb_batch[idx_fg] = rgb_onlyfg

                    pts_mask_bkg_onlybg = torch.ones_like(z_vals_bkg_onlybg).cuda()[..., None]
                    rgb_onlybg, alpha_onlybg, _, depth_onlybg = _raw2outputs(bkg_out_onlybg, z_vals_bkg_onlybg, batch_bkg["rays_d"][idx_bg], pts_mask_bkg_onlybg)   
                    rgb_batch[idx_bg] = rgb_onlybg
                    
                    rgb.append(rgb_batch)

                bkg_rgbs = []
                # get the rendered background rays only querying the mipnerf360
                for i in range(0, batch['rays_o_bkg_only'].shape[0], self.cfg.chunk_bkg):
                    batch_bkg_only = {
                        "rays_o": batch['rays_o_bkg_only'][i:i+self.cfg.chunk_bkg],
                        "rays_d": batch['rays_d_bkg_only'][i:i+self.cfg.chunk_bkg],
                        "viewdirs": batch['viewdirs_bkg_only'][i:i+self.cfg.chunk_bkg],
                        "radii": batch['radii_bkg_only'][i:i+self.cfg.chunk_bkg],
                        "times": batch['time']
                    }
                    train_frac = 1.0
                    rendered_results, ray_history = self.model(
                        batch_bkg_only, train_frac, True, True, self.near_bkg, self.far_bkg
                    )        
                    z_vals_bkg = ray_history[-1]['tdist'][..., :-1]
                    bkg_out = torch.cat([ray_history[-1]['rgb'], ray_history[-1]['density'][..., None]], -1)
                    pts_mask_bkg = torch.ones_like(z_vals_bkg).cuda()[..., None]
                    bkg_rgb, bkg_alpha, _, bkg_depth = _raw2outputs(bkg_out, z_vals_bkg, batch_bkg_only["rays_d"], pts_mask_bkg)   
                    bkg_rgbs.append(bkg_rgb) 

            rgb = torch.cat(rgb, 0).data.to("cpu").numpy()
            target_rgbs = batch['target_rgbs'].data.to("cpu").numpy()

            bkg_rgbs = torch.cat(bkg_rgbs, 0).data.to("cpu").numpy()
            target_rgbs_bkg = batch['target_rgbs_bkg'].data.to("cpu").numpy()            

            rendered[ray_mask] = rgb
            truth[ray_mask] = target_rgbs
            rendered[ray_mask_bkg] = bkg_rgbs
            truth[ray_mask_bkg] = target_rgbs_bkg            

            psnr = psnr_metric(rendered, truth)
            ssim = skimage.metrics.structural_similarity(rendered, truth, channel_axis=True)
            predict = torch.from_numpy(self.scale_for_lpips(rendered.reshape(1, height, width, 3).transpose(0, 3, 1, 2))).float().cuda()
            groundtruth = torch.from_numpy(self.scale_for_lpips(truth.reshape(1, height, width, 3).transpose(0, 3, 1, 2))).float().cuda()
            lpips = self.lpips_func(predict, groundtruth).cpu().detach().item()

            psnrs.append(psnr)
            ssims.append(ssim)
            lpipss.append(lpips)     

            truth = to_8b_image(truth.reshape((height, width, -1)))
            rendered = to_8b_image(rendered.reshape((height, width, -1)))
            image_vis = rendered
            if self.trainer.global_step <= 5000 and np.allclose(rendered, np.array(self.cfg.bgcolor), atol=5.):
                is_empty_img = True
                break

            prog_dir = self.logdir + "/progress_vis/" + frame_name
            if not os.path.exists(prog_dir):
                os.makedirs(prog_dir)
            Image.fromarray(image_vis).save(
                os.path.join(prog_dir, "prog_{:06}.jpg".format(self.trainer.global_step)))

            if is_empty_img:
                print("Produce empty images.")
                
        self.progress_end()    

        psnr_final = np.mean(psnrs).item()
        ssim_final = np.mean(ssims).item()
        lpips_final = np.mean(lpipss).item()
        print(f"Progress PSNR is {psnr_final}, SSIM is {ssim_final}, LPIPS is {lpips_final}")

        return is_empty_img

    def test_metrics(self):
        self.test_begin()

        print('Test Test Images ...')

        # images = []
        is_empty_img = False
        psnrs, ssims, lpipss = [], [], []

        for _, batch in enumerate(tqdm(self.test_dataloader)):
            
            # only access the first batch as we process one image one time
            for k, v in batch.items():
                batch[k] = v[0]
            width = batch['img_width']
            height = batch['img_height']
            ray_mask = batch['ray_mask']
            ray_mask_bkg = batch['ray_mask_bkg']
            frame_name = batch['frame_name']

            rendered = np.full(
                        (height * width, 3), np.array(self.cfg.bgcolor)/255., 
                        dtype='float32')
            truth = np.full(
                        (height * width, 3), np.array(self.cfg.bgcolor)/255., 
                        dtype='float32')

            batch['iter_val'] = torch.full((1,), self.trainer.global_step)
            batch = cpu_data_to_gpu(
                    batch, exclude_keys=EXCLUDE_KEYS_TO_GPU)
            with torch.no_grad():

                rgb = []
                for i in range(0, batch['rays_o_bkg'].shape[0], self.cfg.chunk_bkg):
                    batch_bkg = {
                        "rays_o": batch['rays_o_bkg'][i:i+self.cfg.chunk_bkg],
                        "rays_d": batch['rays_d_bkg'][i:i+self.cfg.chunk_bkg],
                        "viewdirs": batch['viewdirs_bkg'][i:i+self.cfg.chunk_bkg],
                        "radii": batch['radii'][i:i+self.cfg.chunk_bkg],
                        "times": batch['time']
                    }
                    train_frac = 1.0
                    rendered_results, ray_history = self.model(
                        batch_bkg, train_frac, False, False, self.near_bkg, self.far_bkg
                    )

                    batch_human = {
                        "rays": batch['rays'][:, i:i+self.cfg.chunk_bkg, :],
                        "near": batch['near'][i:i+self.cfg.chunk_bkg, :],
                        "far": batch['far'][i:i+self.cfg.chunk_bkg, :],
                        "bgcolor": batch['bgcolor'],
                        "target_rgbs": batch['target_rgbs'][i:i+self.cfg.chunk_bkg, :],
                        "dst_Rs": batch['dst_Rs'],
                        "dst_Ts": batch['dst_Ts'],
                        "cnl_gtfms": batch['cnl_gtfms'],
                        "canonical_joints": batch['canonical_joints'],
                        "motion_weights_priors": batch['motion_weights_priors'],
                        "cnl_bbox_min_xyz": batch['cnl_bbox_min_xyz'],
                        "cnl_bbox_max_xyz": batch['cnl_bbox_max_xyz'],
                        "cnl_bbox_scale_xyz": batch['cnl_bbox_scale_xyz'],
                        "dst_posevec": batch['dst_posevec'],
                        "iter_val": batch['iter_val'],
                        "time": batch['time'],
                        "is_train": batch['is_train']
                    }                

                    net_output = self.human(**batch_human)
                    scaleworld_pts = torch.einsum('ji, bni->bnj', batch['newsmpl_to_scale_world'], to_homogeneous(net_output['newsmpl_pts']))[..., :3]

                    if torch.any(torch.abs(batch_bkg["rays_d"][..., None, :]) < 1e-5):
                        idx = torch.abs(batch_bkg["rays_d"][..., None, :]) > 1e-5
                        idx_new = torch.zeros_like(batch_bkg["rays_d"][..., None, :], dtype=torch.bool).cuda()
                        idx_new[..., 0] = idx[..., 0]
                        if torch.sum(idx_new) < idx_new.shape[0]:
                            idx_col = idx_new[..., 0] == False
                            idx_new[..., 1][idx_col] = idx[..., 1][idx_col]
                            if torch.sum(idx_new) < idx_new.shape[0]:
                                idx_col = idx_new[..., 1] == False
                                idx_new[..., 2][idx_col] = idx[..., 2][idx_col]
                        if torch.sum(idx_new) == idx_new.shape[0]:
                            idx_new_pts = idx_new.repeat(1, scaleworld_pts.shape[1], 1)
                            z_vals_human_all = (scaleworld_pts - batch_bkg["rays_o"][..., None, :])[idx_new_pts].reshape(scaleworld_pts.shape[:2]) / (batch_bkg["rays_d"][..., None, :]+1e-10)[idx_new][..., None]
                        else:
                            print("There are points with very small rays_d!")
                            import pdb
                            pdb.set_trace()
                    else:
                        z_vals_human_ = (scaleworld_pts - batch_bkg["rays_o"][..., None, :]) / (batch_bkg["rays_d"][..., None, :]+1e-10)
                        z_vals_human_all = torch.mean(z_vals_human_, dim=-1)

                    thre_fg = 5e-3     # define the threshold for human fg.
                    pts_mask_human_all = net_output['pts_mask']

                    val = torch.sum(pts_mask_human_all, dim=-1)
                    idx_fg = val > thre_fg    # select the fg human, and leave the bg
                    idx_bg = ~idx_fg

                    device = pts_mask_human_all.device
                    rgb_batch = torch.full((pts_mask_human_all.shape[0], 3), 0, dtype=torch.float32, device=device)

                    z_vals_bkg = ray_history[-1]['tdist'][..., :-1][idx_fg]
                    z_vals_bkg_onlybg = ray_history[-1]['tdist'][..., :-1][idx_bg]
                    z_vals_human = z_vals_human_all[idx_fg]
                    human_out = torch.cat([net_output['human_rgb'][idx_fg], net_output['human_density'][..., None][idx_fg]], -1)

                    bkg_out_all = torch.cat([ray_history[-1]['rgb'], ray_history[-1]['density'][..., None]], -1)
                    bkg_out = bkg_out_all[idx_fg]
                    bkg_out_onlybg = bkg_out_all[idx_bg]

                    total_zvals, total_order = torch.sort(torch.cat([z_vals_bkg, z_vals_human], -1), -1)
                    total_out = torch.cat([bkg_out, human_out], 1)
                    _b, _n, _c = total_out.shape
                    total_out = total_out[
                        torch.arange(_b).view(_b, 1, 1).repeat(1, _n, _c),
                        total_order.view(_b, _n, 1).repeat(1, 1, _c),
                        torch.arange(_c).view(1, 1, _c).repeat(_b, _n, 1),
                    ]     

                    pts_mask_human = pts_mask_human_all[idx_fg]

                    pts_mask_bkg = torch.ones_like(z_vals_bkg).cuda()
                    pts_mask = torch.cat([pts_mask_bkg, pts_mask_human], -1)[..., None]
                    _b, _n, _c = pts_mask.shape
                    pts_mask = pts_mask[
                        torch.arange(_b).view(_b, 1, 1).repeat(1, _n, _c),
                        total_order.view(_b, _n, 1).repeat(1, 1, _c),
                        torch.arange(_c).view(1, 1, _c).repeat(_b, _n, 1),
                    ]   

                    rgb_onlyfg, alpha_onlyfg, _, depth_onlyfg = _raw2outputs(total_out, total_zvals, batch_bkg["rays_d"][idx_fg], pts_mask) 
                    rgb_batch[idx_fg] = rgb_onlyfg

                    pts_mask_bkg_onlybg = torch.ones_like(z_vals_bkg_onlybg).cuda()[..., None]
                    rgb_onlybg, alpha_onlybg, _, depth_onlybg = _raw2outputs(bkg_out_onlybg, z_vals_bkg_onlybg, batch_bkg["rays_d"][idx_bg], pts_mask_bkg_onlybg)   
                    rgb_batch[idx_bg] = rgb_onlybg
                    
                    rgb.append(rgb_batch)

                bkg_rgbs = []
                # get the rendered background rays only querying the mipnerf360
                for i in range(0, batch['rays_o_bkg_only'].shape[0], self.cfg.chunk_bkg):
                    batch_bkg_only = {
                        "rays_o": batch['rays_o_bkg_only'][i:i+self.cfg.chunk_bkg],
                        "rays_d": batch['rays_d_bkg_only'][i:i+self.cfg.chunk_bkg],
                        "viewdirs": batch['viewdirs_bkg_only'][i:i+self.cfg.chunk_bkg],
                        "radii": batch['radii_bkg_only'][i:i+self.cfg.chunk_bkg],
                        "times": batch['time']
                    }
                    train_frac = 1.0
                    rendered_results, ray_history = self.model(
                        batch_bkg_only, train_frac, False, False, self.near_bkg, self.far_bkg
                    )        
                    z_vals_bkg = ray_history[-1]['tdist'][..., :-1]
                    bkg_out = torch.cat([ray_history[-1]['rgb'], ray_history[-1]['density'][..., None]], -1)
                    pts_mask_bkg = torch.ones_like(z_vals_bkg).cuda()[..., None]
                    bkg_rgb, bkg_alpha, _, bkg_depth = _raw2outputs(bkg_out, z_vals_bkg, batch_bkg_only["rays_d"], pts_mask_bkg)   
                    bkg_rgbs.append(bkg_rgb) 

            rgb = torch.cat(rgb, 0).data.to("cpu").numpy()
            target_rgbs = batch['target_rgbs'].data.to("cpu").numpy()

            bkg_rgbs = torch.cat(bkg_rgbs, 0).data.to("cpu").numpy()
            target_rgbs_bkg = batch['target_rgbs_bkg'].data.to("cpu").numpy()            

            rendered[ray_mask] = rgb
            truth[ray_mask] = target_rgbs
            rendered[ray_mask_bkg] = bkg_rgbs
            truth[ray_mask_bkg] = target_rgbs_bkg            

            psnr = psnr_metric(rendered, truth)
            ssim = skimage.metrics.structural_similarity(rendered, truth, channel_axis=True)
            predict = torch.from_numpy(self.scale_for_lpips(rendered.reshape(1, height, width, 3).transpose(0, 3, 1, 2))).float().cuda()
            groundtruth = torch.from_numpy(self.scale_for_lpips(truth.reshape(1, height, width, 3).transpose(0, 3, 1, 2))).float().cuda()
            lpips = self.lpips_func(predict, groundtruth).cpu().detach().item()

            psnrs.append(psnr)
            ssims.append(ssim)
            lpipss.append(lpips)     

            truth = to_8b_image(truth.reshape((height, width, -1)))
            rendered = to_8b_image(rendered.reshape((height, width, -1)))
            image_vis = rendered
            if self.trainer.global_step <= 5000 and np.allclose(rendered, np.array(self.cfg.bgcolor), atol=5.):
                is_empty_img = True
                break

            prog_dir = self.logdir + "/test_vis/" + "test_{:06}".format(self.trainer.global_step)
            if not os.path.exists(prog_dir):
                os.makedirs(prog_dir)
            Image.fromarray(image_vis).save(
                os.path.join(prog_dir, frame_name + ".jpg"))

            if is_empty_img:
                print("Produce empty images.")
                
        self.test_end()    

        psnr_final = np.mean(psnrs).item()
        ssim_final = np.mean(ssims).item()
        lpips_final = np.mean(lpipss).item()
        print(f"Test Images PSNR is {psnr_final}, SSIM is {ssim_final}, LPIPS is {lpips_final}")

        return is_empty_img                


    def allimgs_metrics(self):
        self.test_begin()

        print('Test Movement Images ...')

        is_empty_img = False
        psnrs, ssims, lpipss = [], [], []

        for _, batch in enumerate(tqdm(self.movement_dataloader)):
            
            # only access the first batch as we process one image one time
            for k, v in batch.items():
                batch[k] = v[0]
            width = batch['img_width']
            height = batch['img_height']
            ray_mask = batch['ray_mask']
            ray_mask_bkg = batch['ray_mask_bkg']
            frame_name = batch['frame_name']

            rendered = np.full(
                        (height * width, 3), np.array(self.cfg.bgcolor)/255., 
                        dtype='float32')
            truth = np.full(
                        (height * width, 3), np.array(self.cfg.bgcolor)/255., 
                        dtype='float32')

            batch['iter_val'] = torch.full((1,), self.trainer.global_step)
            batch = cpu_data_to_gpu(
                    batch, exclude_keys=EXCLUDE_KEYS_TO_GPU)
            with torch.no_grad():

                rgb = []
                for i in range(0, batch['rays_o_bkg'].shape[0], self.cfg.chunk_bkg):
                    batch_bkg = {
                        "rays_o": batch['rays_o_bkg'][i:i+self.cfg.chunk_bkg],
                        "rays_d": batch['rays_d_bkg'][i:i+self.cfg.chunk_bkg],
                        "viewdirs": batch['viewdirs_bkg'][i:i+self.cfg.chunk_bkg],
                        "radii": batch['radii'][i:i+self.cfg.chunk_bkg],
                        "times": batch['time']
                    }
                    train_frac = 1.0
                    rendered_results, ray_history = self.model(
                        batch_bkg, train_frac, False, False, self.near_bkg, self.far_bkg
                    )

                    batch_human = {
                        "rays": batch['rays'][:, i:i+self.cfg.chunk_bkg, :],
                        "near": batch['near'][i:i+self.cfg.chunk_bkg, :],
                        "far": batch['far'][i:i+self.cfg.chunk_bkg, :],
                        "bgcolor": batch['bgcolor'],
                        "target_rgbs": batch['target_rgbs'][i:i+self.cfg.chunk_bkg, :],
                        "dst_Rs": batch['dst_Rs'],
                        "dst_Ts": batch['dst_Ts'],
                        "cnl_gtfms": batch['cnl_gtfms'],
                        "canonical_joints": batch['canonical_joints'],
                        "motion_weights_priors": batch['motion_weights_priors'],
                        "cnl_bbox_min_xyz": batch['cnl_bbox_min_xyz'],
                        "cnl_bbox_max_xyz": batch['cnl_bbox_max_xyz'],
                        "cnl_bbox_scale_xyz": batch['cnl_bbox_scale_xyz'],
                        "dst_posevec": batch['dst_posevec'],
                        "iter_val": batch['iter_val'],
                        "time": batch['time'],
                        "is_train": batch['is_train']
                    }                

                    net_output = self.human(**batch_human)
                    scaleworld_pts = torch.einsum('ji, bni->bnj', batch['newsmpl_to_scale_world'], to_homogeneous(net_output['newsmpl_pts']))[..., :3]

                    if torch.any(torch.abs(batch_bkg["rays_d"][..., None, :]) < 1e-5):
                        idx = torch.abs(batch_bkg["rays_d"][..., None, :]) > 1e-5
                        idx_new = torch.zeros_like(batch_bkg["rays_d"][..., None, :], dtype=torch.bool).cuda()
                        idx_new[..., 0] = idx[..., 0]
                        if torch.sum(idx_new) < idx_new.shape[0]:
                            idx_col = idx_new[..., 0] == False
                            idx_new[..., 1][idx_col] = idx[..., 1][idx_col]
                            if torch.sum(idx_new) < idx_new.shape[0]:
                                idx_col = idx_new[..., 1] == False
                                idx_new[..., 2][idx_col] = idx[..., 2][idx_col]
                        if torch.sum(idx_new) == idx_new.shape[0]:
                            idx_new_pts = idx_new.repeat(1, scaleworld_pts.shape[1], 1)
                            z_vals_human_all = (scaleworld_pts - batch_bkg["rays_o"][..., None, :])[idx_new_pts].reshape(scaleworld_pts.shape[:2]) / (batch_bkg["rays_d"][..., None, :]+1e-10)[idx_new][..., None]
                        else:
                            print("There are points with very small rays_d!")
                            import pdb
                            pdb.set_trace()
                    else:
                        z_vals_human_ = (scaleworld_pts - batch_bkg["rays_o"][..., None, :]) / (batch_bkg["rays_d"][..., None, :]+1e-10)
                        z_vals_human_all = torch.mean(z_vals_human_, dim=-1)

                    thre_fg = 5e-3    
                    pts_mask_human_all = net_output['pts_mask']

                    val = torch.sum(pts_mask_human_all, dim=-1)
                    idx_fg = val > thre_fg   
                    idx_bg = ~idx_fg

                    device = pts_mask_human_all.device
                    rgb_batch = torch.full((pts_mask_human_all.shape[0], 3), 0, dtype=torch.float32, device=device)

                    z_vals_bkg = ray_history[-1]['tdist'][..., :-1][idx_fg]
                    z_vals_bkg_onlybg = ray_history[-1]['tdist'][..., :-1][idx_bg]
                    z_vals_human = z_vals_human_all[idx_fg]
                    human_out = torch.cat([net_output['human_rgb'][idx_fg], net_output['human_density'][..., None][idx_fg]], -1)

                    bkg_out_all = torch.cat([ray_history[-1]['rgb'], ray_history[-1]['density'][..., None]], -1)
                    bkg_out = bkg_out_all[idx_fg]
                    bkg_out_onlybg = bkg_out_all[idx_bg]

                    total_zvals, total_order = torch.sort(torch.cat([z_vals_bkg, z_vals_human], -1), -1)
                    total_out = torch.cat([bkg_out, human_out], 1)
                    _b, _n, _c = total_out.shape
                    total_out = total_out[
                        torch.arange(_b).view(_b, 1, 1).repeat(1, _n, _c),
                        total_order.view(_b, _n, 1).repeat(1, 1, _c),
                        torch.arange(_c).view(1, 1, _c).repeat(_b, _n, 1),
                    ]     

                    pts_mask_human = pts_mask_human_all[idx_fg]

                    pts_mask_bkg = torch.ones_like(z_vals_bkg).cuda()
                    pts_mask = torch.cat([pts_mask_bkg, pts_mask_human], -1)[..., None]
                    _b, _n, _c = pts_mask.shape
                    pts_mask = pts_mask[
                        torch.arange(_b).view(_b, 1, 1).repeat(1, _n, _c),
                        total_order.view(_b, _n, 1).repeat(1, 1, _c),
                        torch.arange(_c).view(1, 1, _c).repeat(_b, _n, 1),
                    ]   

                    rgb_onlyfg, alpha_onlyfg, _, depth_onlyfg = _raw2outputs(total_out, total_zvals, batch_bkg["rays_d"][idx_fg], pts_mask) 
                    rgb_batch[idx_fg] = rgb_onlyfg

                    pts_mask_bkg_onlybg = torch.ones_like(z_vals_bkg_onlybg).cuda()[..., None]
                    rgb_onlybg, alpha_onlybg, _, depth_onlybg = _raw2outputs(bkg_out_onlybg, z_vals_bkg_onlybg, batch_bkg["rays_d"][idx_bg], pts_mask_bkg_onlybg)   
                    rgb_batch[idx_bg] = rgb_onlybg
                    
                    rgb.append(rgb_batch)

                bkg_rgbs = []
                # get the rendered background rays only querying the mipnerf360
                for i in range(0, batch['rays_o_bkg_only'].shape[0], self.cfg.chunk_bkg):
                    batch_bkg_only = {
                        "rays_o": batch['rays_o_bkg_only'][i:i+self.cfg.chunk_bkg],
                        "rays_d": batch['rays_d_bkg_only'][i:i+self.cfg.chunk_bkg],
                        "viewdirs": batch['viewdirs_bkg_only'][i:i+self.cfg.chunk_bkg],
                        "radii": batch['radii_bkg_only'][i:i+self.cfg.chunk_bkg],
                        "times": batch['time']
                    }
                    train_frac = 1.0
                    rendered_results, ray_history = self.model(
                        batch_bkg_only, train_frac, False, False, self.near_bkg, self.far_bkg
                    )        
                    z_vals_bkg = ray_history[-1]['tdist'][..., :-1]
                    bkg_out = torch.cat([ray_history[-1]['rgb'], ray_history[-1]['density'][..., None]], -1)
                    pts_mask_bkg = torch.ones_like(z_vals_bkg).cuda()[..., None]
                    bkg_rgb, bkg_alpha, _, bkg_depth = _raw2outputs(bkg_out, z_vals_bkg, batch_bkg_only["rays_d"], pts_mask_bkg)   
                    bkg_rgbs.append(bkg_rgb) 

            rgb = torch.cat(rgb, 0).data.to("cpu").numpy()
            target_rgbs = batch['target_rgbs'].data.to("cpu").numpy()

            bkg_rgbs = torch.cat(bkg_rgbs, 0).data.to("cpu").numpy()
            target_rgbs_bkg = batch['target_rgbs_bkg'].data.to("cpu").numpy()            

            rendered[ray_mask] = rgb
            truth[ray_mask] = target_rgbs
            rendered[ray_mask_bkg] = bkg_rgbs
            truth[ray_mask_bkg] = target_rgbs_bkg            

            psnr = psnr_metric(rendered, truth)
            ssim = skimage.metrics.structural_similarity(rendered, truth, channel_axis=True)
            predict = torch.from_numpy(self.scale_for_lpips(rendered.reshape(1, height, width, 3).transpose(0, 3, 1, 2))).float().cuda()
            groundtruth = torch.from_numpy(self.scale_for_lpips(truth.reshape(1, height, width, 3).transpose(0, 3, 1, 2))).float().cuda()
            lpips = self.lpips_func(predict, groundtruth).cpu().detach().item()

            psnrs.append(psnr)
            ssims.append(ssim)
            lpipss.append(lpips)     

            truth = to_8b_image(truth.reshape((height, width, -1)))
            rendered = to_8b_image(rendered.reshape((height, width, -1)))

            image_vis = rendered
            if self.trainer.global_step <= 5000 and np.allclose(rendered, np.array(self.cfg.bgcolor), atol=5.):
                is_empty_img = True
                break

            prog_dir = self.logdir + "/allimgs_vis/" + "test_{:06}".format(self.trainer.global_step)
            if not os.path.exists(prog_dir):
                os.makedirs(prog_dir)
            Image.fromarray(image_vis).save(
                os.path.join(prog_dir, frame_name + ".jpg"))

            if is_empty_img:
                print("Produce empty images.")
                
        self.test_end()    

        psnr_final = np.mean(psnrs).item()
        ssim_final = np.mean(ssims).item()
        lpips_final = np.mean(lpipss).item()
        print(f"All Images PSNR is {psnr_final}, SSIM is {ssim_final}, LPIPS is {lpips_final}")

        return is_empty_img        

    def free_view(self):
        self.test_begin()

        print('Test Freeview Images ...')

        is_empty_img = False
        psnrs, ssims, lpipss = [], [], []

        for idx_images, batch in enumerate(tqdm(self.freeview_dataloader)):
            
            for k, v in batch.items():
                batch[k] = v[0]
            width = batch['img_width']
            height = batch['img_height']
            ray_mask = batch['ray_mask']
            ray_mask_bkg = batch['ray_mask_bkg']
            frame_name = batch['frame_name']

            rendered = np.full(
                        (height * width, 3), np.array(self.cfg.bgcolor)/255., 
                        dtype='float32')
            truth = np.full(
                        (height * width, 3), np.array(self.cfg.bgcolor)/255., 
                        dtype='float32')

            batch['iter_val'] = torch.full((1,), self.trainer.global_step)
            batch = cpu_data_to_gpu(
                    batch, exclude_keys=EXCLUDE_KEYS_TO_GPU)
            with torch.no_grad():

                rgb = []
                for i in range(0, batch['rays_o_bkg'].shape[0], self.cfg.chunk_bkg):
                    batch_bkg = {
                        "rays_o": batch['rays_o_bkg'][i:i+self.cfg.chunk_bkg],
                        "rays_d": batch['rays_d_bkg'][i:i+self.cfg.chunk_bkg],
                        "viewdirs": batch['viewdirs_bkg'][i:i+self.cfg.chunk_bkg],
                        "radii": batch['radii'][i:i+self.cfg.chunk_bkg],
                        "times": batch['time']
                    }
                    train_frac = 1.0
                    rendered_results, ray_history = self.model(
                        batch_bkg, train_frac, False, False, self.near_bkg, self.far_bkg
                    )

                    batch_human = {
                        "rays": batch['rays'][:, i:i+self.cfg.chunk_bkg, :],
                        "near": batch['near'][i:i+self.cfg.chunk_bkg, :],
                        "far": batch['far'][i:i+self.cfg.chunk_bkg, :],
                        "bgcolor": batch['bgcolor'],
                        "target_rgbs": batch['target_rgbs'][i:i+self.cfg.chunk_bkg, :],
                        "dst_Rs": batch['dst_Rs'],
                        "dst_Ts": batch['dst_Ts'],
                        "cnl_gtfms": batch['cnl_gtfms'],
                        "canonical_joints": batch['canonical_joints'],
                        "motion_weights_priors": batch['motion_weights_priors'],
                        "cnl_bbox_min_xyz": batch['cnl_bbox_min_xyz'],
                        "cnl_bbox_max_xyz": batch['cnl_bbox_max_xyz'],
                        "cnl_bbox_scale_xyz": batch['cnl_bbox_scale_xyz'],
                        "dst_posevec": batch['dst_posevec'],
                        "iter_val": batch['iter_val'],
                        "time": batch['time'],
                        "is_train": batch['is_train']
                    }                

                    net_output = self.human(**batch_human)
                    scaleworld_pts = torch.einsum('ji, bni->bnj', batch['newsmpl_to_scale_world'], to_homogeneous(net_output['newsmpl_pts']))[..., :3]

                    if torch.any(torch.abs(batch_bkg["rays_d"][..., None, :]) < 1e-5):
                        idx = torch.abs(batch_bkg["rays_d"][..., None, :]) > 1e-5
                        idx_new = torch.zeros_like(batch_bkg["rays_d"][..., None, :], dtype=torch.bool).cuda()
                        idx_new[..., 0] = idx[..., 0]
                        if torch.sum(idx_new) < idx_new.shape[0]:
                            idx_col = idx_new[..., 0] == False
                            idx_new[..., 1][idx_col] = idx[..., 1][idx_col]
                            if torch.sum(idx_new) < idx_new.shape[0]:
                                idx_col = idx_new[..., 1] == False
                                idx_new[..., 2][idx_col] = idx[..., 2][idx_col]
                        if torch.sum(idx_new) == idx_new.shape[0]:
                            idx_new_pts = idx_new.repeat(1, scaleworld_pts.shape[1], 1)
                            z_vals_human_all = (scaleworld_pts - batch_bkg["rays_o"][..., None, :])[idx_new_pts].reshape(scaleworld_pts.shape[:2]) / (batch_bkg["rays_d"][..., None, :]+1e-10)[idx_new][..., None]
                        else:
                            print("There are points with very small rays_d!")
                            import pdb
                            pdb.set_trace()
                    else:
                        z_vals_human_ = (scaleworld_pts - batch_bkg["rays_o"][..., None, :]) / (batch_bkg["rays_d"][..., None, :]+1e-10)
                        z_vals_human_all = torch.mean(z_vals_human_, dim=-1)

                    thre_fg = 5e-3     
                    pts_mask_human_all = net_output['pts_mask']

                    val = torch.sum(pts_mask_human_all, dim=-1)
                    idx_fg = val > thre_fg    
                    idx_bg = ~idx_fg

                    device = pts_mask_human_all.device
                    rgb_batch = torch.full((pts_mask_human_all.shape[0], 3), 0, dtype=torch.float32, device=device)

                    z_vals_bkg = ray_history[-1]['tdist'][..., :-1][idx_fg]
                    z_vals_bkg_onlybg = ray_history[-1]['tdist'][..., :-1][idx_bg]
                    z_vals_human = z_vals_human_all[idx_fg]
                    human_out = torch.cat([net_output['human_rgb'][idx_fg], net_output['human_density'][..., None][idx_fg]], -1)

                    bkg_out_all = torch.cat([ray_history[-1]['rgb'], ray_history[-1]['density'][..., None]], -1)
                    bkg_out = bkg_out_all[idx_fg]
                    bkg_out_onlybg = bkg_out_all[idx_bg]

                    total_zvals, total_order = torch.sort(torch.cat([z_vals_bkg, z_vals_human], -1), -1)
                    total_out = torch.cat([bkg_out, human_out], 1)
                    _b, _n, _c = total_out.shape
                    total_out = total_out[
                        torch.arange(_b).view(_b, 1, 1).repeat(1, _n, _c),
                        total_order.view(_b, _n, 1).repeat(1, 1, _c),
                        torch.arange(_c).view(1, 1, _c).repeat(_b, _n, 1),
                    ]     

                    pts_mask_human = pts_mask_human_all[idx_fg]

                    pts_mask_bkg = torch.ones_like(z_vals_bkg).cuda()
                    pts_mask = torch.cat([pts_mask_bkg, pts_mask_human], -1)[..., None]
                    _b, _n, _c = pts_mask.shape
                    pts_mask = pts_mask[
                        torch.arange(_b).view(_b, 1, 1).repeat(1, _n, _c),
                        total_order.view(_b, _n, 1).repeat(1, 1, _c),
                        torch.arange(_c).view(1, 1, _c).repeat(_b, _n, 1),
                    ]   

                    rgb_onlyfg, alpha_onlyfg, _, depth_onlyfg = _raw2outputs(total_out, total_zvals, batch_bkg["rays_d"][idx_fg], pts_mask) 
                    rgb_batch[idx_fg] = rgb_onlyfg

                    pts_mask_bkg_onlybg = torch.ones_like(z_vals_bkg_onlybg).cuda()[..., None]
                    rgb_onlybg, alpha_onlybg, _, depth_onlybg = _raw2outputs(bkg_out_onlybg, z_vals_bkg_onlybg, batch_bkg["rays_d"][idx_bg], pts_mask_bkg_onlybg)   
                    rgb_batch[idx_bg] = rgb_onlybg
                    
                    rgb.append(rgb_batch)

                bkg_rgbs = []

                # get the rendered background rays only querying the mipnerf360
                for i in range(0, batch['rays_o_bkg_only'].shape[0], self.cfg.chunk_bkg):
                    batch_bkg_only = {
                        "rays_o": batch['rays_o_bkg_only'][i:i+self.cfg.chunk_bkg],
                        "rays_d": batch['rays_d_bkg_only'][i:i+self.cfg.chunk_bkg],
                        "viewdirs": batch['viewdirs_bkg_only'][i:i+self.cfg.chunk_bkg],
                        "radii": batch['radii_bkg_only'][i:i+self.cfg.chunk_bkg],
                        "times": batch['time']
                    }
                    train_frac = 1.0
                    rendered_results, ray_history = self.model(
                        batch_bkg_only, train_frac, False, False, self.near_bkg, self.far_bkg
                    )        
                    z_vals_bkg = ray_history[-1]['tdist'][..., :-1]
                    bkg_out = torch.cat([ray_history[-1]['rgb'], ray_history[-1]['density'][..., None]], -1)
                    pts_mask_bkg = torch.ones_like(z_vals_bkg).cuda()[..., None]
                    bkg_rgb, bkg_alpha, _, bkg_depth = _raw2outputs(bkg_out, z_vals_bkg, batch_bkg_only["rays_d"], pts_mask_bkg)   
                    bkg_rgbs.append(bkg_rgb) 

            rgb = torch.cat(rgb, 0).data.to("cpu").numpy()
            target_rgbs = batch['target_rgbs'].data.to("cpu").numpy()

            bkg_rgbs = torch.cat(bkg_rgbs, 0).data.to("cpu").numpy()
            target_rgbs_bkg = batch['target_rgbs_bkg'].data.to("cpu").numpy()            

            rendered[ray_mask] = rgb
            truth[ray_mask] = target_rgbs
            rendered[ray_mask_bkg] = bkg_rgbs
            truth[ray_mask_bkg] = target_rgbs_bkg            

            psnr = psnr_metric(rendered, truth)
            ssim = skimage.metrics.structural_similarity(rendered, truth, channel_axis=True)
            predict = torch.from_numpy(self.scale_for_lpips(rendered.reshape(1, height, width, 3).transpose(0, 3, 1, 2))).float().cuda()
            groundtruth = torch.from_numpy(self.scale_for_lpips(truth.reshape(1, height, width, 3).transpose(0, 3, 1, 2))).float().cuda()
            lpips = self.lpips_func(predict, groundtruth).cpu().detach().item()

            psnrs.append(psnr)
            ssims.append(ssim)
            lpipss.append(lpips)     

            truth = to_8b_image(truth.reshape((height, width, -1)))
            rendered = to_8b_image(rendered.reshape((height, width, -1)))
            image_vis = rendered
            if self.trainer.global_step <= 5000 and np.allclose(rendered, np.array(self.cfg.bgcolor), atol=5.):
                is_empty_img = True
                break

            prog_dir = self.logdir + "/freeview_vis_newtrans/" + "view_{:05}".format(self.cfg.freeview.frame_idx)
            if not os.path.exists(prog_dir):
                os.makedirs(prog_dir)
            Image.fromarray(image_vis).save(
                os.path.join(prog_dir, "image-{:05}".format(idx_images) + ".jpg"))

            if is_empty_img:
                print("Produce empty images.")
                
        self.test_end()    

        psnr_final = np.mean(psnrs).item()
        ssim_final = np.mean(ssims).item()
        lpips_final = np.mean(lpipss).item()
        print(f"Freeview PSNR is {psnr_final}, SSIM is {ssim_final}, LPIPS is {lpips_final}")

        return is_empty_img               


    def configure_optimizers(self):

        return create_optimizer(self.cfg, self.human, self.model)
    
    def training_step(self, batch, batch_idx):

        for k, v in batch.items():
            batch[k] = v[0]

        batch['iter_val'] = torch.full((1,), self.trainer.global_step).cuda()  
        batch_bkg = {
            "rays_o": batch['rays_o_bkg'],
            "rays_d": batch['rays_d_bkg'],
            "viewdirs": batch['viewdirs_bkg'],
            "radii": batch['radii'],
            "times": batch['time']
        }
        train_frac = 1.0
        rendered_results, ray_history = self.model(
            batch_bkg, train_frac, True, True, self.near_bkg, self.far_bkg
        )

        net_output = self.human(**batch)
        ray_grid = batch['ray_grid'] if batch['time'] > 0.005 else None
        newsmpl_to_camera_prev = batch['newsmpl_to_camera_prev'] if batch['time'] > 0.005 else None
        intrinsics_prev = batch['intrinsics_prev'] if batch['time'] > 0.005 else None        
        
        scaleworld_pts = torch.einsum('ji, bni->bnj', batch['newsmpl_to_scale_world'], to_homogeneous(net_output['newsmpl_pts']))[..., :3]

        if torch.any(torch.abs(batch_bkg["rays_d"][..., None, :]) < 1e-5):
            idx = torch.abs(batch_bkg["rays_d"][..., None, :]) > 1e-5
            idx_new = torch.zeros_like(batch_bkg["rays_d"][..., None, :], dtype=torch.bool).cuda()
            idx_new[..., 0] = idx[..., 0]
            if torch.sum(idx_new) < idx_new.shape[0]:
                idx_col = idx_new[..., 0] == False
                idx_new[..., 1][idx_col] = idx[..., 1][idx_col]
                if torch.sum(idx_new) < idx_new.shape[0]:
                    idx_col = idx_new[..., 1] == False
                    idx_new[..., 2][idx_col] = idx[..., 2][idx_col]
            if torch.sum(idx_new) == idx_new.shape[0]:
                idx_new_pts = idx_new.repeat(1, scaleworld_pts.shape[1], 1)
                z_vals_human_all = (scaleworld_pts - batch_bkg["rays_o"][..., None, :])[idx_new_pts].reshape(scaleworld_pts.shape[:2]) / (batch_bkg["rays_d"][..., None, :]+1e-10)[idx_new][..., None]
            else:
                print("There are points with very small rays_d!")
                import pdb
                pdb.set_trace()
        else:
            z_vals_human_ = (scaleworld_pts - batch_bkg["rays_o"][..., None, :]) / (batch_bkg["rays_d"][..., None, :]+1e-10)
            z_vals_human_all = torch.mean(z_vals_human_, dim=-1)

        thre_fg = 5e-3     
        pts_mask_human_all = net_output['pts_mask']     
        val = torch.sum(pts_mask_human_all, dim=-1)
        idx_fg = val > thre_fg    
        idx_bg = ~idx_fg

        device = pts_mask_human_all.device
        rgb_batch = torch.full((pts_mask_human_all.shape[0], 3), 0, dtype=torch.float32, device=device)

        z_vals_bkg = ray_history[-1]['tdist'][..., :-1][idx_fg]
        z_vals_bkg_onlybg = ray_history[-1]['tdist'][..., :-1][idx_bg]
        z_vals_human = z_vals_human_all[idx_fg]
        human_out = torch.cat([net_output['human_rgb'][idx_fg], net_output['human_density'][..., None][idx_fg]], -1)

        bkg_out_all = torch.cat([ray_history[-1]['rgb'], ray_history[-1]['density'][..., None]], -1)
        bkg_out = bkg_out_all[idx_fg]
        bkg_out_onlybg = bkg_out_all[idx_bg]        

        total_zvals, total_order = torch.sort(torch.cat([z_vals_bkg, z_vals_human], -1), -1)
        total_out = torch.cat([bkg_out, human_out], 1)
        _b, _n, _c = total_out.shape
        total_out = total_out[
            torch.arange(_b).view(_b, 1, 1).repeat(1, _n, _c),
            total_order.view(_b, _n, 1).repeat(1, 1, _c),
            torch.arange(_c).view(1, 1, _c).repeat(_b, _n, 1),
        ]     

        # record the human pts order to get corresponding updated weights.
        human_pts_idx = total_order >=  z_vals_bkg.shape[1]

        pts_mask_human = pts_mask_human_all[idx_fg]
        pts_mask_bkg = torch.ones_like(z_vals_bkg).cuda()
        pts_mask = torch.cat([pts_mask_bkg, pts_mask_human], -1)[..., None]
        _b, _n, _c = pts_mask.shape
        pts_mask = pts_mask[
            torch.arange(_b).view(_b, 1, 1).repeat(1, _n, _c),
            total_order.view(_b, _n, 1).repeat(1, 1, _c),
            torch.arange(_c).view(1, 1, _c).repeat(_b, _n, 1),
        ]        
        rgb_onlyfg, alpha_onlyfg, weights_onlyfg, depth_onlyfg = _raw2outputs(total_out, total_zvals, batch['rays_d_bkg'][idx_fg], pts_mask) 

        human_weights_onlyfg = weights_onlyfg[human_pts_idx].reshape(z_vals_human.shape)

        rgb_batch[idx_fg] = rgb_onlyfg

        pts_mask_bkg_onlybg = torch.ones_like(z_vals_bkg_onlybg).cuda()[..., None]
        rgb_onlybg, alpha_onlybg, _, depth_onlybg = _raw2outputs(bkg_out_onlybg, z_vals_bkg_onlybg, batch_bkg["rays_d"][idx_bg], pts_mask_bkg_onlybg)   
        rgb_batch[idx_bg] = rgb_onlybg     

        net_output['rgb'] = rgb_batch

        train_loss, loss_dict = self.get_loss(
            net_output=net_output,
            patch_masks=batch['patch_masks'],
            bgcolor=batch['bgcolor'] / 255.,
            targets=batch['target_patches'],
            ray_grid=ray_grid,
            time=batch['time'],
            idx_fg=idx_fg,
            human_weights_onlyfg=human_weights_onlyfg,
            div_indices=batch['patch_div_indices'],
            newsmpl_to_camera_prev=newsmpl_to_camera_prev,
            intrinsics_prev=intrinsics_prev)            

        if torch.isnan(train_loss):
            import pdb
            pdb.set_trace()
            print("train_loss is nan!")
            net_output['rgb'], net_output['alpha'], _, net_output['depth'] = _raw2outputs(total_out, total_zvals, batch_bkg["rays_d"])   
            
        self.log("train/loss", train_loss.item(), on_step=True, prog_bar=True)
        self.log("train/mse", loss_dict['mse'].item(), on_step=True, prog_bar=True)     
        self.log("train/lpips", loss_dict['lpips'].item(), on_step=True, prog_bar=True)   
        self.log("train/cycle", loss_dict['cycle'], on_step=True, prog_bar=True) 
        if batch['time'] > 0.005:
            self.log("train/flow", loss_dict['flow'], on_step=True, prog_bar=True)              

        is_reload_model = False
        if self.trainer.global_step in [100, 300, 1000, 2500] or \
            self.trainer.global_step % self.cfg.progress.dump_interval == 0:
            is_reload_model = self.progress()     

        return train_loss      

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        optimizer_closure,
        on_tpu,
        using_native_amp,
        using_lbfgs,
    ):
        iter_step = self.trainer.global_step
        optimizer.step(closure=optimizer_closure)
        decay_rate = 0.1
        decay_steps = self.cfg.train.lrate_decay * 1000
        decay_value = decay_rate ** (iter_step / decay_steps)
        cus_lr_names = get_customized_lr_names(self.cfg)

        for param_group in optimizer.param_groups:
            is_assigned_lr = False
            for lr_name in cus_lr_names:
                if lr_name in param_group['name']:
                    base_lr = self.cfg.train[f"lr_{lr_name}"]
                    new_lrate = base_lr * decay_value
                    is_assigned_lr = True
            if not is_assigned_lr:
                new_lrate = self.cfg.train.lr_bkgd * decay_value  
            param_group['lr'] = new_lrate        


    def scale_for_lpips(self, image_tensor):
        return image_tensor * 2. - 1.

    def get_img_rebuild_loss(self, loss_names, rgb, target):
        losses = {}

        if "mse" in loss_names:
            losses["mse"] = img2mse(rgb, target)

        if "l1" in loss_names:
            losses["l1"] = img2l1(rgb, target)

        if "lpips" in loss_names:
            lpips_loss = self.lpips_func(self.scale_for_lpips(rgb.permute(0, 3, 1, 2)), 
                                    self.scale_for_lpips(target.permute(0, 3, 1, 2)))
            losses["lpips"] = torch.mean(lpips_loss)

        return losses

    def flow_func(self, ray_grid, newsmpl_to_camera_prev, intrinsics_prev, weights, deform_pts_prev_final):

        pts_prev_cam = torch.einsum('ji, bni->bnj', newsmpl_to_camera_prev, to_homogeneous(deform_pts_prev_final))[..., :3]
        pts_prev_2d_ = torch.einsum('ji, bni->bnj', intrinsics_prev, pts_prev_cam)
        pts_prev_2d = pts_prev_2d_[..., :-1] / pts_prev_2d_[..., -1:]
        ray_grid_pts = ray_grid.unsqueeze(1).repeat(1, pts_prev_2d.shape[1], 1)
        induced_flow = pts_prev_2d - ray_grid_pts[..., :2]
        flow_loss = img2mae(induced_flow, ray_grid_pts[..., 2:4], weights, ray_grid_pts[..., -1].unsqueeze(-1))
        return flow_loss                

    def get_loss(self, net_output, 
                 patch_masks, bgcolor, targets, ray_grid, time, idx_fg, human_weights_onlyfg, div_indices, newsmpl_to_camera_prev, intrinsics_prev):

        lossweights = self.cfg.train.lossweights
        loss_names = list(lossweights.keys())

        rgb = net_output['rgb']
        losses = self.get_img_rebuild_loss(
                        loss_names, 
                        _unpack_imgs(rgb, patch_masks, bgcolor,
                                     targets, div_indices), 
                        targets)

        if "flow" in loss_names and time > 0.005:
            losses["flow"] = self.flow_func(ray_grid[idx_fg], newsmpl_to_camera_prev, intrinsics_prev, human_weights_onlyfg, net_output['deform_pts_prev_final'][idx_fg])
        else:
            losses["flow"] = 0
        if "cycle" in loss_names:
            dis_pts = net_output['observe_pts'] - net_output['deform_pts_final']
            losses["cycle"] = torch.mean(torch.sum(dis_pts**2, 1) / 2.0)                            

        train_losses = [
            weight * losses[k] for k, weight in lossweights.items()
        ]

        return sum(train_losses), \
               {loss_names[i]: train_losses[i] for i in range(len(loss_names))}
