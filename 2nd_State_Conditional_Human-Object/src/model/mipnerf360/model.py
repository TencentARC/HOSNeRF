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
import json
import skimage

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

    patch_imgs = bgcolor.expand(targets.shape).clone() # (N_patch, H, W, 3)
    for i in range(N_patch):
        patch_imgs[i, patch_masks[i]] = rgbs[div_indices[i]:div_indices[i+1]]

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
        with open(os.path.join(basedir, "transitions_times.json"), 'r') as f:
            frame_infos = json.load(f)    
        for frame_base_name in frame_infos:
            time_info = frame_infos[frame_base_name] 
            transitions_times.append(np.array(time_info['time'], dtype=np.float32))
        self.transitions_times = np.stack(transitions_times, axis=0)  

        embedding_size = 64
        pos_size = pos_size + embedding_size   
        self.bkgd_stateembeds = nn.ParameterList([nn.Parameter(torch.randn(embedding_size), requires_grad=True) for i in range(self.transitions_times.shape[0]+1)])

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

    def predict_density(self, means, covs, randomized, is_train):

        means, covs = self.warp_fn(means, covs, is_train)

        lifted_means, lifted_vars = helper.lift_and_diagonalize(
            means, covs, self.pos_basis_t
        )
        x = helper.integrated_pos_enc(
            lifted_means, lifted_vars, self.min_deg_point, self.max_deg_point
        )

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

    def forward(self, gaussians, viewdirs, randomized, is_train):

        means, covs = gaussians

        raw_density, x = self.predict_density(means, covs, randomized, is_train)
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
                gaussians, batch["viewdirs"], randomized, is_train
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

            rendering = helper.volumetric_rendering(
                ray_results["rgb"],
                weights,
                tdist,
                bg_rgbs,
                far,
                False,
            )

            ray_results["sdist"] = sdist
            ray_results["weights"] = weights

            ray_history.append(ray_results)
            renderings.append(rendering)

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
        self.human = create_network(cfg)
        self.cfg = cfg
        self.prog_dataloader = create_dataloader(self.cfg, data_type='progress')
        self.movement_dataloader = create_dataloader(self.cfg, data_type='movement')
        self.test_dataloader = create_dataloader(self.cfg, data_type='test')
        self.tpose_dataloader = create_dataloader(self.cfg, data_type='tpose')
        self.freeview_dataloader = create_dataloader(self.cfg, data_type='freeview')
        if "lpips" in cfg.train.lossweights.keys():
            self.lpips_func = LPIPS(net='vgg')
            set_requires_grad(self.lpips_func, requires_grad=False)

    def progress_begin(self):
        self.human.eval()
        self.cfg.perturb = 0.  

    def progress_end(self):
        self.human.train()
        self.cfg.perturb = self.cfg.train.perturb          

    def progress(self):
        self.progress_begin()

        print('Evaluate Progress Images ...')

        images = []
        psnrs, ssims, lpipss = [], [], []
        is_empty_img = False
        for _, batch in enumerate(tqdm(self.prog_dataloader)):

            for k, v in batch.items():
                batch[k] = v[0]

            width = batch['img_width']
            height = batch['img_height']
            ray_mask = batch['ray_mask']

            rendered = np.full(
                        (height * width, 3), np.array(self.cfg.bgcolor)/255., 
                        dtype='float32')
            truth = np.full(
                        (height * width, 3), np.array(self.cfg.bgcolor)/255., 
                        dtype='float32')

            batch['iter_val'] = torch.full((1,), self.trainer.global_step)
            data = cpu_data_to_gpu(
                    batch, exclude_keys=EXCLUDE_KEYS_TO_GPU + ['target_rgbs'])
            with torch.no_grad():
                net_output = self.human(**data)

            rgb = net_output['rgb'].data.to("cpu").numpy()
            target_rgbs = batch['target_rgbs']

            rendered[ray_mask] = rgb
            truth[ray_mask] = target_rgbs

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
            images.append(np.concatenate([rendered, truth], axis=1))

            if self.trainer.global_step <= 500000 and \
                np.allclose(rendered, np.array(self.cfg.bgcolor), atol=5.):
                is_empty_img = True

        psnr_final = np.mean(psnrs).item()
        ssim_final = np.mean(ssims).item()
        lpips_final = np.mean(lpipss).item()
        print(f"Prog Image PSNR is {psnr_final}, SSIM is {ssim_final}, LPIPS is {lpips_final}")                

        tiled_image = tile_images(images)
        
        Image.fromarray(tiled_image).save(
            os.path.join(self.logdir, "prog_{:06}.jpg".format(self.trainer.global_step)))

        if is_empty_img:
            print("Produce empty images.")
            
        self.progress_end()

        return is_empty_img



    def configure_optimizers(self):
        return create_optimizer(self.cfg, self.human)
    
    def training_step(self, batch, batch_idx):

        for k, v in batch.items():
            batch[k] = v[0]

        batch['iter_val'] = torch.full((1,), self.trainer.global_step).cuda()   # 4 steps should be the same?
        net_output = self.human(**batch)
        ray_grid = batch['ray_grid'] if batch['time'] > 0.005 else None
        newsmpl_to_camera_prev = batch['newsmpl_to_camera_prev'] if batch['time'] > 0.005 else None
        intrinsics_prev = batch['intrinsics_prev'] if batch['time'] > 0.005 else None

        train_loss, loss_dict = self.get_loss(
            net_output=net_output,
            patch_masks=batch['patch_masks'],
            bgcolor=batch['bgcolor'] / 255.,
            targets=batch['target_patches'],
            ray_grid=ray_grid,
            time=batch['time'],            
            div_indices=batch['patch_div_indices'],
            newsmpl_to_camera_prev=newsmpl_to_camera_prev,
            intrinsics_prev=intrinsics_prev)

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
                new_lrate = self.cfg.train.lr * decay_value
            param_group['lr'] = new_lrate       

    def allimgs_metrics(self):
        self.progress_begin()

        print('Evaluate Movement Images ...')

        images = []
        psnrs, ssims, lpipss = [], [], []
        is_empty_img = False
        for _, batch in enumerate(tqdm(self.movement_dataloader)):

            for k, v in batch.items():
                batch[k] = v[0]

            frame_name = batch['frame_name']                
            width = batch['img_width']
            height = batch['img_height']
            ray_mask = batch['ray_mask']

            rendered = np.full(
                        (height * width, 3), np.array(self.cfg.bgcolor)/255., 
                        dtype='float32')
            truth = np.full(
                        (height * width, 3), np.array(self.cfg.bgcolor)/255., 
                        dtype='float32')

            batch['iter_val'] = torch.full((1,), self.trainer.global_step)
            data = cpu_data_to_gpu(
                    batch, exclude_keys=EXCLUDE_KEYS_TO_GPU + ['target_rgbs'])
            with torch.no_grad():
                net_output = self.human(**data)

            rgb = net_output['rgb'].data.to("cpu").numpy()
            target_rgbs = batch['target_rgbs']

            rendered[ray_mask] = rgb
            truth[ray_mask] = target_rgbs

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
            
            prog_dir = self.logdir + "/allimgs_vis/" + "test_{:06}".format(self.trainer.global_step)
            prog_truth_dir = self.logdir + "/allimgs_vis_truth/" + "test_{:06}".format(self.trainer.global_step)
            if not os.path.exists(prog_dir):
                os.makedirs(prog_dir)
            if not os.path.exists(prog_truth_dir):
                os.makedirs(prog_truth_dir)                
            Image.fromarray(image_vis).save(
                os.path.join(prog_dir, frame_name + ".jpg"))                    
            Image.fromarray(truth).save(
                os.path.join(prog_truth_dir, frame_name + ".jpg"))                   

        psnr_final = np.mean(psnrs).item()
        ssim_final = np.mean(ssims).item()
        lpips_final = np.mean(lpipss).item()
        print(f"All Image PSNR is {psnr_final}, SSIM is {ssim_final}, LPIPS is {lpips_final}")
            
        self.progress_end()

        return is_empty_img      


    def test_metrics(self):
        self.progress_begin()

        print('Evaluate Test Images ...')

        images = []
        psnrs, ssims, lpipss = [], [], []
        is_empty_img = False
        for _, batch in enumerate(tqdm(self.test_dataloader)):

            for k, v in batch.items():
                batch[k] = v[0]

            frame_name = batch['frame_name']                
            width = batch['img_width']
            height = batch['img_height']
            ray_mask = batch['ray_mask']

            rendered = np.full(
                        (height * width, 3), np.array(self.cfg.bgcolor)/255., 
                        dtype='float32')
            truth = np.full(
                        (height * width, 3), np.array(self.cfg.bgcolor)/255., 
                        dtype='float32')

            batch['iter_val'] = torch.full((1,), self.trainer.global_step)
            data = cpu_data_to_gpu(
                    batch, exclude_keys=EXCLUDE_KEYS_TO_GPU + ['target_rgbs'])
            with torch.no_grad():
                net_output = self.human(**data)

            rgb = net_output['rgb'].data.to("cpu").numpy()
            target_rgbs = batch['target_rgbs']

            rendered[ray_mask] = rgb
            truth[ray_mask] = target_rgbs

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
            
            prog_dir = self.logdir + "/testimgs_vis/" + "test_{:06}".format(self.trainer.global_step)
            prog_truth_dir = self.logdir + "/testimgs_vis_truth/" + "test_{:06}".format(self.trainer.global_step)
            if not os.path.exists(prog_dir):
                os.makedirs(prog_dir)
            if not os.path.exists(prog_truth_dir):
                os.makedirs(prog_truth_dir)                
            Image.fromarray(image_vis).save(
                os.path.join(prog_dir, frame_name + ".jpg"))        
            Image.fromarray(truth).save(
                os.path.join(prog_truth_dir, frame_name + ".jpg"))                                 

        psnr_final = np.mean(psnrs).item()
        ssim_final = np.mean(ssims).item()
        lpips_final = np.mean(lpipss).item()
        print(f"Test Image PSNR is {psnr_final}, SSIM is {ssim_final}, LPIPS is {lpips_final}")
            
        self.progress_end()

        return is_empty_img                    


    def test_tpose(self, time):
        self.progress_begin()

        print('Evaluate Tpose Images ...')

        images = []
        psnrs, ssims, lpipss = [], [], []
        is_empty_img = False
        idx_tpose = 0
        for _, batch in enumerate(tqdm(self.tpose_dataloader)):

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

            rgb = net_output['rgb'].data.to("cpu").numpy()

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

    def free_view(self):
        self.progress_begin()

        print('Evaluate Freeview Images ...')

        images = []
        psnrs, ssims, lpipss = [], [], []
        is_empty_img = False
        for idx_images, batch in enumerate(tqdm(self.freeview_dataloader)):

            for k, v in batch.items():
                batch[k] = v[0]

            frame_name = batch['frame_name']                
            width = batch['img_width']
            height = batch['img_height']
            ray_mask = batch['ray_mask']

            rendered = np.full(
                        (height * width, 3), np.array(self.cfg.bgcolor)/255., 
                        dtype='float32')

            batch['iter_val'] = torch.full((1,), self.trainer.global_step)
            data = cpu_data_to_gpu(
                    batch, exclude_keys=EXCLUDE_KEYS_TO_GPU + ['target_rgbs'])
            with torch.no_grad():
                net_output = self.human(**data)

            rgb = net_output['rgb'].data.to("cpu").numpy()

            rendered[ray_mask] = rgb

            rendered = to_8b_image(rendered.reshape((height, width, -1)))
            image_vis = rendered
            
            prog_dir = self.logdir + "/freeview_vis_newtrans/" + "view_{:05}".format(self.cfg.freeview.frame_idx)
            if not os.path.exists(prog_dir):
                os.makedirs(prog_dir)
            Image.fromarray(image_vis).save(
                os.path.join(prog_dir, "image-{:05}".format(idx_images) + ".jpg"))                 

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
            self.test_tpose(time=0)            

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
                 patch_masks, bgcolor, targets, ray_grid, time, div_indices, newsmpl_to_camera_prev, intrinsics_prev):

        lossweights = self.cfg.train.lossweights
        loss_names = list(lossweights.keys())

        rgb = net_output['rgb']
        losses = self.get_img_rebuild_loss(
                        loss_names, 
                        _unpack_imgs(rgb, patch_masks, bgcolor,
                                     targets, div_indices), 
                        targets)

        if "flow" in loss_names and time > 0.005:
            losses["flow"] = self.flow_func(ray_grid, newsmpl_to_camera_prev, intrinsics_prev, net_output['weights'], net_output['deform_pts_prev_final'])
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
