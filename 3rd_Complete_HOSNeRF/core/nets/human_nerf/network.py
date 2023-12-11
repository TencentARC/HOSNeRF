# ------------------------------------------------------------------------------------
# HOSNeRF
# Copyright (c) 2023 Show Lab, National University of Singapore. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# Modified from HumanNeRF (https://github.com/chungyiweng/humannerf)
# ------------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
import numpy as np

from core.utils.network_util import MotionBasisComputer
from core.nets.human_nerf.component_factory import \
    load_positional_embedder, \
    load_canonical_mlp, \
    load_mweight_vol_decoder, \
    load_pose_decoder, \
    load_non_rigid_motion_mlp, \
    load_non_rigid_forward_mlp


class Network(nn.Module):
    def __init__(self, cfg):
        super(Network, self).__init__()
        self.cfg = cfg
        self.density_activation = F.relu
        # motion basis computer
        self.motion_basis_computer = MotionBasisComputer(
                                        total_bones=self.cfg.total_bones)

        # motion weight volume
        self.mweight_vol_decoder = load_mweight_vol_decoder(self.cfg.mweight_volume.module)(
            embedding_size=self.cfg.mweight_volume.embedding_size,
            volume_size=self.cfg.mweight_volume.volume_size,
            total_bones=self.cfg.total_bones
        )

        # non-rigid motion st positional encoding
        self.get_non_rigid_embedder = \
            load_positional_embedder(self.cfg.non_rigid_embedder.module)

        # non-rigid motion MLP
        _, non_rigid_pos_embed_size = \
            self.get_non_rigid_embedder(self.cfg.non_rigid_motion_mlp.multires, 
                                        self.cfg.non_rigid_motion_mlp.i_embed, self.cfg)
        self.non_rigid_mlp = \
            load_non_rigid_motion_mlp(self.cfg.non_rigid_motion_mlp.module)(
                pos_embed_size=non_rigid_pos_embed_size,
                condition_code_size=self.cfg.non_rigid_motion_mlp.condition_code_size,
                mlp_width=self.cfg.non_rigid_motion_mlp.mlp_width,
                mlp_depth=self.cfg.non_rigid_motion_mlp.mlp_depth,
                skips=self.cfg.non_rigid_motion_mlp.skips)

        self.non_rigid_forward_mlp = \
            load_non_rigid_forward_mlp(self.cfg.non_rigid_forward_mlp.module)(
                pos_embed_size=non_rigid_pos_embed_size,
                condition_code_size=self.cfg.non_rigid_forward_mlp.condition_code_size,
                mlp_width=self.cfg.non_rigid_forward_mlp.mlp_width,
                mlp_depth=self.cfg.non_rigid_forward_mlp.mlp_depth,
                skips=self.cfg.non_rigid_forward_mlp.skips)              

        # canonical positional encoding
        get_embedder = load_positional_embedder(self.cfg.embedder.module)
        cnl_pos_embed_fn, cnl_pos_embed_size = \
            get_embedder(self.cfg.canonical_mlp.multires, 
                         self.cfg.canonical_mlp.i_embed)
        self.pos_embed_fn = cnl_pos_embed_fn

        # gt transition times
        transitions_times = []
        embedding_size = 64  
        if os.path.exists(os.path.join(cfg.basedir, "transitions_times.json")):
            with open(os.path.join(cfg.basedir, "transitions_times.json"), 'r') as f:
                frame_infos = json.load(f)    
            for frame_base_name in frame_infos:
                time_info = frame_infos[frame_base_name] 
                transitions_times.append(np.array(time_info['time'], dtype=np.float32))
            self.transitions_times = np.stack(transitions_times, axis=0) 
            self.human_stateembeds = nn.ParameterList([nn.Parameter(torch.randn(embedding_size), requires_grad=True) for i in range(self.transitions_times.shape[0]+1)])                 
        else:
            self.human_stateembeds = nn.ParameterList([nn.Parameter(torch.randn(embedding_size), requires_grad=True)])     

        # canonical mlp 
        skips = [4]
        self.cnl_mlp = \
            load_canonical_mlp(self.cfg.canonical_mlp.module)(
                input_ch=cnl_pos_embed_size+embedding_size,
                mlp_depth=self.cfg.canonical_mlp.mlp_depth, 
                mlp_width=self.cfg.canonical_mlp.mlp_width,
                skips=skips)

        # pose decoder MLP
        self.pose_decoder = \
            load_pose_decoder(self.cfg.pose_decoder.module)(
                total_bones=self.cfg.total_bones,
                embedding_size=self.cfg.pose_decoder.embedding_size,
                mlp_width=self.cfg.pose_decoder.mlp_width,
                mlp_depth=self.cfg.pose_decoder.mlp_depth)


    def _query_mlp(
            self,
            pos_xyz,
            pos_embed_fn, 
            non_rigid_pos_embed_fn,
            non_rigid_mlp_input,
            time):

        # (N_rays, N_samples, 3) --> (N_rays x N_samples, 3)
        pos_flat = torch.reshape(pos_xyz, [-1, pos_xyz.shape[-1]])
        chunk = self.cfg.netchunk_per_gpu*4

        result = self._apply_mlp_kernals(
                        pos_flat=pos_flat,
                        pos_embed_fn=pos_embed_fn,
                        non_rigid_mlp_input=non_rigid_mlp_input,
                        non_rigid_pos_embed_fn=non_rigid_pos_embed_fn,
                        chunk=chunk,
                        time=time)

        output = {}

        output['cnl_pts'] = result['cnl_pts']
        raws_flat = result['raws']
        output['raws'] = torch.reshape(
                            raws_flat, 
                            list(pos_xyz.shape[:-1]) + [raws_flat.shape[-1]])

        return output


    @staticmethod
    def _expand_input(input_data, total_elem):
        assert input_data.shape[0] == 1
        input_size = input_data.shape[1]
        return input_data.expand((total_elem, input_size))


    def _apply_mlp_kernals(
            self, 
            pos_flat,
            pos_embed_fn,
            non_rigid_mlp_input,
            non_rigid_pos_embed_fn,
            chunk,
            time):
        raws = []
        cnl_pts = []

        # iterate ray samples by trunks
        for i in range(0, pos_flat.shape[0], chunk):
            start = i
            end = i + chunk
            if end > pos_flat.shape[0]:
                end = pos_flat.shape[0]
            total_elem = end - start

            xyz = pos_flat[start:end]

            if not self.cfg.ignore_non_rigid_motions:
                non_rigid_embed_xyz = non_rigid_pos_embed_fn(xyz)
                result = self.non_rigid_mlp(
                    pos_embed=non_rigid_embed_xyz,
                    pos_xyz=xyz,
                    condition_code=self._expand_input(non_rigid_mlp_input, total_elem)
                )
                xyz = result['xyz']

            cnl_pts += [xyz]
            xyz_embedded = pos_embed_fn(xyz)

            num_pts, _ = xyz_embedded.shape
            eps = 1e-5
            if len(self.human_stateembeds) == 1:
                embed_state_ = self.human_stateembeds[0]         

            if len(self.human_stateembeds) == 2:
                if time < self.transitions_times[0]-eps:
                    embed_state_ = self.human_stateembeds[0]
                else:
                    embed_state_ = self.human_stateembeds[1]                      

            if len(self.human_stateembeds) == 3:
                if time < self.transitions_times[0]-eps:
                    embed_state_ = self.human_stateembeds[0]
                elif time <= self.transitions_times[1]+eps:
                    embed_state_ = self.human_stateembeds[1]
                else:
                    embed_state_ = self.human_stateembeds[2]    

            if len(self.human_stateembeds) == 4:
                if time < self.transitions_times[0]-eps:
                    embed_state_ = self.human_stateembeds[0]
                elif time <= self.transitions_times[1]+eps:
                    embed_state_ = self.human_stateembeds[1]
                elif time <= self.transitions_times[2]+eps:
                    embed_state_ = self.human_stateembeds[2]
                else:
                    embed_state_ = self.human_stateembeds[3]       

            if len(self.human_stateembeds) == 5:
                if time < self.transitions_times[0]-eps:
                    embed_state_ = self.human_stateembeds[0]
                elif time <= self.transitions_times[1]+eps:
                    embed_state_ = self.human_stateembeds[1]
                elif time <= self.transitions_times[2]+eps:
                    embed_state_ = self.human_stateembeds[2]
                elif time <= self.transitions_times[3]+eps:
                    embed_state_ = self.human_stateembeds[3]
                else:
                    embed_state_ = self.human_stateembeds[4]     

            if len(self.human_stateembeds) == 6:
                if time < self.transitions_times[0]-eps:
                    embed_state_ = self.human_stateembeds[0]
                elif time <= self.transitions_times[1]+eps:
                    embed_state_ = self.human_stateembeds[1]
                elif time <= self.transitions_times[2]+eps:
                    embed_state_ = self.human_stateembeds[2]
                elif time <= self.transitions_times[3]+eps:
                    embed_state_ = self.human_stateembeds[3]
                elif time <= self.transitions_times[4]+eps:
                    embed_state_ = self.human_stateembeds[4]                    
                else:
                    embed_state_ = self.human_stateembeds[5] 

            if len(self.human_stateembeds) == 7:
                if time < self.transitions_times[0]-eps:
                    embed_state_ = self.human_stateembeds[0]
                elif time <= self.transitions_times[1]+eps:
                    embed_state_ = self.human_stateembeds[1]
                elif time <= self.transitions_times[2]+eps:
                    embed_state_ = self.human_stateembeds[2]
                elif time <= self.transitions_times[3]+eps:
                    embed_state_ = self.human_stateembeds[3]
                elif time <= self.transitions_times[4]+eps:
                    embed_state_ = self.human_stateembeds[4]         
                elif time <= self.transitions_times[5]+eps:
                    embed_state_ = self.human_stateembeds[5]                                
                else:
                    embed_state_ = self.human_stateembeds[6]

            embed_state = embed_state_.repeat(num_pts, 1)
            xyz_embedded = torch.cat([xyz_embedded, embed_state], dim=-1)      

            raws += [self.cnl_mlp(
                        pos_embed=xyz_embedded)]

        output = {}
        output['raws'] = torch.cat(raws, dim=0)
        output['cnl_pts'] = torch.cat(cnl_pts, dim=0)

        return output


    def _batchify_rays(self, rays_flat, **kwargs):
        all_ret = {}
        for i in range(0, rays_flat.shape[0], self.cfg.chunk):
            ret = self._render_rays(rays_flat[i:i+self.cfg.chunk], **kwargs)
            for k in ret:
                if k not in all_ret:
                    all_ret[k] = []
                all_ret[k].append(ret[k])

        all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
        return all_ret


    @staticmethod
    def _raw2outputs(raw, raw_mask, z_vals, rays_d, bgcolor=None):
        def _raw2alpha(raw, dists, act_fn=F.relu):
            return 1.0 - torch.exp(-act_fn(raw)*dists)

        dists = z_vals[...,1:] - z_vals[...,:-1]

        infinity_dists = torch.Tensor([1e10])
        infinity_dists = infinity_dists.expand(dists[...,:1].shape).to(dists)
        dists = torch.cat([dists, infinity_dists], dim=-1) 
        dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

        rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
        alpha = _raw2alpha(raw[...,3], dists)  # [N_rays, N_samples]
        alpha = alpha * raw_mask[:, :, 0]

        weights = alpha * torch.cumprod(
            torch.cat([torch.ones((alpha.shape[0], 1)).to(alpha), 
                       1.-alpha + 1e-10], dim=-1), dim=-1)[:, :-1]
        rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

        depth_map = torch.sum(weights * z_vals, -1)
        acc_map = torch.sum(weights, -1)

        rgb_map = rgb_map + (1.-acc_map[...,None]) * bgcolor[None, :]/255.

        return rgb_map, acc_map, weights, depth_map


    @staticmethod
    def _sample_motion_fields(
            pts,
            motion_scale_Rs, 
            motion_Ts, 
            motion_weights_vol,
            cnl_bbox_min_xyz, cnl_bbox_scale_xyz,
            output_list):
        orig_shape = list(pts.shape)
        pts = pts.reshape(-1, 3) # [N_rays x N_samples, 3]

        # remove BG channel
        motion_weights = motion_weights_vol[:-1] 

        weights_list = []
        for i in range(motion_weights.size(0)):
            pos = torch.matmul(motion_scale_Rs[i, :, :], pts.T).T + motion_Ts[i, :]
            pos = (pos - cnl_bbox_min_xyz[None, :]) \
                            * cnl_bbox_scale_xyz[None, :] - 1.0 
            weights = F.grid_sample(input=motion_weights[None, i:i+1, :, :, :], 
                                    grid=pos[None, None, None, :, :],           
                                    padding_mode='zeros', align_corners=True)
            weights = weights[0, 0, 0, 0, :, None] 
            weights_list.append(weights) 
        backwarp_motion_weights = torch.cat(weights_list, dim=-1)
        total_bases = backwarp_motion_weights.shape[-1]

        backwarp_motion_weights_sum = torch.sum(backwarp_motion_weights, 
                                                dim=-1, keepdim=True)
        weighted_motion_fields = []
        for i in range(total_bases):
            pos = torch.matmul(motion_scale_Rs[i, :, :], pts.T).T + motion_Ts[i, :]
            weighted_pos = backwarp_motion_weights[:, i:i+1] * pos
            weighted_motion_fields.append(weighted_pos)
        x_skel = torch.sum(
                        torch.stack(weighted_motion_fields, dim=0), dim=0
                        ) / backwarp_motion_weights_sum.clamp(min=0.0001)
        fg_likelihood_mask = backwarp_motion_weights_sum

        x_skel = x_skel.reshape(orig_shape[:2]+[3])
        backwarp_motion_weights = \
            backwarp_motion_weights.reshape(orig_shape[:2]+[total_bases])
        fg_likelihood_mask = fg_likelihood_mask.reshape(orig_shape[:2]+[1])

        results = {}
        
        if 'x_skel' in output_list: # [N_rays x N_samples, 3]
            results['x_skel'] = x_skel
        if 'fg_likelihood_mask' in output_list: # [N_rays x N_samples, 1]
            results['fg_likelihood_mask'] = fg_likelihood_mask
        
        return results

    @staticmethod
    def _sample_motion_fields_forward(
            cnl_pts,
            motion_scale_Rs_forward, 
            motion_Ts_forward, 
            motion_weights_vol,
            cnl_bbox_min_xyz, cnl_bbox_scale_xyz,
            output_list):
        
        # remove BG channel
        motion_weights = motion_weights_vol[:-1] 

        weights_list = []

        pos = (cnl_pts - cnl_bbox_min_xyz[None, :]) \
                        * cnl_bbox_scale_xyz[None, :] - 1.0 
        weights = F.grid_sample(input=motion_weights[None, :, :, :, :], 
                                grid=pos[None, None, None, :, :],           
                                padding_mode='zeros', align_corners=True)

        forward_motion_weights = weights[0, :, 0, 0, :].permute(1, 0)
        total_bases = forward_motion_weights.shape[-1]

        forward_motion_weights_sum = torch.sum(forward_motion_weights, 
                                                dim=-1, keepdim=True)
        weighted_motion_fields = []
        for i in range(total_bases):
            pos = torch.matmul(motion_scale_Rs_forward[i, :, :], cnl_pts.T).T + motion_Ts_forward[i, :]
            weighted_pos = forward_motion_weights[:, i:i+1] * pos
            weighted_motion_fields.append(weighted_pos)
        x_deform = torch.sum(
                        torch.stack(weighted_motion_fields, dim=0), dim=0
                        ) / forward_motion_weights_sum.clamp(min=0.0001)
        fg_likelihood_mask_forward = forward_motion_weights_sum

        results = {}
        
        if 'x_deform' in output_list: # [N_rays x N_samples, 3]
            results['x_deform'] = x_deform
        if 'fg_likelihood_mask_forward' in output_list: # [N_rays x N_samples, 1]
            results['fg_likelihood_mask_forward'] = fg_likelihood_mask_forward
        
        return results       

    @staticmethod
    def _unpack_ray_batch(ray_batch):
        rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] 
        bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2]) 
        near, far = bounds[...,0], bounds[...,1] 
        return rays_o, rays_d, near, far


    @staticmethod
    def _get_samples_along_ray(N_rays, near, far, N_samples):
        t_vals = torch.linspace(0., 1., steps=N_samples).to(near)
        z_vals = near * (1.-t_vals) + far * (t_vals)
        return z_vals.expand([N_rays, N_samples]) 


    @staticmethod
    def _stratified_sampling(z_vals):
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        
        t_rand = torch.rand(z_vals.shape).to(z_vals)
        z_vals = lower + (upper - lower) * t_rand

        return z_vals


    def _render_rays(
            self, 
            ray_batch, 
            N_samples,
            motion_scale_Rs,
            motion_Ts,
            motion_weights_vol,
            cnl_bbox_min_xyz,
            cnl_bbox_scale_xyz,
            pos_embed_fn,
            non_rigid_pos_embed_fn,
            non_rigid_mlp_input=None,
            bgcolor=None,
            time=None,
            **kwargs):
            # **_):
        
        N_rays = ray_batch.shape[0]
        rays_o, rays_d, near, far = self._unpack_ray_batch(ray_batch)

        z_vals = self._get_samples_along_ray(N_rays, near, far, N_samples)
        if self.cfg.perturb > 0.:
            z_vals = self._stratified_sampling(z_vals)

        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None]
        
        mv_output = self._sample_motion_fields(
                            pts=pts,
                            motion_scale_Rs=motion_scale_Rs[0], 
                            motion_Ts=motion_Ts[0], 
                            motion_weights_vol=motion_weights_vol,
                            cnl_bbox_min_xyz=cnl_bbox_min_xyz, 
                            cnl_bbox_scale_xyz=cnl_bbox_scale_xyz,
                            output_list=['x_skel', 'fg_likelihood_mask'])
        pts_mask = mv_output['fg_likelihood_mask']
        cnl_pts = mv_output['x_skel']

        query_result = self._query_mlp(
                                pos_xyz=cnl_pts,
                                non_rigid_mlp_input=non_rigid_mlp_input,
                                pos_embed_fn=pos_embed_fn,
                                non_rigid_pos_embed_fn=non_rigid_pos_embed_fn,
                                time=time)
        raw = query_result['raws']
        cnl_pts_final = query_result['cnl_pts']

        # flow (t-1)
        if time > 0.005 and kwargs['is_train']:   
            cnl_pts_select = cnl_pts_final.reshape(-1, 3)                  
            mv_output_forward_prev = self._sample_motion_fields_forward(
                                cnl_pts=cnl_pts_select,
                                motion_scale_Rs_forward=kwargs['motion_scale_Rs_prev_forward'][0], 
                                motion_Ts_forward=kwargs['motion_Ts_prev_forward'][0], 
                                motion_weights_vol=motion_weights_vol,
                                cnl_bbox_min_xyz=cnl_bbox_min_xyz, 
                                cnl_bbox_scale_xyz=cnl_bbox_scale_xyz,
                                output_list=['x_deform'])  
            deform_pts_prev = mv_output_forward_prev['x_deform']

            # forward non-rigid
            if not self.cfg.ignore_non_rigid_motions:
                elem_num = deform_pts_prev.shape[0]
                non_rigid_embed_pts_prev = non_rigid_pos_embed_fn(deform_pts_prev)
                result_prev = self.non_rigid_forward_mlp(
                    pos_embed=non_rigid_embed_pts_prev,
                    pos_xyz=deform_pts_prev,
                    condition_code=self._expand_input(kwargs['non_rigid_mlp_input_forward_prev'], elem_num)
                )
                deform_pts_prev_flat = result_prev['xyz']
            else:
                deform_pts_prev_flat = deform_pts_prev
            deform_pts_prev_final = torch.reshape(
                    deform_pts_prev_flat, 
                    list(pts.shape[:-1]) + [deform_pts_prev_flat.shape[-1]])
        else:
            deform_pts_prev_final = None

        # cycle consistency
        idx_forward = pts_mask > 0.005
        if torch.sum(idx_forward) > 0:    
            idx_forward = idx_forward.repeat(1, 1, 3)
            observe_pts = pts[idx_forward].reshape(-1, 3)
            idx_forward = idx_forward.reshape(-1, 3)
            cnl_pts_select = cnl_pts_final[idx_forward].reshape(-1, 3)                      
            mv_output_forward = self._sample_motion_fields_forward(
                                cnl_pts=cnl_pts_select,
                                motion_scale_Rs_forward=kwargs['motion_scale_Rs_forward'][0], 
                                motion_Ts_forward=kwargs['motion_Ts_forward'][0], 
                                motion_weights_vol=motion_weights_vol,
                                cnl_bbox_min_xyz=cnl_bbox_min_xyz, 
                                cnl_bbox_scale_xyz=cnl_bbox_scale_xyz,
                                output_list=['x_deform'])  
            deform_pts = mv_output_forward['x_deform']

            # forward non-rigid
            if not self.cfg.ignore_non_rigid_motions:
                elem_num = deform_pts.shape[0]
                non_rigid_embed_pts = non_rigid_pos_embed_fn(deform_pts)
                result_forward = self.non_rigid_forward_mlp(
                    pos_embed=non_rigid_embed_pts,
                    pos_xyz=deform_pts,
                    condition_code=self._expand_input(non_rigid_mlp_input, elem_num)
                )
                deform_pts_final = result_forward['xyz']
            else:
                deform_pts_final = deform_pts
            
        else:
            deform_pts_final = pts[0, 0, :][None, :]
            observe_pts = pts[0, 0, :][None, :]

        if time > 0.005 and kwargs['is_train']:
            return {'human_rgb': torch.sigmoid(raw[...,:3]),
                    'human_density': self.density_activation(raw[...,3]),
                    'newsmpl_pts': pts,
                    'pts_mask': pts_mask[..., 0],
                    'deform_pts_prev_final': deform_pts_prev_final,
                    'deform_pts_final': deform_pts_final,
                    'observe_pts': observe_pts
                    }            
        else:

            return {'human_rgb': torch.sigmoid(raw[...,:3]),
                    'human_density': self.density_activation(raw[...,3]),
                    'newsmpl_pts': pts,
                    'pts_mask': pts_mask[..., 0],
                    'z_vals': z_vals,
                    'rays_d': rays_d,
                    'deform_pts_final': deform_pts_final,
                    'observe_pts': observe_pts
                    }        
  

    def _get_motion_base(self, dst_Rs, dst_Ts, cnl_gtfms):
        motion_scale_Rs, motion_Ts, motion_scale_Rs_forward, motion_Ts_forward = self.motion_basis_computer(
                                        dst_Rs, dst_Ts, cnl_gtfms)

        return motion_scale_Rs, motion_Ts, motion_scale_Rs_forward, motion_Ts_forward


    @staticmethod
    def _multiply_corrected_Rs(cfg, Rs, correct_Rs):
        total_bones = cfg.total_bones - 1
        return torch.matmul(Rs.reshape(-1, 3, 3),
                            correct_Rs.reshape(-1, 3, 3)).reshape(-1, total_bones, 3, 3)

    
    def forward(self,
                rays, 
                dst_Rs, dst_Ts, cnl_gtfms,
                motion_weights_priors,
                dst_posevec=None,
                near=None, far=None,
                iter_val=1e7,
                **kwargs):

        dst_Rs=dst_Rs[None, ...]
        dst_Ts=dst_Ts[None, ...]
        dst_posevec=dst_posevec[None, ...]
        cnl_gtfms=cnl_gtfms[None, ...]
        motion_weights_priors=motion_weights_priors[None, ...]

        # correct body pose
        if iter_val >= self.cfg.pose_decoder.get('kick_in_iter', 0):
            pose_out = self.pose_decoder(dst_posevec)
            refined_Rs = pose_out['Rs']
            refined_Ts = pose_out['Ts']
            
            dst_Rs_no_root = dst_Rs[:, 1:, ...]
            dst_Rs_no_root = self._multiply_corrected_Rs(
                                        self.cfg, dst_Rs_no_root, 
                                        refined_Rs)
            dst_Rs = torch.cat(
                [dst_Rs[:, 0:1, ...], dst_Rs_no_root], dim=1)

            dst_Ts_no_root = dst_Ts[:, 1:, ...]
            dst_Ts_no_root = dst_Ts_no_root + refined_Ts
            dst_Ts = torch.cat(
                [dst_Ts[:, 0:1, ...], dst_Ts_no_root], dim=1)

        # correct body prev pose
        if kwargs['time'] > 0.005 and kwargs['is_train']:
            dst_Rs_prev=kwargs['dst_Rs_prev'][None, ...]
            dst_Ts_prev=kwargs['dst_Ts_prev'][None, ...]
            dst_posevec_prev=kwargs['dst_posevec_prev'][None, ...]

            if iter_val >= self.cfg.pose_decoder.get('kick_in_iter', 0):
                pose_out_prev = self.pose_decoder(dst_posevec_prev)
                refined_Rs_prev = pose_out_prev['Rs']
                refined_Ts_prev = pose_out_prev['Ts']
                
                dst_Rs_no_root_prev = dst_Rs_prev[:, 1:, ...]
                dst_Rs_no_root_prev = self._multiply_corrected_Rs(
                                            self.cfg, dst_Rs_no_root_prev, 
                                            refined_Rs_prev)
                dst_Rs_prev = torch.cat(
                    [dst_Rs_prev[:, 0:1, ...], dst_Rs_no_root_prev], dim=1)

                dst_Ts_no_root_prev = dst_Ts_prev[:, 1:, ...]
                dst_Ts_no_root_prev = dst_Ts_no_root_prev + refined_Ts_prev
                dst_Ts_prev = torch.cat(
                    [dst_Ts_prev[:, 0:1, ...], dst_Ts_no_root_prev], dim=1)

            motion_scale_Rs_prev, motion_Ts_prev, motion_scale_Rs_prev_forward, motion_Ts_prev_forward = self._get_motion_base(
                                                        dst_Rs=dst_Rs_prev, 
                                                        dst_Ts=dst_Ts_prev, 
                                                        cnl_gtfms=cnl_gtfms)   

            if iter_val < self.cfg.non_rigid_motion_mlp.kick_in_iter:
                # mask-out non_rigid_mlp_input 
                non_rigid_mlp_input_forward_prev = torch.zeros_like(dst_posevec_prev) * dst_posevec_prev
            else:
                non_rigid_mlp_input_forward_prev = dst_posevec_prev

            kwargs.update({
                'motion_scale_Rs_prev_forward': motion_scale_Rs_prev_forward,
                'motion_Ts_prev_forward': motion_Ts_prev_forward,
                'non_rigid_mlp_input_forward_prev': non_rigid_mlp_input_forward_prev,
            })                    

        non_rigid_pos_embed_fn, _ = \
            self.get_non_rigid_embedder(
                multires=self.cfg.non_rigid_motion_mlp.multires,                         
                is_identity=self.cfg.non_rigid_motion_mlp.i_embed,
                cfg=self.cfg, iter_val=iter_val)

        if iter_val < self.cfg.non_rigid_motion_mlp.kick_in_iter:
            # mask-out non_rigid_mlp_input 
            non_rigid_mlp_input = torch.zeros_like(dst_posevec) * dst_posevec
        else:
            non_rigid_mlp_input = dst_posevec

        kwargs.update({
            "pos_embed_fn": self.pos_embed_fn,
            "non_rigid_pos_embed_fn": non_rigid_pos_embed_fn,
            "non_rigid_mlp_input": non_rigid_mlp_input,
            "N_samples": self.cfg.N_samples            
        })

        motion_scale_Rs, motion_Ts, motion_scale_Rs_forward, motion_Ts_forward = self._get_motion_base(
                                                                                            dst_Rs=dst_Rs, 
                                                                                            dst_Ts=dst_Ts, 
                                                                                            cnl_gtfms=cnl_gtfms)
        motion_weights_vol = self.mweight_vol_decoder(
            motion_weights_priors=motion_weights_priors)    
        motion_weights_vol=motion_weights_vol[0] # remove batch dimension

        kwargs.update({
            'motion_scale_Rs': motion_scale_Rs,
            'motion_Ts': motion_Ts,
            'motion_scale_Rs_forward': motion_scale_Rs_forward,
            'motion_Ts_forward': motion_Ts_forward,
            'motion_weights_vol': motion_weights_vol
        })

        rays_o, rays_d = rays
        rays_shape = rays_d.shape 

        rays_o = torch.reshape(rays_o, [-1,3]).float()
        rays_d = torch.reshape(rays_d, [-1,3]).float()
        packed_ray_infos = torch.cat([rays_o, rays_d, near, far], -1)

        all_ret = self._batchify_rays(packed_ray_infos, **kwargs)

        for k in all_ret:
            if k!='deform_pts_prev_final' and k!= 'deform_pts_final' and k != 'observe_pts':
                k_shape = list(rays_shape[:-1]) + list(all_ret[k].shape[1:])
                all_ret[k] = torch.reshape(all_ret[k], k_shape)

        all_ret['bgcolor'] = kwargs['bgcolor']

        return all_ret
