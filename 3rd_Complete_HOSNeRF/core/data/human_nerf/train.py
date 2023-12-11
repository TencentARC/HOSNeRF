# ------------------------------------------------------------------------------------
# HOSNeRF
# Copyright (c) 2023 Show Lab, National University of Singapore. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# Modified from HumanNeRF (https://github.com/chungyiweng/humannerf)
# ------------------------------------------------------------------------------------

import os
import pickle

import numpy as np
import cv2
import torch
import torch.utils.data

from core.utils.image_util import load_image
from core.data.create_dataset import _get_total_train_imgs
from core.utils.body_util import \
    body_pose_to_body_RTs, \
    get_canonical_global_tfms, \
    approx_gaussian_bone_volumes
from core.utils.file_util import list_files, split_path
from core.utils.camera_util import \
    apply_global_tfm_to_camera, \
    get_rays_from_KRT, \
    get_rays_from_KRT_bkg, \
    rays_intersect_3d_bbox


def resize_flow(flow, H_new, W_new):
    H_old, W_old = flow.shape[0:2]
    flow_resized = cv2.resize(flow, (W_new, H_new), interpolation=cv2.INTER_LINEAR)
    flow_resized[:, :, 0] *= H_new / H_old
    flow_resized[:, :, 1] *= W_new / W_old
    return flow_resized

def get_grid(H, W, flows_b, flow_masks_b):

    # |--------------------|  |--------------------|
    # |       j            |  |       v            |
    # |   i   *            |  |   u   *            |
    # |                    |  |                    |
    # |--------------------|  |--------------------|

    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32), indexing='xy')
    grid = np.stack([i,
                     j,
                     flows_b[:, :, 0],
                     flows_b[:, :, 1],
                     flow_masks_b[:, :]], -1)
    return grid    


class Dataset(torch.utils.data.Dataset):
    def __init__(
            self, 
            cfg, 
            dataset_path,
            keyfilter=None,
            maxframes=-1,
            bgcolor=None,
            ray_shoot_mode='image',
            skip=1,
            **_):

        print('[Dataset Path]', dataset_path) 
        self.cfg = cfg

        self.dataset_path = dataset_path
        self.image_dir = os.path.join(dataset_path, 'images')
        self.flow_dir = os.path.join(dataset_path, 'images_flow')
        total_train_imgs = _get_total_train_imgs(self.dataset_path)
        img_idx = np.arange(total_train_imgs)     
        times_all = np.linspace(0., 1., total_train_imgs).astype(np.float32)       

        self.canonical_joints, self.canonical_bbox = \
            self.load_canonical_joints()
    
        if 'motion_weights_priors' in keyfilter:
            self.motion_weights_priors = \
                approx_gaussian_bone_volumes(
                    self.canonical_joints,   
                    self.canonical_bbox['min_xyz'],
                    self.canonical_bbox['max_xyz'],
                    grid_size=self.cfg.mweight_volume.volume_size).astype('float32')

        grid_size=self.cfg.mweight_volume.volume_size
        min_x, min_y, min_z = self.canonical_bbox['min_xyz']
        max_x, max_y, max_z = self.canonical_bbox['max_xyz']
        zgrid, ygrid, xgrid = np.meshgrid(
            np.linspace(min_z, max_z, grid_size),
            np.linspace(min_y, max_y, grid_size),
            np.linspace(min_x, max_x, grid_size),
            indexing='ij')   
        self.world_pos = np.concatenate((xgrid[..., None], ygrid[..., None], zgrid[..., None]), axis=-1)
        num_voxels = grid_size*grid_size*grid_size
        self.voxel_size = np.power(((self.canonical_bbox['max_xyz'] - self.canonical_bbox['min_xyz']).prod() / num_voxels), 1/3)                       

        self.cameras = self.load_train_cameras()
        self.mesh_infos = self.load_train_mesh_infos()

        self.is_train = False
        framelist = self.load_train_frames()
        self.framelist = framelist[::skip]
        self.img_idx = img_idx[::skip]
        if maxframes > 0:
            self.framelist = self.framelist[:maxframes]
            self.img_idx = self.img_idx[:maxframes]

        if bgcolor == None: # only train bgcolor=None, remove the test images.
            self.is_train = True
            all_idx = np.arange(total_train_imgs)
            skip_test = total_train_imgs // 16
            maxframes_test = 16
            test_idx = all_idx[::skip_test]
            test_idx = test_idx[:maxframes_test]
            self.img_idx = np.array([i for i in np.arange(total_train_imgs) if
                            (i not in test_idx)])    
            self.framelist = [self.framelist[i] for i in self.img_idx]

        self.times = times_all[self.img_idx]
        print(f' -- Total Frames: {self.get_total_frames()}')                 

        self.keyfilter = keyfilter
        self.bgcolor = bgcolor

        self.ray_shoot_mode = ray_shoot_mode

    def load_canonical_joints(self):
        cl_joint_path = os.path.join(self.dataset_path, 'canonical_joints.pkl')
        with open(cl_joint_path, 'rb') as f:
            cl_joint_data = pickle.load(f)
        canonical_joints_body = cl_joint_data['joints'].astype('float32')
        object_joint = canonical_joints_body[23] + (canonical_joints_body[23] - canonical_joints_body[19])
        canonical_joints = np.concatenate((canonical_joints_body, object_joint[None, :]), axis=0)

        object_joint_left = canonical_joints_body[22] + (canonical_joints_body[22] - canonical_joints_body[18])
        canonical_joints = np.concatenate((canonical_joints, object_joint_left[None, :]), axis=0)        

        canonical_bbox = self.skeleton_to_bbox(self.cfg, canonical_joints)

        return canonical_joints, canonical_bbox

    def load_train_cameras(self):
        cameras = None
        with open(os.path.join(self.dataset_path, 'cameras_scaleworld.pkl'), 'rb') as f: 
            cameras = pickle.load(f)
        return cameras

    @staticmethod
    def skeleton_to_bbox(cfg, skeleton):

        min_xyz = np.min(skeleton, axis=0) - cfg.bbox_offset
        max_xyz = np.max(skeleton, axis=0) + cfg.bbox_offset

        return {
            'min_xyz': min_xyz,
            'max_xyz': max_xyz
        }

    def load_train_mesh_infos(self):
        mesh_infos = None
        with open(os.path.join(self.dataset_path, 'mesh_infos.pkl'), 'rb') as f:   
            mesh_infos = pickle.load(f)
        for frame_name in mesh_infos.keys():
            human_tpose_joints = mesh_infos[frame_name]['tpose_joints'].astype('float32')
            init_object_joint = human_tpose_joints[23] + (human_tpose_joints[23] - human_tpose_joints[19])
            mesh_infos[frame_name]['tpose_joints'] = np.concatenate((human_tpose_joints, init_object_joint[None, :]), axis=0)
            mesh_infos[frame_name]['poses'] = np.concatenate((mesh_infos[frame_name]['poses'].astype('float32'), np.zeros(shape=[3], dtype='float32')), axis=0)

            init_object_joint_left = human_tpose_joints[22] + (human_tpose_joints[22] - human_tpose_joints[18])
            mesh_infos[frame_name]['tpose_joints'] = np.concatenate((mesh_infos[frame_name]['tpose_joints'], init_object_joint_left[None, :]), axis=0)
            mesh_infos[frame_name]['poses'] = np.concatenate((mesh_infos[frame_name]['poses'].astype('float32'), np.zeros(shape=[3], dtype='float32')), axis=0)

            bbox = self.skeleton_to_bbox(self.cfg, mesh_infos[frame_name]['joints'])
            mesh_infos[frame_name]['bbox'] = bbox

        return mesh_infos

    def load_train_frames(self):
        img_paths = list_files(os.path.join(self.dataset_path, 'images'),
                               exts=['.png'])
        return [split_path(ipath)[1] for ipath in img_paths]
    
    def query_dst_skeleton(self, frame_name):
        return {
            'poses': self.mesh_infos[frame_name]['poses'].astype('float32'),
            'dst_tpose_joints': \
                self.mesh_infos[frame_name]['tpose_joints'].astype('float32'),
            'bbox': self.mesh_infos[frame_name]['bbox'].copy(),
            'Rh': self.mesh_infos[frame_name]['Rh'].astype('float32'),
            'Th': self.mesh_infos[frame_name]['Th'].astype('float32')
        }

    @staticmethod
    def select_rays(select_inds, rays_o, rays_d, ray_img, rays_o_bkg, rays_d_bkg, viewdirs_bkg, radii, ray_grid, near, far):
        rays_o = rays_o[select_inds]
        rays_d = rays_d[select_inds]
        ray_img = ray_img[select_inds]
        rays_o_bkg = rays_o_bkg[select_inds]
        rays_d_bkg = rays_d_bkg[select_inds]
        viewdirs_bkg = viewdirs_bkg[select_inds]
        radii = radii[select_inds]
        ray_grid = ray_grid[select_inds]
        near = near[select_inds]
        far = far[select_inds]
        return rays_o, rays_d, ray_img, rays_o_bkg, rays_d_bkg, viewdirs_bkg, radii, ray_grid, near, far

    @staticmethod
    def select_rays_original(select_inds, rays_o, rays_d, ray_img, rays_o_bkg, rays_d_bkg, viewdirs_bkg, radii, near, far):
        rays_o = rays_o[select_inds]
        rays_d = rays_d[select_inds]
        ray_img = ray_img[select_inds]
        rays_o_bkg = rays_o_bkg[select_inds]
        rays_d_bkg = rays_d_bkg[select_inds]
        viewdirs_bkg = viewdirs_bkg[select_inds]
        radii = radii[select_inds]
        near = near[select_inds]
        far = far[select_inds]
        return rays_o, rays_d, ray_img, rays_o_bkg, rays_d_bkg, viewdirs_bkg, radii, near, far        
    
    def get_patch_ray_indices(
            self, 
            N_patch, 
            ray_mask, 
            subject_mask, 
            bbox_mask,
            patch_size, 
            H, W):

        assert subject_mask.dtype == np.bool
        assert bbox_mask.dtype == np.bool

        bbox_exclude_subject_mask = np.bitwise_and(
            bbox_mask,
            np.bitwise_not(subject_mask)
        )

        list_ray_indices = []
        list_mask = []
        list_xy_min = []
        list_xy_max = []
        sel_ray_masks = np.zeros_like(ray_mask)

        total_rays = 0
        patch_div_indices = [total_rays]
        for _ in range(N_patch):
            # let p = self.cfg.patch.sample_subject_ratio
            # prob p: we sample on subject area
            # prob (1-p): we sample on non-subject area but still in bbox
            if np.random.rand(1)[0] < self.cfg.patch.sample_subject_ratio:
                candidate_mask = subject_mask
            else:
                candidate_mask = bbox_exclude_subject_mask

            ray_indices, mask, xy_min, xy_max, sel_ray_mask = \
                self._get_patch_ray_indices(ray_mask, candidate_mask, 
                                            patch_size, H, W)

            assert len(ray_indices.shape) == 1
            total_rays += len(ray_indices)

            list_ray_indices.append(ray_indices)
            list_mask.append(mask)
            list_xy_min.append(xy_min)
            list_xy_max.append(xy_max)
            
            patch_div_indices.append(total_rays)

        select_inds = np.concatenate(list_ray_indices, axis=0)
        patch_info = {
            'mask': np.stack(list_mask, axis=0),
            'xy_min': np.stack(list_xy_min, axis=0),
            'xy_max': np.stack(list_xy_max, axis=0)
        }
        patch_div_indices = np.array(patch_div_indices)

        return select_inds, patch_info, patch_div_indices


    def _get_patch_ray_indices(
            self, 
            ray_mask, 
            candidate_mask, 
            patch_size, 
            H, W):

        assert len(ray_mask.shape) == 1
        assert ray_mask.dtype == np.bool
        assert candidate_mask.dtype == np.bool

        valid_ys, valid_xs = np.where(candidate_mask)

        # determine patch center
        select_idx = np.random.choice(valid_ys.shape[0], 
                                      size=[1], replace=False)[0]
        center_x = valid_xs[select_idx]
        center_y = valid_ys[select_idx]

        # determine patch boundary
        half_patch_size = patch_size // 2
        x_min = np.clip(a=center_x-half_patch_size, 
                        a_min=0, 
                        a_max=W-patch_size)
        x_max = x_min + patch_size
        y_min = np.clip(a=center_y-half_patch_size,
                        a_min=0,
                        a_max=H-patch_size)
        y_max = y_min + patch_size

        sel_ray_mask = np.zeros_like(candidate_mask)
        sel_ray_mask[y_min:y_max, x_min:x_max] = True

        #####################################################
        ## Below we determine the selected ray indices
        ## and patch valid mask

        sel_ray_mask = sel_ray_mask.reshape(-1)
        inter_mask = sel_ray_mask       # to keep the patch size to be 32.
        select_masked_inds = np.where(inter_mask)

        masked_indices = np.cumsum(ray_mask) - 1
        select_inds = masked_indices[select_masked_inds]
        
        inter_mask = inter_mask.reshape(H, W)

        return select_inds, \
                inter_mask[y_min:y_max, x_min:x_max], \
                np.array([x_min, y_min]), np.array([x_max, y_max]), sel_ray_mask
    
    def load_image(self, frame_name, bg_color):
        imagepath = os.path.join(self.image_dir, '{}.png'.format(frame_name))
        orig_img = np.array(load_image(imagepath))

        bwd_flow_path = os.path.join(self.flow_dir, '{}_bwd.npz'.format(frame_name))
        bwd_data = np.load(bwd_flow_path)
        bwd_flow, bwd_mask = bwd_data['flow'], bwd_data['mask']
        bwd_mask = np.float32(bwd_mask)            

        maskpath = os.path.join(self.dataset_path, 
                                'masks', 
                                '{}.png'.format(frame_name))
        alpha_mask = np.array(load_image(maskpath))
        
        # undistort image
        if frame_name in self.cameras and 'distortions' in self.cameras[frame_name]:
            K = self.cameras[frame_name]['intrinsics']
            D = self.cameras[frame_name]['distortions']
            orig_img = cv2.undistort(orig_img, K, D)
            alpha_mask = cv2.undistort(alpha_mask, K, D)

        alpha_mask = alpha_mask / 255.
        img = orig_img
        if self.cfg.resize_img_scale != 1.:
            img = cv2.resize(img, None, 
                                fx=self.cfg.resize_img_scale,
                                fy=self.cfg.resize_img_scale,
                                interpolation=cv2.INTER_LANCZOS4)
            alpha_mask = cv2.resize(alpha_mask, None, 
                                    fx=self.cfg.resize_img_scale,
                                    fy=self.cfg.resize_img_scale,
                                    interpolation=cv2.INTER_LINEAR)
            H, W = img.shape[:2]
            bwd_flow = resize_flow(bwd_flow, H, W)
            bwd_mask = cv2.resize(bwd_mask, (W, H),
                                interpolation=cv2.INTER_NEAREST)      

        H, W = img.shape[:2]
        grid = get_grid(int(H), int(W), bwd_flow, bwd_mask) # [H, W, 5]                                                                       
                                
        return img, alpha_mask, grid

    def load_image_original(self, frame_name, bg_color):
        imagepath = os.path.join(self.image_dir, '{}.png'.format(frame_name))
        orig_img = np.array(load_image(imagepath))     

        maskpath = os.path.join(self.dataset_path, 
                                'masks', 
                                '{}.png'.format(frame_name))
        alpha_mask = np.array(load_image(maskpath))
        
        # undistort image
        if frame_name in self.cameras and 'distortions' in self.cameras[frame_name]:
            K = self.cameras[frame_name]['intrinsics']
            D = self.cameras[frame_name]['distortions']
            orig_img = cv2.undistort(orig_img, K, D)
            alpha_mask = cv2.undistort(alpha_mask, K, D)

        alpha_mask = alpha_mask / 255.
        img = orig_img
        if self.cfg.resize_img_scale != 1.:
            img = cv2.resize(img, None, 
                                fx=self.cfg.resize_img_scale,
                                fy=self.cfg.resize_img_scale,
                                interpolation=cv2.INTER_LANCZOS4)
            alpha_mask = cv2.resize(alpha_mask, None, 
                                    fx=self.cfg.resize_img_scale,
                                    fy=self.cfg.resize_img_scale,
                                    interpolation=cv2.INTER_LINEAR)                                                                    
                                
        return img, alpha_mask        


    def get_total_frames(self):
        return len(self.framelist)

    def sample_patch_rays(self, img, H, W,
                          subject_mask, bbox_mask, ray_mask,
                          rays_o, rays_d, ray_img, rays_o_bkg, rays_d_bkg, viewdirs_bkg, radii, ray_grid, near, far):

        select_inds, patch_info, patch_div_indices = \
            self.get_patch_ray_indices(
                N_patch=self.cfg.patch.N_patches, 
                ray_mask=ray_mask, 
                subject_mask=subject_mask, 
                bbox_mask=bbox_mask,
                patch_size=self.cfg.patch.size, 
                H=H, W=W)

        rays_o, rays_d, ray_img, rays_o_bkg, rays_d_bkg, viewdirs_bkg, radii, ray_grid, near, far = self.select_rays(
            select_inds, rays_o, rays_d, ray_img, rays_o_bkg, rays_d_bkg, viewdirs_bkg, radii, ray_grid, near, far)

        targets = []
        for i in range(self.cfg.patch.N_patches):
            x_min, y_min = patch_info['xy_min'][i] 
            x_max, y_max = patch_info['xy_max'][i]
            targets.append(img[y_min:y_max, x_min:x_max])
        target_patches = np.stack(targets, axis=0) # (N_patches, P, P, 3)

        patch_masks = patch_info['mask']  # boolean array (N_patches, P, P)

        return rays_o, rays_d, ray_img, rays_o_bkg, rays_d_bkg, viewdirs_bkg, radii, ray_grid, near, far, \
                target_patches, patch_masks, patch_div_indices

    def sample_patch_rays_original(self, img, H, W,
                          subject_mask, bbox_mask, ray_mask,
                          rays_o, rays_d, ray_img, rays_o_bkg, rays_d_bkg, viewdirs_bkg, radii, near, far):

        select_inds, patch_info, patch_div_indices = \
            self.get_patch_ray_indices(
                N_patch=self.cfg.patch.N_patches, 
                ray_mask=ray_mask, 
                subject_mask=subject_mask, 
                bbox_mask=bbox_mask,
                patch_size=self.cfg.patch.size, 
                H=H, W=W)

        rays_o, rays_d, ray_img, rays_o_bkg, rays_d_bkg, viewdirs_bkg, radii, near, far = self.select_rays_original(
            select_inds, rays_o, rays_d, ray_img, rays_o_bkg, rays_d_bkg, viewdirs_bkg, radii, near, far)

        targets = []
        for i in range(self.cfg.patch.N_patches):
            x_min, y_min = patch_info['xy_min'][i] 
            x_max, y_max = patch_info['xy_max'][i]
            targets.append(img[y_min:y_max, x_min:x_max])
        target_patches = np.stack(targets, axis=0) # (N_patches, P, P, 3)

        patch_masks = patch_info['mask']  # boolean array (N_patches, P, P)

        return rays_o, rays_d, ray_img, rays_o_bkg, rays_d_bkg, viewdirs_bkg, radii, near, far, \
                target_patches, patch_masks, patch_div_indices                

    def __len__(self):
        return self.get_total_frames()

    def __getitem__(self, idx):

        frame_name = self.framelist[idx]
        time = self.times[idx]
        results = {
            'frame_name': frame_name,
            'time': time,
            'is_train': self.is_train,
        }

        if self.bgcolor is None:
            bgcolor = (np.random.rand(3) * 255.).astype('float32')
        else:
            bgcolor = np.array(self.bgcolor, dtype='float32')

        if time > 0.005 and self.is_train:
            img, alpha, grid = self.load_image(frame_name, bgcolor)
        else:
            img, alpha = self.load_image_original(frame_name, bgcolor)
        img = (img / 255.).astype('float32')

        H, W = img.shape[0:2]

        dst_skel_info = self.query_dst_skeleton(frame_name)
        dst_bbox = dst_skel_info['bbox']
        dst_poses = dst_skel_info['poses']
        dst_tpose_joints = dst_skel_info['dst_tpose_joints']

        assert frame_name in self.cameras
        K = self.cameras[frame_name]['intrinsics'][:3, :3].copy()
        K[:2] *= self.cfg.resize_img_scale

        E = self.cameras[frame_name]['smpl_to_camera']
        E, newsmpl_to_smpl = apply_global_tfm_to_camera(
                        E=E, 
                        Rh=dst_skel_info['Rh'],
                        Th=dst_skel_info['Th'])
        R = E[:3, :3]
        T = E[:3, 3]

        smpl_to_scale_world = self.cameras[frame_name]['smpl_to_scale_world']
        scaleworld_to_camera = self.cameras[frame_name]['scaleworld_to_camera']  
        newsmpl_to_scale_world = smpl_to_scale_world @ newsmpl_to_smpl

        rays_o, rays_d = get_rays_from_KRT(H, W, K, R, T)
        ray_img_ori = img.reshape(-1, 3) 
        rays_o_ori = rays_o.reshape(-1, 3) # (H, W, 3) --> (N_rays, 3)
        rays_d_ori = rays_d.reshape(-1, 3)
        if time > 0.005 and self.is_train:
            ray_grid = grid.reshape(-1, 5)                

        # (selected N_samples, ), (selected N_samples, ), (N_samples, )
        near, far, ray_mask = rays_intersect_3d_bbox(dst_bbox, rays_o_ori, rays_d_ori)
        rays_o = rays_o_ori[ray_mask]
        rays_d = rays_d_ori[ray_mask]
        ray_img = ray_img_ori[ray_mask]
        if time > 0.005 and self.is_train:
            ray_grid = ray_grid[ray_mask]           

        R_colmap = scaleworld_to_camera[:3, :3]
        T_colmap = scaleworld_to_camera[:3, 3]
        rays_o_bkg, rays_d_bkg, viewdirs_bkg, radii = get_rays_from_KRT_bkg(H, W, K, R_colmap, T_colmap)
        rays_o_bkg_ori = rays_o_bkg.reshape(-1, 3) # (H, W, 3) --> (N_rays, 3)
        rays_d_bkg_ori = rays_d_bkg.reshape(-1, 3)        
        viewdirs_bkg_ori = viewdirs_bkg.reshape(-1, 3)     
        radii_ori = radii.reshape(-1)[..., None]

        # get the same selected rays using ray_mask     
        rays_o_bkg = rays_o_bkg_ori[ray_mask]
        rays_d_bkg = rays_d_bkg_ori[ray_mask]
        viewdirs_bkg = viewdirs_bkg_ori[ray_mask]
        radii = radii_ori[ray_mask]

        # get the background rays using ray_mask  
        ray_mask_bkg = ~ray_mask
        rays_o_bkg_only = rays_o_bkg_ori[ray_mask_bkg]
        rays_d_bkg_only = rays_d_bkg_ori[ray_mask_bkg]
        viewdirs_bkg_only = viewdirs_bkg_ori[ray_mask_bkg]
        radii_bkg_only = radii_ori[ray_mask_bkg] 
        ray_img_bkg = ray_img_ori[ray_mask_bkg]

        near = near[:, None].astype('float32')
        far = far[:, None].astype('float32')

        if self.ray_shoot_mode == 'image':
            pass
        elif self.ray_shoot_mode == 'patch':
            if time > 0.005 and self.is_train:
                rays_o, rays_d, ray_img, rays_o_bkg, rays_d_bkg, viewdirs_bkg, radii, ray_grid, near, far, \
                target_patches, patch_masks, patch_div_indices = \
                    self.sample_patch_rays(img=img, H=H, W=W,
                                            subject_mask=alpha[:, :, 0] > 0.,
                                            bbox_mask=ray_mask.reshape(H, W),
                                            ray_mask=ray_mask,
                                            rays_o=rays_o, 
                                            rays_d=rays_d, 
                                            ray_img=ray_img, 
                                            rays_o_bkg=rays_o_bkg,
                                            rays_d_bkg=rays_d_bkg,
                                            viewdirs_bkg=viewdirs_bkg,
                                            radii=radii,
                                            ray_grid=ray_grid,
                                            near=near, 
                                            far=far)
            else:
                rays_o, rays_d, ray_img, rays_o_bkg, rays_d_bkg, viewdirs_bkg, radii, near, far, \
                target_patches, patch_masks, patch_div_indices = \
                    self.sample_patch_rays_original(img=img, H=H, W=W,
                                            subject_mask=alpha[:, :, 0] > 0.,
                                            bbox_mask=ray_mask.reshape(H, W),
                                            ray_mask=ray_mask,
                                            rays_o=rays_o, 
                                            rays_d=rays_d, 
                                            ray_img=ray_img, 
                                            rays_o_bkg=rays_o_bkg,
                                            rays_d_bkg=rays_d_bkg,
                                            viewdirs_bkg=viewdirs_bkg,
                                            radii=radii,
                                            near=near, 
                                            far=far)
        else:
            assert False, f"Ivalid Ray Shoot Mode: {self.ray_shoot_mode}"
    
        batch_rays = np.stack([rays_o, rays_d], axis=0) 
        subject_mask = alpha[:, :, 0] > 0.

        if 'rays' in self.keyfilter:
            if time > 0.005 and self.is_train:
                results.update({
                    'img_width': W,
                    'img_height': H,
                    'ray_mask': ray_mask,
                    'ray_mask_bkg': ray_mask_bkg,
                    'rays': batch_rays,
                    'rays_o_bkg': rays_o_bkg,
                    'rays_d_bkg': rays_d_bkg,
                    'viewdirs_bkg': viewdirs_bkg,
                    'radii': radii,
                    'rays_o_bkg_only': rays_o_bkg_only,
                    'rays_d_bkg_only': rays_d_bkg_only,
                    'viewdirs_bkg_only': viewdirs_bkg_only,
                    'radii_bkg_only': radii_bkg_only,
                    'newsmpl_to_scale_world': newsmpl_to_scale_world.astype('float32'),
                    'near': near,
                    'far': far,
                    'bgcolor': bgcolor,
                    'subject_mask': subject_mask,
                    'ray_grid': ray_grid})                
            else:
                results.update({
                    'img_width': W,
                    'img_height': H,
                    'ray_mask': ray_mask,
                    'ray_mask_bkg': ray_mask_bkg,
                    'rays': batch_rays,
                    'rays_o_bkg': rays_o_bkg,
                    'rays_d_bkg': rays_d_bkg,
                    'viewdirs_bkg': viewdirs_bkg,
                    'radii': radii,
                    'rays_o_bkg_only': rays_o_bkg_only,
                    'rays_d_bkg_only': rays_d_bkg_only,
                    'viewdirs_bkg_only': viewdirs_bkg_only,
                    'radii_bkg_only': radii_bkg_only,
                    'newsmpl_to_scale_world': newsmpl_to_scale_world.astype('float32'),
                    'near': near,
                    'far': far,
                    'subject_mask': subject_mask,
                    'bgcolor': bgcolor})

            if self.ray_shoot_mode == 'patch':
                if time > 0.005 and self.is_train:
                    results.update({
                        'patch_div_indices': patch_div_indices,
                        'patch_masks': patch_masks,
                        'target_patches': target_patches,
                        'ray_grid': ray_grid})
                else:
                    results.update({
                        'patch_div_indices': patch_div_indices,
                        'patch_masks': patch_masks,
                        'target_patches': target_patches})   

        if 'target_rgbs' in self.keyfilter:
            results['target_rgbs'] = ray_img
            results['target_rgbs_bkg'] = ray_img_bkg

        if 'motion_bases' in self.keyfilter:
            dst_Rs, dst_Ts = body_pose_to_body_RTs(
                    dst_poses, dst_tpose_joints
                )
            cnl_gtfms = get_canonical_global_tfms(
                            self.canonical_joints)
            results.update({
                'dst_Rs': dst_Rs,
                'dst_Ts': dst_Ts,
                'cnl_gtfms': cnl_gtfms,
                'canonical_joints': self.canonical_joints
            })

            if time > 0.005 and self.is_train:
                frame_name_prev = self.framelist[idx-1]
                time_prev = self.times[idx-1]
                dst_skel_info_prev = self.query_dst_skeleton(frame_name_prev)
                dst_poses_prev = dst_skel_info_prev['poses']
                dst_tpose_joints_prev = dst_skel_info_prev['dst_tpose_joints']
                dst_Rs_prev, dst_Ts_prev = body_pose_to_body_RTs(
                        dst_poses_prev, dst_tpose_joints_prev
                    )

                assert frame_name_prev in self.cameras
                K_prev = self.cameras[frame_name_prev]['intrinsics'][:3, :3].copy()
                K_prev[:2] *= self.cfg.resize_img_scale

                E_prev = self.cameras[frame_name_prev]['smpl_to_camera']
                E_prev, newsmpl_to_smpl_prev = apply_global_tfm_to_camera(
                        E=E_prev, 
                        Rh=dst_skel_info_prev['Rh'],
                        Th=dst_skel_info_prev['Th'])
                R_prev = E_prev[:3, :3]
                T_prev = E_prev[:3, 3]

                results.update({
                    'newsmpl_to_camera_prev': E_prev.astype('float32'),
                    'intrinsics_prev': K_prev.astype('float32'),
                })     

                results.update({
                    'dst_Rs_prev': dst_Rs_prev,
                    'dst_Ts_prev': dst_Ts_prev,
                })            

        if 'motion_weights_priors' in self.keyfilter:
            results['motion_weights_priors'] = self.motion_weights_priors.copy()

        # get the bounding box of canonical volume
        if 'cnl_bbox' in self.keyfilter:
            min_xyz = self.canonical_bbox['min_xyz'].astype('float32')
            max_xyz = self.canonical_bbox['max_xyz'].astype('float32')
            results.update({
                'cnl_bbox_min_xyz': min_xyz,
                'cnl_bbox_max_xyz': max_xyz,
                'cnl_bbox_scale_xyz': 2.0 / (max_xyz - min_xyz)
            })
            assert np.all(results['cnl_bbox_scale_xyz'] >= 0)

        if 'dst_posevec_75' in self.keyfilter:
            # 1. ignore global orientation
            # 2. add a small value to avoid all zeros
            dst_posevec_75 = dst_poses[3:] + 1e-2
            results.update({
                'dst_posevec': dst_posevec_75,
            })
            if time > 0.005 and self.is_train:
                dst_posevec_75_prev = dst_poses_prev[3:] + 1e-2
                results.update({
                    'dst_posevec_prev': dst_posevec_75_prev,
                })            

        return results
