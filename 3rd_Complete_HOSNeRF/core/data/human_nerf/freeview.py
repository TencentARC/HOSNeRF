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
from core.utils.camera_util import \
    rotate_camera_by_frame_idx, \
    apply_global_tfm_to_camera, \
    get_rays_from_KRT, \
    get_rays_from_KRT_bkg, \
    rays_intersect_3d_bbox
from core.utils.file_util import list_files, split_path


class Dataset(torch.utils.data.Dataset):
    ROT_CAM_PARAMS = {
        'zju_mocap': {'rotate_axis': 'z', 'inv_angle': True},
        'wild': {'rotate_axis': 'y', 'inv_angle': False}
    }

    def __init__(
            self, 
            cfg,
            dataset_path,
            keyfilter=None,
            maxframes=-1,
            skip=1,
            bgcolor=None,
            src_type="zju_mocap",
            **_):

        print('[Dataset Path]', dataset_path) 

        self.dataset_path = dataset_path
        self.image_dir = os.path.join(dataset_path, 'images')
        self.cfg = cfg

        self.canonical_joints, self.canonical_bbox = \
            self.load_canonical_joints()

        if 'motion_weights_priors' in keyfilter:
            self.motion_weights_priors = \
                approx_gaussian_bone_volumes(
                    self.canonical_joints, 
                    self.canonical_bbox['min_xyz'],
                    self.canonical_bbox['max_xyz'],
                    grid_size=self.cfg.mweight_volume.volume_size).astype('float32')

        cameras = self.load_train_cameras()
        mesh_infos = self.load_train_mesh_infos()

        framelist = self.load_train_frames() 
        self.framelist = framelist[::skip]
        if maxframes > 0:
            self.framelist = self.framelist[:maxframes]  

        self.train_frame_idx = self.cfg.freeview.frame_idx
        print(f' -- Frame Idx: {self.train_frame_idx}')


        total_train_imgs = _get_total_train_imgs(self.dataset_path)
        times_all = np.linspace(0., 1., total_train_imgs).astype(np.float32) 
        self.times = times_all[self.train_frame_idx]

        self.total_frames = self.cfg.render_frames
        print(f' -- Total Rendered Frames: {self.total_frames}')

        self.train_frame_name = framelist[self.train_frame_idx]
        self.smpl_to_camera = cameras[framelist[self.train_frame_idx]]['smpl_to_camera']
        self.intrinsics = cameras[framelist[self.train_frame_idx]]['intrinsics'][:3, :3].copy()
        self.smpl_to_scale_world = cameras[framelist[self.train_frame_idx]]['smpl_to_scale_world']
        self.scaleworld_to_camera = cameras[framelist[self.train_frame_idx]]['scaleworld_to_camera']  
        self.train_mesh_info = mesh_infos[framelist[self.train_frame_idx]]

        self.bgcolor = bgcolor if bgcolor is not None else [255., 255., 255.]
        self.keyfilter = keyfilter
        self.src_type = src_type

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
    
    def query_dst_skeleton(self):
        return {
            'poses': self.train_mesh_info['poses'].astype('float32'),
            'dst_tpose_joints': \
                self.train_mesh_info['tpose_joints'].astype('float32'),
            'bbox': self.train_mesh_info['bbox'].copy(),
            'Rh': self.train_mesh_info['Rh'].astype('float32'),
            'Th': self.train_mesh_info['Th'].astype('float32')
        }

    def get_freeview_camera(self, frame_idx, total_frames, trans=None):
        E, T_smpl = rotate_camera_by_frame_idx(
                extrinsics=self.smpl_to_camera, 
                frame_idx=frame_idx,
                period=total_frames,
                trans=trans,
                **self.ROT_CAM_PARAMS[self.src_type])
        K = self.intrinsics
        K[:2] *= self.cfg.resize_img_scale
        return K, E, T_smpl    

    def load_image(self, frame_name, bg_color):
        imagepath = os.path.join(self.image_dir, '{}.png'.format(frame_name))
        orig_img = np.array(load_image(imagepath))

        maskpath = os.path.join(self.dataset_path, 
                                'masks', 
                                '{}.png'.format(frame_name))
        alpha_mask = np.array(load_image(maskpath))

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

    def __len__(self):
        return self.total_frames

    def __getitem__(self, idx):
        frame_name = self.train_frame_name
        time = self.times
        results = {
            'idx_frame': idx,
            'frame_name': frame_name,
            'time': time,
            'is_train': False
        }

        bgcolor = np.array(self.bgcolor, dtype='float32')

        img, _ = self.load_image(frame_name, bgcolor)
        img = img / 255.
        H, W = img.shape[0:2]

        dst_skel_info = self.query_dst_skeleton()
        dst_bbox = dst_skel_info['bbox']
        dst_poses = dst_skel_info['poses']
        dst_tpose_joints = dst_skel_info['dst_tpose_joints']
        dst_Rh = dst_skel_info['Rh']
        dst_Th = dst_skel_info['Th']

        K, E, T_smpl = self.get_freeview_camera(
                        frame_idx=idx,
                        total_frames=self.total_frames,
                        trans=dst_Th)

        T_world = self.smpl_to_scale_world @ T_smpl @ np.linalg.inv(self.smpl_to_scale_world)
        E_colmap = self.scaleworld_to_camera @ T_world

        smpl_to_scale_world = np.linalg.inv(T_world) @ self.smpl_to_scale_world @ T_smpl

        E, newsmpl_to_smpl = apply_global_tfm_to_camera(
                E=E, 
                Rh=dst_Rh,
                Th=dst_Th)
        R = E[:3, :3]
        T = E[:3, 3]

        rays_o, rays_d = get_rays_from_KRT(H, W, K, R, T)
        ray_img_ori = img.reshape(-1, 3) 
        rays_o = rays_o.reshape(-1, 3) # (H, W, 3) --> (N_rays, 3)
        rays_d = rays_d.reshape(-1, 3)

        # (selected N_samples, ), (selected N_samples, ), (N_samples, )
        near, far, ray_mask = rays_intersect_3d_bbox(dst_bbox, rays_o, rays_d)
        rays_o = rays_o[ray_mask]
        rays_d = rays_d[ray_mask]
        ray_img = ray_img_ori[ray_mask]

        newsmpl_to_scale_world = smpl_to_scale_world @ newsmpl_to_smpl

        R_colmap = E_colmap[:3, :3].astype('float32')
        T_colmap = E_colmap[:3, 3].astype('float32')
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
    
        batch_rays = np.stack([rays_o, rays_d], axis=0) 

        if 'rays' in self.keyfilter:
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
                'bgcolor': bgcolor})

        if 'target_rgbs' in self.keyfilter:
            results['target_rgbs'] = ray_img
            results['target_rgbs_bkg'] = ray_img_bkg

        if 'motion_bases' in self.keyfilter:
            dst_Rs, dst_Ts = body_pose_to_body_RTs(
                    dst_poses, dst_tpose_joints)
            cnl_gtfms = get_canonical_global_tfms(self.canonical_joints)
            results.update({
                'dst_Rs': dst_Rs,
                'dst_Ts': dst_Ts,
                'cnl_gtfms': cnl_gtfms, 
                'canonical_joints': self.canonical_joints
            })                                    

        if 'motion_weights_priors' in self.keyfilter:
            results['motion_weights_priors'] = \
                self.motion_weights_priors.copy()

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


        return results
