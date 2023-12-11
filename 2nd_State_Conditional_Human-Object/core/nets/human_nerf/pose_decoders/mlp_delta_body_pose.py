# ------------------------------------------------------------------------------------
# HOSNeRF
# Copyright (c) 2023 Show Lab, National University of Singapore. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# Modified from HumanNeRF (https://github.com/chungyiweng/humannerf)
# ------------------------------------------------------------------------------------

import torch.nn as nn

from core.utils.network_util import initseq, RodriguesModule

class BodyPoseRefiner(nn.Module):
    def __init__(self,
                 total_bones=23,
                 embedding_size=69,
                 mlp_width=256,
                 mlp_depth=4,
                 **_):
        super(BodyPoseRefiner, self).__init__()
        
        block_mlps = [nn.Linear(embedding_size, mlp_width), nn.ReLU()]
        
        for _ in range(0, mlp_depth-2):
            block_mlps += [nn.Linear(mlp_width, mlp_width), nn.ReLU()]

        self.total_bones = total_bones - 1

        block_mlps_dstR = [nn.Linear(mlp_width, mlp_width), nn.ReLU()]
        for _ in range(3, mlp_depth-1):
            block_mlps_dstR += [nn.Linear(mlp_width, mlp_width), nn.ReLU()]
        block_mlps_dstR += [nn.Linear(mlp_width, 3 * self.total_bones)]

        block_mlps_dstT = [nn.Linear(mlp_width, mlp_width), nn.ReLU()]
        for _ in range(3, mlp_depth-1):
            block_mlps_dstT += [nn.Linear(mlp_width, mlp_width), nn.ReLU()]
        block_mlps_dstT += [nn.Linear(mlp_width, 3 * self.total_bones)]

        self.block_mlps = nn.Sequential(*block_mlps)
        initseq(self.block_mlps)

        self.block_mlps_dstR = nn.Sequential(*block_mlps_dstR)
        initseq(self.block_mlps_dstR)          

        self.block_mlps_dstT = nn.Sequential(*block_mlps_dstT)
        initseq(self.block_mlps_dstT)         

        # init the weights of the last layer as very small value
        # -- at the beginning, we hope the rotation matrix can be identity 
        init_val = 1e-5
        last_layer_dstR = self.block_mlps_dstR[-1]
        last_layer_dstR.weight.data.uniform_(-init_val, init_val)
        last_layer_dstR.bias.data.zero_()

        init_val = 1e-5
        last_layer_dstT = self.block_mlps_dstT[-1]
        last_layer_dstT.weight.data.uniform_(-init_val, init_val)
        last_layer_dstT.bias.data.zero_()        

        self.rodriguez = RodriguesModule()

    def forward(self, pose_input):

        h = self.block_mlps(pose_input)
        rvec = self.block_mlps_dstR(h).view(-1, 3)
        Rs = self.rodriguez(rvec).view(-1, self.total_bones, 3, 3)
        Ts = self.block_mlps_dstT(h).view(-1, self.total_bones, 3)

        return {
            "Rs": Rs,
            "Ts": Ts,
        }        
