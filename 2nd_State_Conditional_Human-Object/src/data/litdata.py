# ------------------------------------------------------------------------------------
# HOSNeRF
# Copyright (c) 2023 Show Lab, National University of Singapore. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# Modified from NeRF-Factory (https://github.com/kakaobrain/nerf-factory)
# Copyright (c) 2022 POSTECH, KAIST, Kakao Brain Corp. All Rights Reserved.
# ------------------------------------------------------------------------------------

from typing import Optional

import gin

from src.data.data_util.nerf_360_v2 import load_nerf_360_v2_data
from src.data.interface import LitData
from src.data.interface_human import LitDataHuman


@gin.configurable(denylist=["datadir", "scene_name"])
class LitDataNeRF360V2(LitData):
    def __init__(
        self,
        datadir: str,
        scene_name: str,
        factor: int = 0,
        cam_scale_factor: float = 0.95,
        train_skip: int = 1,
        val_skip: int = 1,
        test_skip: int = 1,
        near: Optional[float] = None,
        far: Optional[float] = None,
        strict_scaling: bool = False,
    ):
        (
            self.images,
            self.masks,
            self.intrinsics,
            self.extrinsics,
            self.image_sizes,
            self.near,
            self.far,
            self.ndc_coeffs,
            (self.i_train, self.i_val, self.i_test, self.i_all),
            self.render_poses,
        ) = load_nerf_360_v2_data(
            datadir=datadir,
            scene_name=scene_name,
            factor=factor,
            cam_scale_factor=cam_scale_factor,
            train_skip=train_skip,
            val_skip=val_skip,
            test_skip=test_skip,
            near=near,
            far=far,
            strict_scaling=strict_scaling,
        )

        super(LitDataNeRF360V2, self).__init__(datadir)

@gin.configurable(denylist=["datadir", "scene_name"])
class LitDataHumanObject(LitDataHuman):
    def __init__(
        self,
        datadir,
        scene_name,        
        cfg
    ):

        super(LitDataHumanObject, self).__init__(cfg)        
