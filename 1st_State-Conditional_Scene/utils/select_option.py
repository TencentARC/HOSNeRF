# ------------------------------------------------------------------------------------
# HOSNeRF
# Copyright (c) 2023 Show Lab, National University of Singapore. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# Modified from NeRF-Factory (https://github.com/kakaobrain/nerf-factory)
# Copyright (c) 2022 POSTECH, KAIST, Kakao Brain Corp. All Rights Reserved.
# ------------------------------------------------------------------------------------

from typing import *

from src.data.litdata import LitDataNeRF360V2
from src.model.mipnerf360.model import LitMipNeRF360


def select_model(
    model_name: str,
    basedir,
):

    if model_name == "state_mipnerf360":
        return LitMipNeRF360(basedir)
    else:
        raise f"Unknown model named {model_name}"


def select_dataset(
    dataset_name: str,
    datadir: str,
    scene_name: str,
):
    if dataset_name == "nerf_360_v2":
        data_fun = LitDataNeRF360V2

    return data_fun(
        datadir=datadir,
        scene_name=scene_name,
    )
