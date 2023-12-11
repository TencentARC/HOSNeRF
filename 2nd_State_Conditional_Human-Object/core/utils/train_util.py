# ------------------------------------------------------------------------------------
# HOSNeRF
# Copyright (c) 2023 Show Lab, National University of Singapore. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# Modified from HumanNeRF (https://github.com/chungyiweng/humannerf)
# ------------------------------------------------------------------------------------

###############################################################################
## Misc Functions
###############################################################################

def cpu_data_to_gpu(cpu_data, exclude_keys=None):
    if exclude_keys is None:
        exclude_keys = []

    gpu_data = {}
    for key, val in cpu_data.items():
        if key in exclude_keys:
            continue

        if isinstance(val, list):
            assert len(val) > 0
            if not isinstance(val[0], str): # ignore string instance
                gpu_data[key] = [x.cuda() for x in val]
        elif isinstance(val, dict):
            gpu_data[key] = {sub_k: sub_val.cuda() for sub_k, sub_val in val.items()}
        else:
            gpu_data[key] = val.cuda()

    return gpu_data
