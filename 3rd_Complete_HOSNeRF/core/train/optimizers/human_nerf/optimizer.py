# ------------------------------------------------------------------------------------
# HOSNeRF
# Copyright (c) 2023 Show Lab, National University of Singapore. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# Modified from HumanNeRF (https://github.com/chungyiweng/humannerf)
# ------------------------------------------------------------------------------------

import torch.optim as optim

_optimizers = {
    'adam': optim.Adam
}

def get_customized_lr_names(cfg):
    return [k[3:] for k in cfg.train.keys() if k.startswith('lr_')]

def get_optimizer(cfg, network, bkgd_model):
    optimizer = _optimizers[cfg.train.optimizer]

    cus_lr_names = get_customized_lr_names(cfg)
    params = []
    print('\n\n********** learnable parameters **********\n')

    for key, value in network.named_parameters():
        if not value.requires_grad:
            continue

        is_assigned_lr = False
        for lr_name in cus_lr_names:
            if lr_name in key:
                params += [{"params": [value], 
                            "lr": cfg.train[f'lr_{lr_name}'],
                            "name": key}]
                print(f"{key}: lr = {cfg.train[f'lr_{lr_name}']}")
                is_assigned_lr = True

        if not is_assigned_lr:
            print("There should not be not assigned_lr!")
            import pdb
            pdb.set_trace()
            params += [{"params": [value], 
                        "name": key}]
            print(f"{key}: lr = {cfg.train.lr}")

    if cfg.train.lr_bkgd > 0:
        for key, value in bkgd_model.named_parameters():
            params += [{"params": [value], 
                        "lr": cfg.train['lr_bkgd'],
                        "name": key}]
            print(f"{key}: lr = {cfg.train.lr_bkgd}")
        print('\n******************************************\n\n')

    if cfg.train.optimizer == 'adam':
        optimizer = optimizer(params, lr=cfg.train.lr_cnl_mlp, betas=(0.9, 0.999))
    else:
        assert False, "Unsupported Optimizer."
        
    return optimizer
