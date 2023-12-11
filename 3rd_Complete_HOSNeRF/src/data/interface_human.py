# ------------------------------------------------------------------------------------
# HOSNeRF
# Copyright (c) 2023 Show Lab, National University of Singapore. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------

from typing import *
import gin
import pytorch_lightning as pl
from core.data import create_dataloader

@gin.configurable()
class LitDataHuman(pl.LightningDataModule):
    def __init__(
        self,
        cfg
    ):

        super(LitDataHuman, self).__init__()
        self.cfg = cfg

    def train_dataloader(self):
        
        return create_dataloader(self.cfg, 'train')

    def val_dataloader(self):

        return create_dataloader(self.cfg, 'progress')

    def test_dataloader(self):

        return create_dataloader(self.cfg, 'movement')

    def predict_dataloader(self):

        return create_dataloader(self.cfg, 'progress')
