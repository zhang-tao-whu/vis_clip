# -*- coding: utf-8 -*-
# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/MinVIS/blob/main/LICENSE

# Copyright (c) Facebook, Inc. and its affiliates.
from detectron2.config import CfgNode as CN

def add_ctdvis_config(cfg):
    cfg.MODEL.MASK_FORMER.REID_HIDDEN_DIM = 256
    cfg.MODEL.MASK_FORMER.NUM_REID_HEAD_LAYERS = 3



