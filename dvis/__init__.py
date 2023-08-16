# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/MinVIS/blob/main/LICENSE

# Copyright (c) Facebook, Inc. and its affiliates.

# config
from .config import add_minvis_config, add_dvis_config

from .video_mask2former_transformer_decoder import\
    VideoMultiScaleMaskedTransformerDecoder_minvis, VideoMultiScaleMaskedTransformerDecoder_dvis, \
    VideoMultiScaleMaskedTransformerDecoder_minvis_clip, VideoMultiScaleMaskedTransformerDecoder_dvis_clip
from .meta_architecture import MinVIS, DVIS_online, DVIS_offline, DVIS_online_clip

from .video_mask2former_transformer_decoder_ov import VideoMultiScaleMaskedTransformerDecoder_minvis_OV, \
    VideoMultiScaleMaskedTransformerDecoder_dvis_OV

from .meta_architecture_ov import MinVIS_OV, DVIS_online_OV
from .backbones.clip import CLIP

# video
from .data_video import (
    YTVISDatasetMapper,
    CocoClipDatasetMapper,
    YTVISEvaluator,
    PanopticDatasetVideoMapper,
    SemanticDatasetVideoMapper,
    CocoPanoClipDatasetMapper,
    VPSEvaluator,
    VSSEvaluator,
    build_combined_loader,
    build_detection_train_loader,
    build_detection_test_loader,
    get_detection_dataset_dicts,
)
