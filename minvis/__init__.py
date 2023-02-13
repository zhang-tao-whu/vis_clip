# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/MinVIS/blob/main/LICENSE

# Copyright (c) Facebook, Inc. and its affiliates.

# config
from .config import add_minvis_config

# ori minvis models
from .video_mask2former_transformer_decoder_minvis import VideoMultiScaleMaskedTransformerDecoder_frame
from .video_maskformer_model_minvis import VideoMaskFormer_frame

# models
from .video_maskformer_model import VideoMaskFormer_online
from .video_mask2former_transformer_decoder import VideoMultiScaleMaskedTransformerDecoder_frame_
from .video_maskformer_offline_model import VideoMaskFormer_frame_offline
# video
from .data_video import (
    YTVISDatasetMapper,
    CocoClipDatasetMapper,
    YTVISEvaluator,
    PanopticDatasetVideoMapper,
    VPSEvaluator,
    build_combined_loader,
    build_detection_train_loader,
    build_detection_test_loader,
    get_detection_dataset_dicts,
)
