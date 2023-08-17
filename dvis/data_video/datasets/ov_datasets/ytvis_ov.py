# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/MinVIS/blob/main/LICENSE

# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/sukjunhwang/IFC

import contextlib
import io
import json
import copy
import logging
import numpy as np
import os
import pycocotools.mask as mask_util
from fvcore.common.file_io import PathManager
from fvcore.common.timer import Timer

from detectron2.structures import Boxes, BoxMode, PolygonMasks
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.builtin_meta import _get_builtin_metadata
from detectron2.data.datasets.coco import register_coco_instances

"""
This file contains functions to parse YTVIS dataset of
COCO-format annotations into dicts in "Detectron2 format".
"""
import logging
logger = logging.getLogger(__name__)

COCO_CATEGORIES = [
    {"color": [220, 20, 60], "isthing": 1, "id": 1, "name": "person"},
    {"color": [119, 11, 32], "isthing": 1, "id": 2, "name": "bicycle"},
    {"color": [0, 0, 142], "isthing": 1, "id": 3, "name": "car"},
    {"color": [0, 0, 230], "isthing": 1, "id": 4, "name": "motorcycle"},
    {"color": [106, 0, 228], "isthing": 1, "id": 5, "name": "airplane"},
    {"color": [0, 60, 100], "isthing": 1, "id": 6, "name": "bus"},
    {"color": [0, 80, 100], "isthing": 1, "id": 7, "name": "train"},
    {"color": [0, 0, 70], "isthing": 1, "id": 8, "name": "truck"},
    {"color": [0, 0, 192], "isthing": 1, "id": 9, "name": "boat"},
    {"color": [250, 170, 30], "isthing": 1, "id": 10, "name": "traffic light"},
    {"color": [100, 170, 30], "isthing": 1, "id": 11, "name": "fire hydrant"},
    {"color": [220, 220, 0], "isthing": 1, "id": 13, "name": "stop sign"},
    {"color": [175, 116, 175], "isthing": 1, "id": 14, "name": "parking meter"},
    {"color": [250, 0, 30], "isthing": 1, "id": 15, "name": "bench"},
    {"color": [165, 42, 42], "isthing": 1, "id": 16, "name": "bird"},
    {"color": [255, 77, 255], "isthing": 1, "id": 17, "name": "cat"},
    {"color": [0, 226, 252], "isthing": 1, "id": 18, "name": "dog"},
    {"color": [182, 182, 255], "isthing": 1, "id": 19, "name": "horse"},
    {"color": [0, 82, 0], "isthing": 1, "id": 20, "name": "sheep"},
    {"color": [120, 166, 157], "isthing": 1, "id": 21, "name": "cow"},
    {"color": [110, 76, 0], "isthing": 1, "id": 22, "name": "elephant"},
    {"color": [174, 57, 255], "isthing": 1, "id": 23, "name": "bear"},
    {"color": [199, 100, 0], "isthing": 1, "id": 24, "name": "zebra"},
    {"color": [72, 0, 118], "isthing": 1, "id": 25, "name": "giraffe"},
    {"color": [255, 179, 240], "isthing": 1, "id": 27, "name": "backpack"},
    {"color": [0, 125, 92], "isthing": 1, "id": 28, "name": "umbrella"},
    {"color": [209, 0, 151], "isthing": 1, "id": 31, "name": "handbag"},
    {"color": [188, 208, 182], "isthing": 1, "id": 32, "name": "tie"},
    {"color": [0, 220, 176], "isthing": 1, "id": 33, "name": "suitcase"},
    {"color": [255, 99, 164], "isthing": 1, "id": 34, "name": "frisbee"},
    {"color": [92, 0, 73], "isthing": 1, "id": 35, "name": "skis"},
    {"color": [133, 129, 255], "isthing": 1, "id": 36, "name": "snowboard"},
    {"color": [78, 180, 255], "isthing": 1, "id": 37, "name": "sports ball"},
    {"color": [0, 228, 0], "isthing": 1, "id": 38, "name": "kite"},
    {"color": [174, 255, 243], "isthing": 1, "id": 39, "name": "baseball bat"},
    {"color": [45, 89, 255], "isthing": 1, "id": 40, "name": "baseball glove"},
    {"color": [134, 134, 103], "isthing": 1, "id": 41, "name": "skateboard"},
    {"color": [145, 148, 174], "isthing": 1, "id": 42, "name": "surfboard"},
    {"color": [255, 208, 186], "isthing": 1, "id": 43, "name": "tennis racket"},
    {"color": [197, 226, 255], "isthing": 1, "id": 44, "name": "bottle"},
    {"color": [171, 134, 1], "isthing": 1, "id": 46, "name": "wine glass"},
    {"color": [109, 63, 54], "isthing": 1, "id": 47, "name": "cup"},
    {"color": [207, 138, 255], "isthing": 1, "id": 48, "name": "fork"},
    {"color": [151, 0, 95], "isthing": 1, "id": 49, "name": "knife"},
    {"color": [9, 80, 61], "isthing": 1, "id": 50, "name": "spoon"},
    {"color": [84, 105, 51], "isthing": 1, "id": 51, "name": "bowl"},
    {"color": [74, 65, 105], "isthing": 1, "id": 52, "name": "banana"},
    {"color": [166, 196, 102], "isthing": 1, "id": 53, "name": "apple"},
    {"color": [208, 195, 210], "isthing": 1, "id": 54, "name": "sandwich"},
    {"color": [255, 109, 65], "isthing": 1, "id": 55, "name": "orange"},
    {"color": [0, 143, 149], "isthing": 1, "id": 56, "name": "broccoli"},
    {"color": [179, 0, 194], "isthing": 1, "id": 57, "name": "carrot"},
    {"color": [209, 99, 106], "isthing": 1, "id": 58, "name": "hot dog"},
    {"color": [5, 121, 0], "isthing": 1, "id": 59, "name": "pizza"},
    {"color": [227, 255, 205], "isthing": 1, "id": 60, "name": "donut"},
    {"color": [147, 186, 208], "isthing": 1, "id": 61, "name": "cake"},
    {"color": [153, 69, 1], "isthing": 1, "id": 62, "name": "chair"},
    {"color": [3, 95, 161], "isthing": 1, "id": 63, "name": "couch"},
    {"color": [163, 255, 0], "isthing": 1, "id": 64, "name": "potted plant"},
    {"color": [119, 0, 170], "isthing": 1, "id": 65, "name": "bed"},
    {"color": [0, 182, 199], "isthing": 1, "id": 67, "name": "dining table"},
    {"color": [0, 165, 120], "isthing": 1, "id": 70, "name": "toilet"},
    {"color": [183, 130, 88], "isthing": 1, "id": 72, "name": "tv"},
    {"color": [95, 32, 0], "isthing": 1, "id": 73, "name": "laptop"},
    {"color": [130, 114, 135], "isthing": 1, "id": 74, "name": "mouse"},
    {"color": [110, 129, 133], "isthing": 1, "id": 75, "name": "remote"},
    {"color": [166, 74, 118], "isthing": 1, "id": 76, "name": "keyboard"},
    {"color": [219, 142, 185], "isthing": 1, "id": 77, "name": "cell phone"},
    {"color": [79, 210, 114], "isthing": 1, "id": 78, "name": "microwave"},
    {"color": [178, 90, 62], "isthing": 1, "id": 79, "name": "oven"},
    {"color": [65, 70, 15], "isthing": 1, "id": 80, "name": "toaster"},
    {"color": [127, 167, 115], "isthing": 1, "id": 81, "name": "sink"},
    {"color": [59, 105, 106], "isthing": 1, "id": 82, "name": "refrigerator"},
    {"color": [142, 108, 45], "isthing": 1, "id": 84, "name": "book"},
    {"color": [196, 172, 0], "isthing": 1, "id": 85, "name": "clock"},
    {"color": [95, 54, 80], "isthing": 1, "id": 86, "name": "vase"},
    {"color": [128, 76, 255], "isthing": 1, "id": 87, "name": "scissors"},
    {"color": [201, 57, 1], "isthing": 1, "id": 88, "name": "teddy bear"},
    {"color": [246, 0, 122], "isthing": 1, "id": 89, "name": "hair drier"},
    {"color": [191, 162, 208], "isthing": 1, "id": 90, "name": "toothbrush"},
    {"color": [255, 255, 128], "isthing": 0, "id": 92, "name": "banner"},
    {"color": [147, 211, 203], "isthing": 0, "id": 93, "name": "blanket"},
    {"color": [150, 100, 100], "isthing": 0, "id": 95, "name": "bridge"},
    {"color": [168, 171, 172], "isthing": 0, "id": 100, "name": "cardboard"},
    {"color": [146, 112, 198], "isthing": 0, "id": 107, "name": "counter"},
    {"color": [210, 170, 100], "isthing": 0, "id": 109, "name": "curtain"},
    {"color": [92, 136, 89], "isthing": 0, "id": 112, "name": "door-stuff"},
    {"color": [218, 88, 184], "isthing": 0, "id": 118, "name": "floor-wood"},
    {"color": [241, 129, 0], "isthing": 0, "id": 119, "name": "flower"},
    {"color": [217, 17, 255], "isthing": 0, "id": 122, "name": "fruit"},
    {"color": [124, 74, 181], "isthing": 0, "id": 125, "name": "gravel"},
    {"color": [70, 70, 70], "isthing": 0, "id": 128, "name": "house"},
    {"color": [255, 228, 255], "isthing": 0, "id": 130, "name": "light"},
    {"color": [154, 208, 0], "isthing": 0, "id": 133, "name": "mirror-stuff"},
    {"color": [193, 0, 92], "isthing": 0, "id": 138, "name": "net"},
    {"color": [76, 91, 113], "isthing": 0, "id": 141, "name": "pillow"},
    {"color": [255, 180, 195], "isthing": 0, "id": 144, "name": "platform"},
    {"color": [106, 154, 176], "isthing": 0, "id": 145, "name": "playingfield"},
    {"color": [230, 150, 140], "isthing": 0, "id": 147, "name": "railroad"},
    {"color": [60, 143, 255], "isthing": 0, "id": 148, "name": "river"},
    {"color": [128, 64, 128], "isthing": 0, "id": 149, "name": "road"},
    {"color": [92, 82, 55], "isthing": 0, "id": 151, "name": "roof"},
    {"color": [254, 212, 124], "isthing": 0, "id": 154, "name": "sand"},
    {"color": [73, 77, 174], "isthing": 0, "id": 155, "name": "sea"},
    {"color": [255, 160, 98], "isthing": 0, "id": 156, "name": "shelf"},
    {"color": [255, 255, 255], "isthing": 0, "id": 159, "name": "snow"},
    {"color": [104, 84, 109], "isthing": 0, "id": 161, "name": "stairs"},
    {"color": [169, 164, 131], "isthing": 0, "id": 166, "name": "tent"},
    {"color": [225, 199, 255], "isthing": 0, "id": 168, "name": "towel"},
    {"color": [137, 54, 74], "isthing": 0, "id": 171, "name": "wall-brick"},
    {"color": [135, 158, 223], "isthing": 0, "id": 175, "name": "wall-stone"},
    {"color": [7, 246, 231], "isthing": 0, "id": 176, "name": "wall-tile"},
    {"color": [107, 255, 200], "isthing": 0, "id": 177, "name": "wall-wood"},
    {"color": [58, 41, 149], "isthing": 0, "id": 178, "name": "water-other"},
    {"color": [183, 121, 142], "isthing": 0, "id": 180, "name": "window-blind"},
    {"color": [255, 73, 97], "isthing": 0, "id": 181, "name": "window-other"},
    {"color": [107, 142, 35], "isthing": 0, "id": 184, "name": "tree-merged"},
    {"color": [190, 153, 153], "isthing": 0, "id": 185, "name": "fence-merged"},
    {"color": [146, 139, 141], "isthing": 0, "id": 186, "name": "ceiling-merged"},
    {"color": [70, 130, 180], "isthing": 0, "id": 187, "name": "sky-other-merged"},
    {"color": [134, 199, 156], "isthing": 0, "id": 188, "name": "cabinet-merged"},
    {"color": [209, 226, 140], "isthing": 0, "id": 189, "name": "table-merged"},
    {"color": [96, 36, 108], "isthing": 0, "id": 190, "name": "floor-other-merged"},
    {"color": [96, 96, 96], "isthing": 0, "id": 191, "name": "pavement-merged"},
    {"color": [64, 170, 64], "isthing": 0, "id": 192, "name": "mountain-merged"},
    {"color": [152, 251, 152], "isthing": 0, "id": 193, "name": "grass-merged"},
    {"color": [208, 229, 228], "isthing": 0, "id": 194, "name": "dirt-merged"},
    {"color": [206, 186, 171], "isthing": 0, "id": 195, "name": "paper-merged"},
    {"color": [152, 161, 64], "isthing": 0, "id": 196, "name": "food-other-merged"},
    {"color": [116, 112, 0], "isthing": 0, "id": 197, "name": "building-other-merged"},
    {"color": [0, 114, 143], "isthing": 0, "id": 198, "name": "rock-merged"},
    {"color": [102, 102, 156], "isthing": 0, "id": 199, "name": "wall-other-merged"},
    {"color": [250, 141, 255], "isthing": 0, "id": 200, "name": "rug-merged"},
]

YTVIS_CATEGORIES_2019 = [
    {"color": [220, 20, 60], "isthing": 1, "id": 1, "name": "person"},
    {"color": [0, 82, 0], "isthing": 1, "id": 2, "name": "giant_panda"},
    {"color": [119, 11, 32], "isthing": 1, "id": 3, "name": "lizard"},
    {"color": [165, 42, 42], "isthing": 1, "id": 4, "name": "parrot"},
    {"color": [134, 134, 103], "isthing": 1, "id": 5, "name": "skateboard"},
    {"color": [0, 0, 142], "isthing": 1, "id": 6, "name": "sedan"},
    {"color": [255, 109, 65], "isthing": 1, "id": 7, "name": "ape"},
    {"color": [0, 226, 252], "isthing": 1, "id": 8, "name": "dog"},
    {"color": [5, 121, 0], "isthing": 1, "id": 9, "name": "snake"},
    {"color": [0, 60, 100], "isthing": 1, "id": 10, "name": "monkey"},
    {"color": [250, 170, 30], "isthing": 1, "id": 11, "name": "hand"},
    {"color": [100, 170, 30], "isthing": 1, "id": 12, "name": "rabbit"},
    {"color": [179, 0, 194], "isthing": 1, "id": 13, "name": "duck"},
    {"color": [255, 77, 255], "isthing": 1, "id": 14, "name": "cat"},
    {"color": [120, 166, 157], "isthing": 1, "id": 15, "name": "cow"},
    {"color": [73, 77, 174], "isthing": 1, "id": 16, "name": "fish"},
    {"color": [0, 80, 100], "isthing": 1, "id": 17, "name": "train"},
    {"color": [182, 182, 255], "isthing": 1, "id": 18, "name": "horse"},
    {"color": [0, 143, 149], "isthing": 1, "id": 19, "name": "turtle"},
    {"color": [174, 57, 255], "isthing": 1, "id": 20, "name": "bear"},
    {"color": [0, 0, 230], "isthing": 1, "id": 21, "name": "motorbike"},
    {"color": [72, 0, 118], "isthing": 1, "id": 22, "name": "giraffe"},
    {"color": [255, 179, 240], "isthing": 1, "id": 23, "name": "leopard"},
    {"color": [0, 125, 92], "isthing": 1, "id": 24, "name": "fox"},
    {"color": [209, 0, 151], "isthing": 1, "id": 25, "name": "deer"},
    {"color": [188, 208, 182], "isthing": 1, "id": 26, "name": "owl"},
    {"color": [145, 148, 174], "isthing": 1, "id": 27, "name": "surfboard"},
    {"color": [106, 0, 228], "isthing": 1, "id": 28, "name": "airplane"},
    {"color": [0, 0, 70], "isthing": 1, "id": 29, "name": "truck"},
    {"color": [199, 100, 0], "isthing": 1, "id": 30, "name": "zebra"},
    {"color": [166, 196, 102], "isthing": 1, "id": 31, "name": "tiger"},
    {"color": [110, 76, 0], "isthing": 1, "id": 32, "name": "elephant"},
    {"color": [133, 129, 255], "isthing": 1, "id": 33, "name": "snowboard"},
    {"color": [0, 0, 192], "isthing": 1, "id": 34, "name": "boat"},
    {"color": [183, 130, 88], "isthing": 1, "id": 35, "name": "shark"},
    {"color": [130, 114, 135], "isthing": 1, "id": 36, "name": "mouse"},
    {"color": [107, 142, 35], "isthing": 1, "id": 37, "name": "frog"},
    {"color": [0, 228, 0], "isthing": 1, "id": 38, "name": "eagle"},
    {"color": [174, 255, 243], "isthing": 1, "id": 39, "name": "earless_seal"},
    {"color": [255, 208, 186], "isthing": 1, "id": 40, "name": "tennis_racket"},
]


YTVIS_CATEGORIES_2021 = [
    {"color": [106, 0, 228], "isthing": 1, "id": 1, "name": "airplane"},
    {"color": [174, 57, 255], "isthing": 1, "id": 2, "name": "bear"},
    {"color": [255, 109, 65], "isthing": 1, "id": 3, "name": "bird"},
    {"color": [0, 0, 192], "isthing": 1, "id": 4, "name": "boat"},
    {"color": [0, 0, 142], "isthing": 1, "id": 5, "name": "car"},
    {"color": [255, 77, 255], "isthing": 1, "id": 6, "name": "cat"},
    {"color": [120, 166, 157], "isthing": 1, "id": 7, "name": "cow"},
    {"color": [209, 0, 151], "isthing": 1, "id": 8, "name": "deer"},
    {"color": [0, 226, 252], "isthing": 1, "id": 9, "name": "dog"},
    {"color": [179, 0, 194], "isthing": 1, "id": 10, "name": "duck"},
    {"color": [174, 255, 243], "isthing": 1, "id": 11, "name": "earless_seal"},
    {"color": [110, 76, 0], "isthing": 1, "id": 12, "name": "elephant"},
    {"color": [73, 77, 174], "isthing": 1, "id": 13, "name": "fish"},
    {"color": [250, 170, 30], "isthing": 1, "id": 14, "name": "flying_disc"},
    {"color": [0, 125, 92], "isthing": 1, "id": 15, "name": "fox"},
    {"color": [107, 142, 35], "isthing": 1, "id": 16, "name": "frog"},
    {"color": [0, 82, 0], "isthing": 1, "id": 17, "name": "giant_panda"},
    {"color": [72, 0, 118], "isthing": 1, "id": 18, "name": "giraffe"},
    {"color": [182, 182, 255], "isthing": 1, "id": 19, "name": "horse"},
    {"color": [255, 179, 240], "isthing": 1, "id": 20, "name": "leopard"},
    {"color": [119, 11, 32], "isthing": 1, "id": 21, "name": "lizard"},
    {"color": [0, 60, 100], "isthing": 1, "id": 22, "name": "monkey"},
    {"color": [0, 0, 230], "isthing": 1, "id": 23, "name": "motorbike"},
    {"color": [130, 114, 135], "isthing": 1, "id": 24, "name": "mouse"},
    {"color": [165, 42, 42], "isthing": 1, "id": 25, "name": "parrot"},
    {"color": [220, 20, 60], "isthing": 1, "id": 26, "name": "person"},
    {"color": [100, 170, 30], "isthing": 1, "id": 27, "name": "rabbit"},
    {"color": [183, 130, 88], "isthing": 1, "id": 28, "name": "shark"},
    {"color": [134, 134, 103], "isthing": 1, "id": 29, "name": "skateboard"},
    {"color": [5, 121, 0], "isthing": 1, "id": 30, "name": "snake"},
    {"color": [133, 129, 255], "isthing": 1, "id": 31, "name": "snowboard"},
    {"color": [188, 208, 182], "isthing": 1, "id": 32, "name": "squirrel"},
    {"color": [145, 148, 174], "isthing": 1, "id": 33, "name": "surfboard"},
    {"color": [255, 208, 186], "isthing": 1, "id": 34, "name": "tennis_racket"},
    {"color": [166, 196, 102], "isthing": 1, "id": 35, "name": "tiger"},
    {"color": [0, 80, 100], "isthing": 1, "id": 36, "name": "train"},
    {"color": [0, 0, 70], "isthing": 1, "id": 37, "name": "truck"},
    {"color": [0, 143, 149], "isthing": 1, "id": 38, "name": "turtle"},
    {"color": [0, 228, 0], "isthing": 1, "id": 39, "name": "whale"},
    {"color": [199, 100, 0], "isthing": 1, "id": 40, "name": "zebra"},
]


OVIS_CATEGORIES = [
    {"color": [220, 20, 60], "isthing": 1, "id": 1, "name": "Person"},
    {"color": [255, 109, 65], "isthing": 1, "id": 2, "name": "Bird"},
    {"color": [255, 77, 255], "isthing": 1, "id": 3, "name": "Cat"},
    {"color": [0, 226, 252], "isthing": 1, "id": 4, "name": "Dog"},
    {"color": [182, 182, 255], "isthing": 1, "id": 5, "name": "Horse"},
    {"color": [0, 125, 92], "isthing": 1, "id": 6, "name": "Sheep"},   # new, fox color
    {"color": [120, 166, 157], "isthing": 1, "id": 7, "name": "Cow"},
    {"color": [110, 76, 0], "isthing": 1, "id": 8, "name": "Elephant"},
    {"color": [174, 57, 255], "isthing": 1, "id": 9, "name": "Bear"},
    {"color": [199, 100, 0], "isthing": 1, "id": 10, "name": "Zebra"},
    {"color": [72, 0, 118], "isthing": 1, "id": 11, "name": "Giraffe"},
    {"color": [179, 0, 194], "isthing": 1, "id": 12, "name": "Poultry"},  # new, duck color
    {"color": [0, 82, 0], "isthing": 1, "id": 13, "name": "Giant_panda"},
    {"color": [119, 11, 32], "isthing": 1, "id": 14, "name": "Lizard"},
    {"color": [165, 42, 42], "isthing": 1, "id": 15, "name": "Parrot"},
    {"color": [0, 60, 100], "isthing": 1, "id": 16, "name": "Monkey"},
    {"color": [100, 170, 30], "isthing": 1, "id": 17, "name": "Rabbit"},
    {"color": [166, 196, 102], "isthing": 1, "id": 18, "name": "Tiger"},
    {"color": [73, 77, 174], "isthing": 1, "id": 19, "name": "Fish"},
    {"color": [0, 143, 149], "isthing": 1, "id": 20, "name": "Turtle"},
    {"color": [134, 134, 103], "isthing": 1, "id": 21, "name": "Bicycle"},  # new, skateboard color
    {"color": [0, 0, 230], "isthing": 1, "id": 22, "name": "Motorcycle"},
    {"color": [106, 0, 228], "isthing": 1, "id": 23, "name": "Airplane"},
    {"color": [0, 0, 192], "isthing": 1, "id": 24, "name": "Boat"},
    {"color": [0, 0, 70], "isthing": 1, "id": 25, "name": "Vehical"},  # new, truck color

]


def _get_ytvis_2019_instances_meta_ov():
    YTVIS_CATEGORIES_2019_OV = get_ytvis19_categories_with_prompt_eng()
    thing_ids = [k["id"] for k in YTVIS_CATEGORIES_2019_OV if k["isthing"] == 1]
    thing_colors = [k["color"] for k in YTVIS_CATEGORIES_2019_OV if k["isthing"] == 1]
    assert len(thing_ids) == 40, len(thing_ids)
    # Mapping from the incontiguous YTVIS category id to an id in [0, 39]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes_ov = [k["name"] for k in YTVIS_CATEGORIES_2019_OV if k["isthing"] == 1]
    thing_classes = [k["name"] for k in YTVIS_CATEGORIES_2019 if k["isthing"] == 1]

    stuff_classes = []
    stuff_colors = []
    stuff_dataset_id_to_contiguous_id = {}

    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "classes_ov": thing_classes_ov,
        "thing_colors": thing_colors,
        "stuff_classes": stuff_classes,
        "stuff_colors": stuff_colors,
        "stuff_dataset_id_to_contiguous_id": stuff_dataset_id_to_contiguous_id,
    }
    return ret

def _get_ytvis_2021_instances_meta_ov():
    YTVIS_CATEGORIES_2021_OV = get_ytvis21_categories_with_prompt_eng()
    thing_ids = [k["id"] for k in YTVIS_CATEGORIES_2021_OV if k["isthing"] == 1]
    thing_colors = [k["color"] for k in YTVIS_CATEGORIES_2021_OV if k["isthing"] == 1]
    assert len(thing_ids) == 40, len(thing_ids)
    # Mapping from the incontiguous YTVIS category id to an id in [0, 39]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes_ov = [k["name"] for k in YTVIS_CATEGORIES_2021_OV if k["isthing"] == 1]
    thing_classes = [k["name"] for k in YTVIS_CATEGORIES_2021 if k["isthing"] == 1]

    stuff_classes = []
    stuff_colors = []
    stuff_dataset_id_to_contiguous_id = {}

    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "classes_ov": thing_classes_ov,
        "thing_colors": thing_colors,
        "stuff_classes": stuff_classes,
        "stuff_colors": stuff_colors,
        "stuff_dataset_id_to_contiguous_id": stuff_dataset_id_to_contiguous_id,
    }
    return ret

def _get_ovis_instances_meta_ov():
    OVIS_CATEGORIES_OV = get_ovis_categories_with_prompt_eng()
    thing_ids = [k["id"] for k in OVIS_CATEGORIES_OV if k["isthing"] == 1]
    thing_colors = [k["color"] for k in OVIS_CATEGORIES_OV if k["isthing"] == 1]
    assert len(thing_ids) == 25, len(thing_ids)
    # Mapping from the incontiguous YTVIS category id to an id in [0, 39]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes_ov = [k["name"] for k in OVIS_CATEGORIES_OV if k["isthing"] == 1]
    thing_classes = [k["name"] for k in OVIS_CATEGORIES if k["isthing"] == 1]

    stuff_classes = []
    stuff_colors = []
    stuff_dataset_id_to_contiguous_id = {}

    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "classes_ov": thing_classes_ov,
        "thing_colors": thing_colors,
        "stuff_classes": stuff_classes,
        "stuff_colors": stuff_colors,
        "stuff_dataset_id_to_contiguous_id": stuff_dataset_id_to_contiguous_id,
    }
    return ret

def _get_coco_instances_meta_ov():
    COCO_CATEGORIES_OV = get_coco_categories_with_prompt_eng()
    meta = {}
    thing_classes_ov = [k["name"] for k in COCO_CATEGORIES_OV if k["isthing"] == 1]
    thing_classes = [k["name"] for k in COCO_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in COCO_CATEGORIES_OV if k["isthing"] == 1]
    stuff_classes_ov = []
    stuff_classes = []
    stuff_colors = []

    meta["classes_ov"] = thing_classes_ov
    meta["thing_classes"] = thing_classes
    meta["thing_colors"] = thing_colors
    meta["stuff_classes"] = stuff_classes
    meta["stuff_colors"] = stuff_colors

    thing_dataset_id_to_contiguous_id = {}
    stuff_dataset_id_to_contiguous_id = {}
    contiguous_id_to_class_name = []

    idx = 0
    for i, cat in enumerate(COCO_CATEGORIES):
        if cat["isthing"]:
            thing_dataset_id_to_contiguous_id[cat["id"]] = idx
            idx += 1
            contiguous_id_to_class_name.append(cat["name"])

    meta["thing_dataset_id_to_contiguous_id"] = thing_dataset_id_to_contiguous_id
    meta["stuff_dataset_id_to_contiguous_id"] = stuff_dataset_id_to_contiguous_id
    meta["contiguous_id_to_class_name"] = contiguous_id_to_class_name

    return meta

def load_ytvis_json(json_file, image_root, dataset_name=None, extra_annotation_keys=None):
    from ..ytvis_api.ytvos import YTVOS

    timer = Timer()
    json_file = PathManager.get_local_path(json_file)
    with contextlib.redirect_stdout(io.StringIO()):
        ytvis_api = YTVOS(json_file)
    if timer.seconds() > 1:
        logger.info("Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds()))

    id_map = None
    if dataset_name is not None:
        meta = MetadataCatalog.get(dataset_name)
        cat_ids = sorted(ytvis_api.getCatIds())
        cats = ytvis_api.loadCats(cat_ids)
        # The categories in a custom json file may not be sorted.
        thing_classes = [c["name"] for c in sorted(cats, key=lambda x: x["id"])]
        meta.thing_classes = thing_classes

        # In COCO, certain category ids are artificially removed,
        # and by convention they are always ignored.
        # We deal with COCO's id issue and translate
        # the category ids to contiguous ids in [0, 80).

        # It works by looking at the "categories" field in the json, therefore
        # if users' own json also have incontiguous ids, we'll
        # apply this mapping as well but print a warning.
        if not (min(cat_ids) == 1 and max(cat_ids) == len(cat_ids)):
            if "coco" not in dataset_name:
                logger.warning(
                    """
Category ids in annotations are not in [1, #categories]! We'll apply a mapping for you.
"""
                )
        id_map = {v: i for i, v in enumerate(cat_ids)}
        meta.thing_dataset_id_to_contiguous_id = id_map

    # sort indices for reproducible results
    vid_ids = sorted(ytvis_api.vids.keys())
    # vids is a list of dicts, each looks something like:
    # {'license': 1,
    #  'flickr_url': ' ',
    #  'file_names': ['ff25f55852/00000.jpg', 'ff25f55852/00005.jpg', ..., 'ff25f55852/00175.jpg'],
    #  'height': 720,
    #  'width': 1280,
    #  'length': 36,
    #  'date_captured': '2019-04-11 00:55:41.903902',
    #  'id': 2232}
    vids = ytvis_api.loadVids(vid_ids)

    anns = [ytvis_api.vidToAnns[vid_id] for vid_id in vid_ids]
    total_num_valid_anns = sum([len(x) for x in anns])
    total_num_anns = len(ytvis_api.anns)
    if total_num_valid_anns < total_num_anns:
        logger.warning(
            f"{json_file} contains {total_num_anns} annotations, but only "
            f"{total_num_valid_anns} of them match to images in the file."
        )

    vids_anns = list(zip(vids, anns))
    logger.info("Loaded {} videos in YTVIS format from {}".format(len(vids_anns), json_file))

    dataset_dicts = []

    ann_keys = ["iscrowd", "category_id", "id"] + (extra_annotation_keys or [])

    num_instances_without_valid_segmentation = 0

    for (vid_dict, anno_dict_list) in vids_anns:
        record = {}
        record["file_names"] = [os.path.join(image_root, vid_dict["file_names"][i]) for i in range(vid_dict["length"])]
        record["height"] = vid_dict["height"]
        record["width"] = vid_dict["width"]
        record["length"] = vid_dict["length"]
        video_id = record["video_id"] = vid_dict["id"]

        video_objs = []
        for frame_idx in range(record["length"]):
            frame_objs = []
            for anno in anno_dict_list:
                assert anno["video_id"] == video_id

                obj = {key: anno[key] for key in ann_keys if key in anno}

                _bboxes = anno.get("bboxes", None)
                _segm = anno.get("segmentations", None)

                if not (_bboxes and _segm and _bboxes[frame_idx] and _segm[frame_idx]):
                    continue

                bbox = _bboxes[frame_idx]
                segm = _segm[frame_idx]

                obj["bbox"] = bbox
                obj["bbox_mode"] = BoxMode.XYWH_ABS

                if isinstance(segm, dict):
                    if isinstance(segm["counts"], list):
                        # convert to compressed RLE
                        segm = mask_util.frPyObjects(segm, *segm["size"])
                elif segm:
                    # filter out invalid polygons (< 3 points)
                    segm = [poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6]
                    if len(segm) == 0:
                        num_instances_without_valid_segmentation += 1
                        continue  # ignore this instance
                obj["segmentation"] = segm

                if id_map:
                    obj["category_id"] = id_map[obj["category_id"]]
                frame_objs.append(obj)
            video_objs.append(frame_objs)
        record["annotations"] = video_objs
        dataset_dicts.append(record)

    if num_instances_without_valid_segmentation > 0:
        logger.warning(
            "Filtered out {} instances without valid segmentation. ".format(
                num_instances_without_valid_segmentation
            )
            + "There might be issues in your dataset generation process. "
            "A valid polygon should be a list[float] with even length >= 6."
        )
    return dataset_dicts


def register_ytvis_instances(name, metadata, json_file, image_root):
    """
    Register a dataset in YTVIS's json annotation format for
    instance tracking.

    Args:
        name (str): the name that identifies a dataset, e.g. "ytvis_train".
        metadata (dict): extra metadata associated with this dataset.  You can
            leave it as an empty dict.
        json_file (str): path to the json instance annotation file.
        image_root (str or path-like): directory which contains all the images.
    """
    assert isinstance(name, str), name
    assert isinstance(json_file, (str, os.PathLike)), json_file
    assert isinstance(image_root, (str, os.PathLike)), image_root
    # 1. register a function which returns dicts
    DatasetCatalog.register(name, lambda: load_ytvis_json(json_file, image_root, name))

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, evaluator_type="ytvis", **metadata
    )

_PREDEFINED_SPLITS_YTVIS_2019 = {
    "ytvis_2019_train_ov": ("ytvis_2019/train/JPEGImages",
                         "ytvis_2019/train.json"),
    "ytvis_2019_val_ov": ("ytvis_2019/valid/JPEGImages",
                       "ytvis_2019/valid.json"),
    "ytvis_2019_test_ov": ("ytvis_2019/test/JPEGImages",
                        "ytvis_2019/test.json"),
}


# ==== Predefined splits for YTVIS 2021 ===========
_PREDEFINED_SPLITS_YTVIS_2021 = {
    "ytvis_2021_train_ov": ("ytvis_2021/train/JPEGImages",
                         "ytvis_2021/train.json"),
    "ytvis_2021_val_ov": ("ytvis_2021/valid/JPEGImages",
                       "ytvis_2021/valid.json"),
    "ytvis_2023_val_ov": ("ytvis_2023/valid/JPEGImages",
                       "ytvis_2023/valid/instances.json"),
    "ytvis_2021_test_ov": ("ytvis_2021/test/JPEGImages",
                        "ytvis_2021/test.json"),
}

_PREDEFINED_SPLITS_OVIS = {
    "ovis_train_ov": ("ovis/train",
                         "ovis/annotations/annotations_train.json"),
    "ovis_val_ov": ("ovis/valid",
                       "ovis/annotations/annotations_valid.json"),
    "ovis_test_ov": ("ovis/test",
                        "ovis/annotations/annotations_test.json"),
}

_PREDEFINED_SPLITS_COCO_VIDEO = {
    "coco_instance_video_ov": ("coco/train2017", "coco/annotations/instances_train2017.json"),
}

def register_all_ytvis_2019_ov(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_YTVIS_2019.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_ytvis_2019_instances_meta_ov(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )

def register_all_coco_video_ov(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_COCO_VIDEO.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_coco_instances(
            key,
            _get_coco_instances_meta_ov(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )

def register_all_ytvis_2021_ov(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_YTVIS_2021.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_ytvis_2021_instances_meta_ov(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


def register_all_ovis_ov(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_OVIS.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_ovis_instances_meta_ov(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )

def get_ytvis19_categories_with_prompt_eng():
    COCO_CATEGORIES_ = copy.deepcopy(YTVIS_CATEGORIES_2019)
    coco_id_names = open('./dvis/data_video/datasets/ov_datasets/ytvis19_instance_with_prompt_eng.txt').read().splitlines()
    coco_idx = 0
    for line in coco_id_names:
        idx, name = line.split(':')
        idx = int(idx)
        if idx == 0 or name == "invalid_class_id":
            continue
        #print(COCO_CATEGORIES_[coco_idx]["name"], '->', name)
        assert COCO_CATEGORIES_[coco_idx]["id"] == idx
        assert COCO_CATEGORIES_[coco_idx]["name"] in name
        COCO_CATEGORIES_[coco_idx]["name"] = name
        coco_idx += 1
    return COCO_CATEGORIES_

def get_ytvis21_categories_with_prompt_eng():
    COCO_CATEGORIES_ = copy.deepcopy(YTVIS_CATEGORIES_2021)
    coco_id_names = open('./dvis/data_video/datasets/ov_datasets/ytvis21_instance_with_prompt_eng.txt').read().splitlines()
    coco_idx = 0
    for line in coco_id_names:
        idx, name = line.split(':')
        idx = int(idx)
        if idx == 0 or name == "invalid_class_id":
            continue
        #print(COCO_CATEGORIES_[coco_idx]["name"], '->', name)
        assert COCO_CATEGORIES_[coco_idx]["id"] == idx
        assert COCO_CATEGORIES_[coco_idx]["name"] in name
        COCO_CATEGORIES_[coco_idx]["name"] = name
        coco_idx += 1
    return COCO_CATEGORIES_

def get_ovis_categories_with_prompt_eng():
    COCO_CATEGORIES_ = copy.deepcopy(OVIS_CATEGORIES)
    coco_id_names = open('./dvis/data_video/datasets/ov_datasets/ovis_instance_with_prompt_eng.txt').read().splitlines()
    coco_idx = 0
    for line in coco_id_names:
        idx, name = line.split(':')
        idx = int(idx)
        if idx == 0 or name == "invalid_class_id":
            continue
        #print(COCO_CATEGORIES_[coco_idx]["name"], '->', name)
        assert COCO_CATEGORIES_[coco_idx]["id"] == idx
        COCO_CATEGORIES_[coco_idx]["name"] = name
        coco_idx += 1
    return COCO_CATEGORIES_

def get_coco_categories_with_prompt_eng():
    COCO_CATEGORIES_ = copy.deepcopy(COCO_CATEGORIES)
    coco_id_names = open('./dvis/data_video/datasets/ov_datasets/coco_panoptic_with_prompt_eng.txt').read().splitlines()
    coco_idx = 0
    for line in coco_id_names:
        idx, name = line.split(':')
        idx = int(idx)
        if idx == 0 or name == "invalid_class_id":
            continue
        #print(COCO_CATEGORIES_[coco_idx]["name"], '->', name)
        assert COCO_CATEGORIES_[coco_idx]["id"] == idx
        #assert COCO_CATEGORIES_[coco_idx]["name"] in name
        COCO_CATEGORIES_[coco_idx]["name"] = name
        coco_idx += 1
    # only for instance
    ret = []
    for item in COCO_CATEGORIES_:
        if item['isthing'] == 1:
            ret.append(item)
    return ret

_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_ytvis_2019_ov(_root)
register_all_ytvis_2021_ov(_root)
register_all_ovis_ov(_root)
register_all_coco_video_ov(_root)
