# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/sukjunhwang/IFC

import contextlib
import copy
import io
import itertools
import json
import logging
import numpy as np
import os
from collections import OrderedDict
import pycocotools.mask as mask_util
import torch
from .datasets.ytvis_api.ytvos import YTVOS
from .datasets.ytvis_api.ytvoseval import YTVOSeval
from tabulate import tabulate

import detectron2.utils.comm as comm
from detectron2.config import CfgNode
from detectron2.data import MetadataCatalog
from detectron2.evaluation import DatasetEvaluator
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import create_small_table

from PIL import Image


class VSSEvaluator(DatasetEvaluator):
    """
    Evaluate AR for object proposals, AP for instance detection/segmentation, AP
    for keypoint detection outputs using COCO's metrics.
    See http://cocodataset.org/#detection-eval and
    http://cocodataset.org/#keypoints-eval to understand its metrics.

    In addition to COCO, this evaluator is able to support any bounding box detection,
    instance segmentation, or keypoint detection dataset.
    """

    def __init__(
        self,
        dataset_name,
        tasks=None,
        distributed=True,
        output_dir=None,
        *,
        use_fast_impl=True,
    ):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
                It must have either the following corresponding metadata:

                    "json_file": the path to the COCO format annotation

                Or it must be in detectron2's standard dataset format
                so it can be converted to COCO format automatically.
            tasks (tuple[str]): tasks that can be evaluated under the given
                configuration. A task is one of "bbox", "segm", "keypoints".
                By default, will infer this automatically from predictions.
            distributed (True): if True, will collect results from all ranks and run evaluation
                in the main process.
                Otherwise, will only evaluate the results in the current process.
            output_dir (str): optional, an output directory to dump all
                results predicted on the dataset. The dump contains two files:

                1. "instances_predictions.pth" a file in torch serialization
                   format that contains all the raw original predictions.
                2. "coco_instances_results.json" a json file in COCO's result
                   format.
            use_fast_impl (bool): use a fast but **unofficial** implementation to compute AP.
                Although the results should be very close to the official implementation in COCO
                API, it is still recommended to compute results with the official API for use in
                papers. The faster implementation also uses more RAM.
        """
        self._logger = logging.getLogger(__name__)
        self._distributed = distributed
        self._output_dir = output_dir
        self._use_fast_impl = use_fast_impl

        if tasks is not None and isinstance(tasks, CfgNode):
            self._logger.warning(
                "COCO Evaluator instantiated using config, this is deprecated behavior."
                " Please pass in explicit arguments instead."
            )
            self._tasks = None  # Infering it from predictions should be better
        else:
            self._tasks = tasks

        self._cpu_device = torch.device("cpu")

        self._metadata = MetadataCatalog.get(dataset_name)
        dataset_id_to_contiguous_id = self._metadata.stuff_dataset_id_to_contiguous_id
        self.contiguous_id_to_dataset_id = {}
        for i, key in enumerate(dataset_id_to_contiguous_id.keys()):
            self.contiguous_id_to_dataset_id.update({i: key})

        # Test set json files do not contain annotations (evaluation must be
        # performed using the COCO evaluation server).
        self._do_evaluation = False

    def reset(self):
        self._predictions = []
        PathManager.mkdirs(self._output_dir)

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a COCO model (e.g., GeneralizedRCNN).
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a COCO model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """
        # outputs (T, W, H)
        assert len(inputs) == 1, "More than one inputs are loaded for inference!"

        video_id = inputs[0]["video_id"]
        image_names = [inputs[0]['file_names'][idx] for idx in inputs[0]["frame_idx"]]
        img_shape = outputs['image_size']
        sem_seg_result = outputs['pred_masks'].numpy()  # (t, h, w, 3)
        print(sem_seg_result.shape)
        sem_seg_result_ = np.zeros_like(sem_seg_result) + 255
        unique_cls = np.unique(sem_seg_result[:, :, :, 0])
        for cls in unique_cls:
            cls_ = self.contiguous_id_to_dataset_id[cls]
            sem_seg_result_[sem_seg_result == cls] = cls_
        sem_seg_result = sem_seg_result_
        for i, image_name in enumerate(image_names):
            image_ = Image.fromarray(sem_seg_result[i])
            if not os.path.exists(os.path.join(self._output_dir, video_id)):
                os.makedirs(os.path.join(self._output_dir, video_id))
            image_.save(os.path.join(self._output_dir, video_id, image_name.split('/')[-1].split('.')[0] + '.png'))
        return

    def evaluate(self):
        """
        Args:
            img_ids: a list of image IDs to evaluate on. Default to None for the whole dataset
        """
        return {}