# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import math
from typing import Tuple
import einops

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks

from mask2former_video.modeling.criterion import VideoSetCriterion
from mask2former_video.modeling.matcher import VideoHungarianMatcher, VideoHungarianMatcher_Consistent
from mask2former_video.utils.memory import retry_if_cuda_oom
from mask2former_video.modeling.transformer_decoder.video_mask2former_transformer_decoder import SelfAttentionLayer, CrossAttentionLayer, FFNLayer, MLP, _get_activation_fn
from mask2former_video.modeling.transformer_decoder.position_encoding import PositionEmbeddingSineTime
from .video_maskformer_model import QueryTracker_mine

from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)


@META_ARCH_REGISTRY.register()
class VideoMaskFormer_frame_offline(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        criterion: nn.Module,
        num_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        # video
        num_frames,
        window_inference,
        num_class,
        max_num,
        max_iter_num,
        # type
        panoptic_on=False,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            num_queries: int, number of queries
            object_mask_threshold: float, threshold to filter query based on classification score
                for panoptic segmentation inference
            overlap_threshold: overlap threshold used in general inference for panoptic segmentation
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            sem_seg_postprocess_before_inference: whether to resize the prediction back
                to original input size before semantic segmentation inference or after.
                For high-resolution dataset like Mapillary, resizing predictions before
                inference will cause OOM error.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            semantic_on: bool, whether to output semantic segmentation prediction
            instance_on: bool, whether to output instance segmentation prediction
            panoptic_on: bool, whether to output panoptic segmentation prediction
            test_topk_per_image: int, instance segmentation parameter, keep topk instances per image
        """
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        for p in self.sem_seg_head.parameters():
            p.requires_grad_(False)
        self.criterion = criterion
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.metadata = metadata
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        self.num_frames = num_frames
        self.window_inference = window_inference
        self.max_num = max_num

        self.tracker = QueryTracker_mine(
            hidden_channel=256,
            feedforward_channel=2048,
            num_head=8,
            decoder_layer_num=6,
            mask_dim=256,
            class_num=num_class,)
        for p in self.tracker.parameters():
            p.requires_grad_(False)

        self.offline_tracker = QueryTracker_offline(
        # self.offline_tracker = QueryTracker_offline_transCls(
            hidden_channel=256,
            feedforward_channel=2048,
            num_head=8,
            decoder_layer_num=6,
            mask_dim=256,
            class_num=num_class,
        )
        self.iter = 0
        self.max_iter_num = max_iter_num

        self.panoptic_on = panoptic_on
        self.keep = False

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

        # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

        # loss weights
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT
        contrast_weight = cfg.MODEL.MASK_FORMER.CONTRAST_WEIGHT

        # building criterion
        # matcher = VideoHungarianMatcher(
        #     cost_class=class_weight,
        #     cost_mask=mask_weight,
        #     cost_dice=dice_weight,
        #     num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        # )
        matcher = VideoHungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS * cfg.INPUT.SAMPLING_FRAME_NUM,
        )

        weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight,
                       "loss_contrast": contrast_weight}

        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "masks", "contrast"]

        criterion = VideoSetCriterion(
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS * cfg.INPUT.SAMPLING_FRAME_NUM,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
        )

        max_iter_num = cfg.SOLVER.MAX_ITER
        panoptic_on = cfg.MODEL.PANOPTIC_ON

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": True,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            # video
            "num_frames": cfg.INPUT.SAMPLING_FRAME_NUM,
            "window_inference": cfg.MODEL.MASK_FORMER.TEST.WINDOW_INFERENCE,
            "num_class": cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            "max_num": cfg.MODEL.MASK_FORMER.TEST.MAX_NUM,
            "max_iter_num": max_iter_num,
            "panoptic_on": panoptic_on,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def segmentor_windows_inference(self, images_tensor, window_size=5):
        image_outputs = {}
        iters = len(images_tensor) // window_size
        if len(images_tensor) % window_size != 0:
            iters += 1

        outs_list = []
        for i in range(iters):
            start_idx = i * window_size
            end_idx = (i + 1) * window_size

            features = self.backbone(images_tensor[start_idx:end_idx])
            out = self.sem_seg_head(features)

            del features['res2'], features['res3'], features['res4'], features['res5']
            del out['pred_masks'], out['pred_logits']
            for j in range(len(out['aux_outputs'])):
                del out['aux_outputs'][j]['pred_masks'], out['aux_outputs'][j]['pred_logits']
            outs_list.append(out)

        image_outputs['pred_embds'] = torch.cat([x['pred_embds'] for x in outs_list], dim=2).detach()
        image_outputs['mask_features'] = torch.cat([x['mask_features'] for x in outs_list], dim=0).detach()
        return image_outputs

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """

        if 'keep' in batched_inputs[0].keys():
            self.keep = batched_inputs[0]['keep']
        else:
            self.keep = False

        images = []
        for video in batched_inputs:
            for frame in video["image"]:
                images.append(frame.to(self.device))
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)
        self.backbone.eval()
        self.sem_seg_head.eval()
        self.tracker.eval()

        if not self.training and self.window_inference:
            outputs, online_pred_logits = self.run_window_inference(images.tensor, window_size=3)
        else:
            with torch.no_grad():
                # features = self.backbone(images.tensor)
                # image_outputs = self.sem_seg_head(features)
                # del features['res2'], features['res3'], features['res4'], features['res5']
                image_outputs = self.segmentor_windows_inference(images.tensor, window_size=21)
                frame_embds = image_outputs['pred_embds'].clone().detach()  # b c t q
                mask_features = image_outputs['mask_features'].clone().detach().unsqueeze(0)
                del image_outputs['mask_features']
                image_outputs = self.tracker(frame_embds, mask_features, resume=self.keep)
                online_pred_logits = image_outputs['pred_logits'] # (b, t, q, c)
                frame_embds_ = self.tracker.frame_forward(frame_embds)
                del frame_embds
                instance_embeds = image_outputs['pred_embds'].clone().detach()
                del image_outputs['pred_embds']
                for j in range(len(image_outputs['aux_outputs'])):
                    del image_outputs['aux_outputs'][j]['pred_masks'], image_outputs['aux_outputs'][j]['pred_logits']
                torch.cuda.empty_cache()
            outputs = self.offline_tracker(instance_embeds, frame_embds_, mask_features)

        # outputs['pred_embds'] = self.embed_proj(outputs['pred_embds'].detach().permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        #outputs['pred_embds'] = outputs['pred_embds']


        if self.training:
            # mask classification target
            targets = self.prepare_targets(batched_inputs, images)

            if self.iter < self.max_iter_num // 2:
                image_outputs, outputs, targets = self.frame_decoder_loss_reshape(outputs, targets,
                                                                                  image_outputs=image_outputs)
            else:
                image_outputs, outputs, targets = self.frame_decoder_loss_reshape(outputs, targets,
                                                                                  image_outputs=None)
            self.iter += 1

            # bipartite matching-based loss
            #losses = self.criterion(outputs, targets)
            losses = self.criterion(outputs, targets, matcher_outputs=image_outputs)

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
            return losses
        else:
            #outputs = self.post_processing(outputs)
            outputs, online_pred_logits = self.post_processing_(outputs, online_logits=online_pred_logits)

            mask_cls_results = outputs["pred_logits"]
            mask_pred_results = outputs["pred_masks"]
            pred_ids = outputs["ids"]

            mask_cls_result = mask_cls_results[0]
            mask_pred_result = mask_pred_results[0]
            pred_id = pred_ids[0]
            first_resize_size = (images.tensor.shape[-2], images.tensor.shape[-1])

            input_per_image = batched_inputs[0]
            image_size = images.image_sizes[0]  # image size without padding after data augmentation

            height = input_per_image.get("height", image_size[0])  # raw image size before data augmentation
            width = input_per_image.get("width", image_size[1])
            if self.panoptic_on:
                return retry_if_cuda_oom(self.inference_video_pano)(mask_cls_result, mask_pred_result,
                                                                    image_size, height, width,
                                                                    first_resize_size, pred_id,
                                                                    online_pred_cls=online_pred_logits)
            else:
                return retry_if_cuda_oom(self.inference_video)(mask_cls_result, mask_pred_result, image_size,
                                                               height, width, first_resize_size, pred_id,
                                                               online_pred_cls=online_pred_logits)

    def frame_decoder_loss_reshape(self, outputs, targets, image_outputs=None):
        outputs['pred_masks'] = einops.rearrange(outputs['pred_masks'], 'b q t h w -> b q () (t h) w')
        #outputs['pred_logits'] = einops.rearrange(outputs['pred_logits'], 'b t q c -> (b t) q c')
        outputs['pred_logits'] = outputs['pred_logits'][:, 0, :, :]
        if image_outputs is not None:
            image_outputs['pred_masks'] = einops.rearrange(image_outputs['pred_masks'], 'b q t h w -> b q () (t h) w')
            #image_outputs['pred_logits'] = einops.rearrange(image_outputs['pred_logits'], 'b t q c -> (b t) q c')
            image_outputs['pred_logits'] = image_outputs['pred_logits'].mean(dim=1)
        if 'aux_outputs' in outputs:
            for i in range(len(outputs['aux_outputs'])):
                outputs['aux_outputs'][i]['pred_masks'] = einops.rearrange(
                    outputs['aux_outputs'][i]['pred_masks'], 'b q t h w -> b q () (t h) w'
                )
                # outputs['aux_outputs'][i]['pred_logits'] = einops.rearrange(
                #     outputs['aux_outputs'][i]['pred_logits'], 'b t q c -> (b t) q c'
                # )
                outputs['aux_outputs'][i]['pred_logits'] = outputs['aux_outputs'][i]['pred_logits'][:, 0, :, :]

        gt_instances = []
        for targets_per_video in targets:
            targets_per_video['masks'] = einops.rearrange(
                targets_per_video['masks'], 'q t h w -> q () (t h) w'
                )
            gt_instances.append(targets_per_video)
        #     # labels: N (num instances)
        #     # ids: N, num_labeled_frames
        #     # masks: N, num_labeled_frames, H, W
        #     num_labeled_frames = targets_per_video['ids'].shape[1]
        #     for f in range(num_labeled_frames):
        #         labels = targets_per_video['labels']
        #         ids = targets_per_video['ids'][:, [f]]
        #         masks = targets_per_video['masks'][:, [f], :, :]
        #         gt_instances.append({"labels": labels, "ids": ids, "masks": masks})
        # # outputs -> {'masks': (bt, q, h, w), 'logits': (bt, 1, c)}
        # # gt_instances -> [per image gt * bt], per image gt -> {'labels': (N, ), 'ids': (N, ), 'masks': (N, H, W)}
        return image_outputs, outputs, gt_instances

    def post_processing_(self, outputs, online_logits=None):
        pred_logits, pred_masks, pred_embds = outputs['pred_logits'], outputs['pred_masks'], outputs['pred_embds']

        # pred_logits: 1 t q c
        # pred_masks: 1 q t h w
        # pred_embeds: 1 c t q
        pred_logits = pred_logits[0]
        if online_logits is not None:
            online_logits = online_logits[0] # (t, q, c)
        pred_scores = torch.max(F.softmax(pred_logits, dim=-1)[..., :-1], dim=-1)[0]
        pred_masks = einops.rearrange(pred_masks[0], 'q t h w -> t q h w')
        pred_embds = einops.rearrange(pred_embds[0], 'c t q -> t q c')

        pred_logits = list(torch.unbind(pred_logits))
        pred_masks = list(torch.unbind(pred_masks))
        pred_embds = list(torch.unbind(pred_embds))

        out_logits = []
        out_masks = []
        out_embds = []
        out_scores = []
        out_logits.append(pred_logits[0])
        out_masks.append(pred_masks[0])
        out_embds += [pred_embds[0]] * 3
        out_scores += [pred_scores[0]] * 3

        for i in range(1, len(pred_logits)):
            # indices = self.match_from_embds_(out_embds[-3:], pred_embds[i], scores=out_scores[-3:])
            # indices = self.match_from_embds_(out_embds[-3:], pred_embds[i])

            # out_logits.append(pred_logits[i][indices, :])
            # out_masks.append(pred_masks[i][indices, :, :])
            # out_embds.append(pred_embds[i][indices, :])
            # out_scores.append(pred_scores[i][indices])

            out_logits.append(pred_logits[i])
            out_masks.append(pred_masks[i])
            out_embds.append(pred_embds[i])
            out_scores.append(pred_scores[i])

        out_logits = sum(out_logits)/len(out_logits)
        out_masks = torch.stack(out_masks, dim=1)  # q h w -> q t h w

        if online_logits is not None:
            online_logits = torch.mean(online_logits, dim=0) # (q, c)

        out_logits = out_logits.unsqueeze(0)
        out_masks = out_masks.unsqueeze(0)

        outputs['pred_logits'] = out_logits
        outputs['pred_masks'] = out_masks
        outputs['ids'] = [torch.arange(0, out_masks.size(1))]

        return outputs, online_logits

    def run_window_inference(self, images_tensor, window_size=30):
        iters = len(images_tensor) // window_size
        if len(images_tensor) % window_size != 0:
            iters += 1

        overall_mask_features = []
        overall_frame_embds = []
        overall_instance_embds = []

        online_pred_logits = []

        for i in range(iters):
            start_idx = i * window_size
            end_idx = (i+1) * window_size

            features = self.backbone(images_tensor[start_idx:end_idx])
            out = self.sem_seg_head(features)
            del features['res2'], features['res3'], features['res4'], features['res5']
            del out['pred_masks'], out['pred_logits']
            for j in range(len(out['aux_outputs'])):
                del out['aux_outputs'][j]['pred_masks'], out['aux_outputs'][j]['pred_logits']

            frame_embds = out['pred_embds']  # b c t q
            mask_features = out['mask_features'].unsqueeze(0)

            overall_mask_features.append(mask_features.cpu())
            overall_frame_embds.append(frame_embds)

            if i != 0 or self.keep:
                track_out = self.tracker(frame_embds, mask_features, resume=True)
            else:
                track_out = self.tracker(frame_embds, mask_features)

            online_pred_logits.append(track_out['pred_logits'].clone())
            del track_out['pred_masks'], track_out['pred_logits']
            for j in range(len(track_out['aux_outputs'])):
                del track_out['aux_outputs'][j]['pred_masks'], track_out['aux_outputs'][j]['pred_logits']

            instance_embds = track_out['pred_embds']
            overall_instance_embds.append(instance_embds)

        overall_frame_embds = torch.cat(overall_frame_embds, dim=2)
        overall_instance_embds = torch.cat(overall_instance_embds, dim=2)
        overall_mask_features = torch.cat(overall_mask_features, dim=1)

        online_pred_logits = torch.cat(online_pred_logits, dim=1)

        overall_frame_embds_ = self.tracker.frame_forward(overall_frame_embds)
        del overall_frame_embds

        # merge outputs
        outputs = self.offline_tracker(overall_instance_embds, overall_frame_embds_, overall_mask_features)
        return outputs, online_pred_logits

    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        gt_instances = []
        for targets_per_video in targets:
            _num_instance = len(targets_per_video["instances"][0]) # targets_per_video["instances"] -> [per frame [per instance id]]
            mask_shape = [_num_instance, self.num_frames, h_pad, w_pad]
            gt_masks_per_video = torch.zeros(mask_shape, dtype=torch.bool, device=self.device)

            gt_ids_per_video = []
            for f_i, targets_per_frame in enumerate(targets_per_video["instances"]):
                targets_per_frame = targets_per_frame.to(self.device)
                h, w = targets_per_frame.image_size

                gt_ids_per_video.append(targets_per_frame.gt_ids[:, None])
                if isinstance(targets_per_frame.gt_masks, BitMasks):
                    gt_masks_per_video[:, f_i, :h, :w] = targets_per_frame.gt_masks.tensor
                else:  # polygon
                    gt_masks_per_video[:, f_i, :h, :w] = targets_per_frame.gt_masks

            gt_ids_per_video = torch.cat(gt_ids_per_video, dim=1) #(n_inst, frame)
            valid_idx = (gt_ids_per_video != -1).any(dim=-1) # filter which instance in any frames

            gt_classes_per_video = targets_per_frame.gt_classes[valid_idx]          # N,
            gt_ids_per_video = gt_ids_per_video[valid_idx]                          # N, num_frames, i instance not in j frame, id[i, j] = -1

            gt_instances.append({"labels": gt_classes_per_video, "ids": gt_ids_per_video})
            gt_masks_per_video = gt_masks_per_video[valid_idx].float()          # N, num_frames, H, W
            gt_instances[-1].update({"masks": gt_masks_per_video})
        #gt_instances -> [per video instance], per video instance {'labels': (N, ), 'ids': (N, f), 'masks': (N, f, H, W)}
        return gt_instances

    def inference_video_pano(self, pred_cls, pred_masks, img_size, output_height, output_width,
                             first_resize_size, pred_id, online_pred_cls=None):
        # pred_cls (N, C)
        # pred_masks (N, T, H, W)
        pred_cls_ = F.softmax(pred_cls, dim=-1)
        if online_pred_cls is not None:
            online_pred_cls = F.softmax(online_pred_cls, dim=-1)[:, :-1]
            pred_cls_[:, :-1] = torch.maximum(pred_cls_[:, :-1], online_pred_cls.to(pred_cls_))
        mask_pred = pred_masks.sigmoid()
        scores, labels = pred_cls_.max(-1)

        keep = labels.ne(self.sem_seg_head.num_classes) & (scores > self.object_mask_threshold)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_ids = pred_id[keep]
        cur_masks = mask_pred[keep]
        cur_mask_cls = pred_cls[keep]
        cur_mask_cls = cur_mask_cls[:, :-1]

        cur_masks = F.interpolate(
            cur_masks, size=first_resize_size, mode="bilinear", align_corners=False
        )

        cur_masks = cur_masks[:, :, :img_size[0], :img_size[1]]
        cur_masks = F.interpolate(
            cur_masks, size=(output_height, output_width), mode="bilinear", align_corners=False
        )

        cur_prob_masks = cur_scores.view(-1, 1, 1, 1).to(cur_masks.device) * cur_masks

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((cur_masks.size(1), h, w), dtype=torch.int32, device=cur_masks.device)
        segments_infos = []
        out_ids = []

        current_segment_id = 0

        if cur_masks.shape[0] == 0:
            # We didn't detect any mask :(
            return {
                "image_size": (output_height, output_width),
                'pred_masks': panoptic_seg.cpu(),
                'segments_infos': segments_infos
            }
        else:
            # take argmax
            cur_mask_ids = cur_prob_masks.argmax(0) # (T, H, W)
            stuff_memory_list = {}
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                isthing = pred_class < len(self.metadata.thing_dataset_id_to_contiguous_id)
                mask_area = (cur_mask_ids == k).sum().item()
                original_area = (cur_masks[k] >= 0.5).sum().item()
                mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)
                if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                    if mask_area / original_area < self.overlap_threshold:
                        continue
                    # merge stuff regions
                    if not isthing:
                        if int(pred_class) in stuff_memory_list.keys():
                            panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                            continue
                        else:
                            stuff_memory_list[int(pred_class)] = current_segment_id + 1
                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id

                    segments_infos.append(
                        {
                            "id": current_segment_id,
                            "isthing": bool(isthing),
                            "category_id": int(pred_class),
                        }
                    )
                    out_ids.append(cur_ids[k])

            return {
                    "image_size": (output_height, output_width),
                    'pred_masks': panoptic_seg.cpu(),
                    'segments_infos': segments_infos,
                    'pred_ids': out_ids
            }

    def inference_video(self, pred_cls, pred_masks, img_size, output_height, output_width,
                        first_resize_size, pred_id, online_pred_cls=None):
        if len(pred_cls) > 0:
            scores = F.softmax(pred_cls, dim=-1)[:, :-1]
            if online_pred_cls is not None:
                online_pred_cls = F.softmax(online_pred_cls, dim=-1)[:, :-1]
                scores = torch.maximum(scores, online_pred_cls.to(scores))
            labels = torch.arange(self.sem_seg_head.num_classes, device=self.device).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
            # keep top-10 predictions
            scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.max_num, sorted=False)
            labels_per_image = labels[topk_indices]
            topk_indices = topk_indices // self.sem_seg_head.num_classes
            pred_masks = pred_masks[topk_indices]
            pred_ids = pred_id[topk_indices]

            pred_masks = F.interpolate(
                pred_masks, size=first_resize_size, mode="bilinear", align_corners=False
            )

            pred_masks = pred_masks[:, :, : img_size[0], : img_size[1]]
            pred_masks = F.interpolate(
                pred_masks, size=(output_height, output_width), mode="bilinear", align_corners=False
            )

            masks = pred_masks > 0.

            out_scores = scores_per_image.tolist()
            out_labels = labels_per_image.tolist()
            out_ids = pred_ids.tolist()
            out_masks = [m for m in masks.cpu()]
        else:
            out_scores = []
            out_labels = []
            out_masks = []
            out_ids = []

        video_output = {
            "image_size": (output_height, output_width),
            "pred_scores": out_scores,
            "pred_labels": out_labels,
            "pred_masks": out_masks,
            "pred_ids": out_ids
        }

        return video_output

class QueryTracker_offline(torch.nn.Module):
    def __init__(self,
                 hidden_channel=256,
                 feedforward_channel=2048,
                 num_head=8,
                 decoder_layer_num=6,
                 mask_dim=256,
                 class_num=25,):
        super(QueryTracker_offline, self).__init__()

        # init transformer layers
        self.num_heads = num_head
        self.num_layers = decoder_layer_num
        self.transformer_obj_self_attention_layers = nn.ModuleList()
        self.transformer_time_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        self.conv_short_aggregate_layers = nn.ModuleList()
        self.conv_norms = nn.ModuleList()

        for _ in range(self.num_layers):
            self.transformer_time_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_channel,
                    nhead=num_head,
                    dropout=0.0,
                    normalize_before=False,
                )
            )

            self.conv_short_aggregate_layers.append(
                nn.Sequential(
                    nn.Conv1d(hidden_channel, hidden_channel, kernel_size=5, stride=1,
                              padding='same', padding_mode='replicate'),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(hidden_channel, hidden_channel, kernel_size=3, stride=1,
                              padding='same', padding_mode='replicate'),
                )
            )

            self.conv_norms.append(nn.LayerNorm(hidden_channel))

            self.transformer_obj_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_channel,
                    nhead=num_head,
                    dropout=0.0,
                    normalize_before=False,
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_channel,
                    nhead=num_head,
                    dropout=0.0,
                    normalize_before=False,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_channel,
                    dim_feedforward=feedforward_channel,
                    dropout=0.0,
                    normalize_before=False,
                )
            )

        self.decoder_norm = nn.LayerNorm(hidden_channel)
        N_steps = hidden_channel // 2
        self.pe_layer = PositionEmbeddingSineTime(N_steps, normalize=True)

        # init heads
        self.class_embed = nn.Linear(hidden_channel, class_num + 1)
        self.mask_embed = MLP(hidden_channel, hidden_channel, mask_dim, 3)

        self.activation_proj = nn.Linear(hidden_channel, 1)

    def forward(self, instance_embeds, frame_embeds, mask_features):
        # instance_embds (b, c, t, q)
        # frame_embds (b, c, t, q)
        n_batch, n_channel, n_frames, n_instance = instance_embeds.size()
        outputs = []
        time_embds = self.pe_layer(instance_embeds.permute(2, 0, 3, 1).flatten(1, 2))

        output = instance_embeds
        #instance_embeds = instance_embeds.permute(3, 0, 2, 1).flatten(1, 2)
        frame_embeds = frame_embeds.permute(3, 0, 2, 1).flatten(1, 2)

        for i in range(self.num_layers):
            output = output.permute(2, 0, 3, 1) #(t, b, q, c)
            output = output.flatten(1, 2) # (t, bq, c)
            output = self.transformer_time_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                #query_pos=time_embds
                query_pos=None
            )

            output = output.permute(1, 2, 0)  # (bq, c, t)
            output = self.conv_norms[i](
                (self.conv_short_aggregate_layers[i](output) + output).transpose(1, 2)).transpose(1, 2)
            output = output.reshape(n_batch, n_instance, n_channel,
                                    n_frames).permute(1, 0, 3, 2).flatten(1, 2)  # (q, bt, c)

            # output = output.reshape(n_frames, n_batch, n_instance, n_channel)
            # output = output.permute(2, 1, 0, 3).flatten(1, 2) # (q, bt, c)

            output = self.transformer_obj_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=None
            )

            output = self.transformer_cross_attention_layers[i](
                output, frame_embeds,
                memory_mask=None,
                memory_key_padding_mask=None,
                pos=None, query_pos=None
            )
            output = self.transformer_ffn_layers[i](
                output
            )

            output = output.reshape(n_instance, n_batch, n_frames, n_channel).permute(1, 3, 2, 0) # (b, c, t, q)
            outputs.append(output)

        outputs = torch.stack(outputs, dim=0).permute(3, 0, 4, 1, 2) # (l, b, c, t, q) -> (frame, decoder_layer, q, b, c)
        outputs_class, outputs_masks = self.prediction(outputs, mask_features)
        outputs = self.decoder_norm(outputs)
        out = {
           'pred_logits': outputs_class[-1].transpose(1, 2),
           'pred_masks': outputs_masks[-1],
           'aux_outputs': self._set_aux_loss(
               outputs_class, outputs_masks
           ),
           'pred_embds': outputs[:, -1].permute(2, 3, 0, 1)  # b c t q
        }
        # pred_logits (bs, t, nq, c)
        # pred_masks (bs, nq, t, h, w)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{"pred_logits": a.transpose(1, 2), "pred_masks": b}
                for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
                ]

    def windows_prediction(self, outputs, mask_features, windows=5):
        iters = outputs.size(0) // windows
        if outputs.size(0) % windows != 0:
            iters += 1
        outputs_classes = []
        outputs_masks = []
        for i in range(iters):
            start_idx = i * windows
            end_idx = (i + 1) * windows
            clip_outputs = outputs[start_idx:end_idx]
            decoder_output = self.decoder_norm(clip_outputs)
            decoder_output = decoder_output.permute(1, 3, 0, 2, 4)  # (L, B, T, q, C)
            mask_embed = self.mask_embed(decoder_output)
            outputs_mask = torch.einsum("lbtqc,btchw->lbqthw", mask_embed,
                                        mask_features[:, start_idx:end_idx].to(mask_embed.device))
            outputs_classes.append(decoder_output)
            outputs_masks.append(outputs_mask.cpu().to(torch.float32))
        outputs_classes = torch.cat(outputs_classes, dim=2)
        outputs_classes = self.pred_class(outputs_classes)
        return outputs_classes.cpu().to(torch.float32), torch.cat(outputs_masks, dim=3)

    def pred_class(self, decoder_output):
        # decoder_output  (L, B, T, q, c)
        T = decoder_output.size(2)
        activation = self.activation_proj(decoder_output).softmax(dim=2) # (L, B, T, q, 1)

        class_output = (decoder_output * activation).sum(dim=2, keepdim=True) # (L, B, 1, q, c)
        # class_output = torch.cat([class_output, class_output.detach().repeat(1, 1, T - 1, 1, 1)], dim=2)
        debug = False
        if debug:
            temp = class_output[-1, 0, 0]
            temp = self.class_embed(temp).softmax(-1)
            fg = torch.max(temp, dim=-1)
            print(fg[0])
            print(fg[1])
            fg = torch.logical_and(fg[0] > 0.3, fg[1] != temp.size(1) - 1)
            print(activation[-1, 0].transpose(0, 1)[fg] * 100)

        class_output = class_output.repeat(1, 1, T, 1, 1)
        outputs_class = self.class_embed(class_output).transpose(2, 3)
        return outputs_class

    def prediction(self, outputs, mask_features, test_GFLOPS=False):
        # outputs (T, L, q, b, c)
        # mask_features (b, T, C, H, W)
        if self.training or test_GFLOPS:
            decoder_output = self.decoder_norm(outputs)
            decoder_output = decoder_output.permute(1, 3, 0, 2, 4)  # (L, B, T, q, C)
            outputs_class = self.pred_class(decoder_output)
            # output_class (L, B, q, T, Cls+1), activation (L, B, T, q, 1)
            mask_embed = self.mask_embed(decoder_output)
            outputs_mask = torch.einsum("lbtqc,btchw->lbqthw", mask_embed, mask_features)
        else:
            outputs = outputs[:, -1:]
            outputs_class, outputs_mask = self.windows_prediction(outputs, mask_features, windows=5)
        return outputs_class, outputs_mask


class QueryTracker_offline_transCls(torch.nn.Module):
    def __init__(self,
                 hidden_channel=256,
                 feedforward_channel=2048,
                 num_head=8,
                 decoder_layer_num=6,
                 mask_dim=256,
                 class_num=25,):
        super(QueryTracker_offline_transCls, self).__init__()

        # init transformer layers
        self.num_heads = num_head
        self.num_layers = decoder_layer_num
        self.transformer_obj_self_attention_layers = nn.ModuleList()
        self.transformer_time_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        self.conv_short_aggregate_layers = nn.ModuleList()
        self.conv_norms = nn.ModuleList()

        for _ in range(self.num_layers):
            self.transformer_time_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_channel,
                    nhead=num_head,
                    dropout=0.0,
                    normalize_before=False,
                )
            )

            self.conv_short_aggregate_layers.append(
                nn.Sequential(
                    nn.Conv1d(hidden_channel, hidden_channel, kernel_size=5, stride=1,
                              padding='same', padding_mode='replicate'),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(hidden_channel, hidden_channel, kernel_size=3, stride=1,
                              padding='same', padding_mode='replicate'),
                )
            )

            self.conv_norms.append(nn.LayerNorm(hidden_channel))

            self.transformer_obj_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_channel,
                    nhead=num_head,
                    dropout=0.0,
                    normalize_before=False,
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_channel,
                    nhead=num_head,
                    dropout=0.0,
                    normalize_before=False,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_channel,
                    dim_feedforward=feedforward_channel,
                    dropout=0.0,
                    normalize_before=False,
                )
            )

        self.decoder_norm = nn.LayerNorm(hidden_channel)

        # init heads
        self.class_embed = nn.Linear(hidden_channel, class_num + 1)
        self.mask_embed = MLP(hidden_channel, hidden_channel, mask_dim, 3)

        # class token generation
        self.transformer_class_mix_self_attention_layers = nn.ModuleList()
        self.transformer_class_mix_ffn_layers = nn.ModuleList()
        for _ in range(3):
            self.transformer_class_mix_self_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_channel,
                    nhead=num_head,
                    dropout=0.0,
                    normalize_before=False,
                )
            )
            self.transformer_class_mix_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_channel,
                    dim_feedforward=feedforward_channel,
                    dropout=0.0,
                    normalize_before=False,
                )
            )

        self.cls_token = nn.Embedding(1, hidden_channel)


    def forward(self, instance_embeds, frame_embeds, mask_features):
        # instance_embds (b, c, t, q)
        # frame_embds (b, c, t, q)
        n_batch, n_channel, n_frames, n_instance = instance_embeds.size()
        outputs = []

        output = instance_embeds
        #instance_embeds = instance_embeds.permute(3, 0, 2, 1).flatten(1, 2)
        frame_embeds = frame_embeds.permute(3, 0, 2, 1).flatten(1, 2)

        for i in range(self.num_layers):
            output = output.permute(2, 0, 3, 1) #(t, b, q, c)
            output = output.flatten(1, 2) # (t, bq, c)
            output = self.transformer_time_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                #query_pos=time_embds
                query_pos=None
            )

            output = output.permute(1, 2, 0)  # (bq, c, t)
            output = self.conv_norms[i](
                (self.conv_short_aggregate_layers[i](output) + output).transpose(1, 2)).transpose(1, 2)
            output = output.reshape(n_batch, n_instance, n_channel,
                                    n_frames).permute(1, 0, 3, 2).flatten(1, 2)  # (q, bt, c)

            # output = output.reshape(n_frames, n_batch, n_instance, n_channel)
            # output = output.permute(2, 1, 0, 3).flatten(1, 2) # (q, bt, c)

            output = self.transformer_obj_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=None
            )

            output = self.transformer_cross_attention_layers[i](
                output, frame_embeds,
                memory_mask=None,
                memory_key_padding_mask=None,
                pos=None, query_pos=None
            )
            output = self.transformer_ffn_layers[i](
                output
            )

            output = output.reshape(n_instance, n_batch, n_frames, n_channel).permute(1, 3, 2, 0) # (b, c, t, q)
            outputs.append(output)

        outputs = torch.stack(outputs, dim=0).permute(3, 0, 4, 1, 2) # (l, b, c, t, q) -> (frame, decoder_layer, q, b, c)
        outputs_class, outputs_masks = self.prediction(outputs, mask_features)
        outputs = self.decoder_norm(outputs)
        out = {
           'pred_logits': outputs_class[-1].transpose(1, 2),
           'pred_masks': outputs_masks[-1],
           'aux_outputs': self._set_aux_loss(
               outputs_class, outputs_masks
           ),
           'pred_embds': outputs[:, -1].permute(2, 3, 0, 1)  # b c t q
        }
        # pred_logits (bs, t, nq, c)
        # pred_masks (bs, nq, t, h, w)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{"pred_logits": a.transpose(1, 2), "pred_masks": b}
                for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
                ]

    def windows_prediction(self, outputs, mask_features, windows=5):
        iters = outputs.size(0) // windows
        if outputs.size(0) % windows != 0:
            iters += 1
        outputs_classes = []
        outputs_masks = []
        for i in range(iters):
            start_idx = i * windows
            end_idx = (i + 1) * windows
            clip_outputs = outputs[start_idx:end_idx]
            decoder_output = self.decoder_norm(clip_outputs)
            decoder_output = decoder_output.permute(1, 3, 0, 2, 4)  # (L, B, T, q, C)
            mask_embed = self.mask_embed(decoder_output)
            outputs_mask = torch.einsum("lbtqc,btchw->lbqthw", mask_embed,
                                        mask_features[:, start_idx:end_idx].to(mask_embed.device))
            outputs_classes.append(decoder_output)
            outputs_masks.append(outputs_mask.cpu().to(torch.float32))
        outputs_classes = torch.cat(outputs_classes, dim=2)
        outputs_classes = self.pred_class(outputs_classes)
        return outputs_classes.cpu().to(torch.float32), torch.cat(outputs_masks, dim=3)

    def pred_class(self, decoder_output):
        # decoder_output  (L, B, T, q, c)
        L, B, T, Q, C = decoder_output.size()
        class_output = decoder_output.permute(2, 0, 1, 3, 4).flatten(1, 3) # (T, LBQ, C)
        class_token = self.cls_token.weight.unsqueeze(1).repeat(1, L * B * Q, 1)


        for i in range(3):
            class_token = self.transformer_class_mix_self_attention_layers[i](
                class_token, class_output,
                memory_mask=None,
                memory_key_padding_mask=None,
                pos=None, query_pos=None
            )

            class_token = self.transformer_class_mix_ffn_layers[i](
                class_token
            )

        class_token = class_token.reshape(1, L, B, Q, C).permute(1, 2, 0, 3, 4) # (L, B, 1, Q, C)
        outputs_class = self.class_embed(class_token).repeat(1, 1, T, 1, 1).transpose(2, 3)
        return outputs_class

    def prediction(self, outputs, mask_features):
        # outputs (T, L, q, b, c)
        # mask_features (b, T, C, H, W)
        if self.training:
            decoder_output = self.decoder_norm(outputs)
            decoder_output = decoder_output.permute(1, 3, 0, 2, 4)  # (L, B, T, q, C)
            outputs_class = self.pred_class(decoder_output)
            # output_class (L, B, q, T, Cls+1), activation (L, B, T, q, 1)
            mask_embed = self.mask_embed(decoder_output)
            outputs_mask = torch.einsum("lbtqc,btchw->lbqthw", mask_embed, mask_features)
        else:
            outputs_class, outputs_mask = self.windows_prediction(outputs, mask_features, windows=5)
        return outputs_class, outputs_mask