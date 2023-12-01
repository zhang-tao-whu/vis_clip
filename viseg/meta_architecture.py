from typing import Tuple
import einops
import copy

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.structures import Boxes, ImageList, Instances, BitMasks

from mask2former_video.modeling.criterion import VideoSetCriterion
from mask2former_video.modeling.matcher import VideoHungarianMatcher, VideoHungarianMatcher_Consistent
from mask2former_video.utils.memory import retry_if_cuda_oom
from scipy.optimize import linear_sum_assignment
import random
from .mask2former_transformer_decoder import SelfAttentionLayer, CrossAttentionLayer, FFNLayer

class MinVIS(nn.Module):
    """
    Copied from "https://github.com/NVlabs/MinVIS".
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

        # building criterion
        matcher = VideoHungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )

        weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}

        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "masks"]

        criterion = VideoSetCriterion(
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
        )

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
        }

    @property
    def device(self):
        return self.pixel_mean.device

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
        images = []
        for video in batched_inputs:
            for frame in video["image"]:
                images.append(frame.to(self.device))
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        if not self.training and self.window_inference:
            outputs = self.run_window_inference(images.tensor, window_size=3)
        else:
            features = self.backbone(images.tensor)
            outputs = self.sem_seg_head(features)

        if self.training:
            # mask classification target
            targets = self.prepare_targets(batched_inputs, images)
            outputs, targets = self.frame_decoder_loss_reshape(outputs, targets)

            # bipartite matching-based loss
            losses = self.criterion(outputs, targets)

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
            return losses
        else:
            outputs = self.post_processing(outputs)

            mask_cls_results = outputs["pred_logits"]
            mask_pred_results = outputs["pred_masks"]

            mask_cls_result = mask_cls_results[0]
            mask_pred_result = mask_pred_results[0]
            first_resize_size = (images.tensor.shape[-2], images.tensor.shape[-1])

            input_per_image = batched_inputs[0]
            image_size = images.image_sizes[0]  # image size without padding after data augmentation

            height = input_per_image.get("height", image_size[0])  # raw image size before data augmentation
            width = input_per_image.get("width", image_size[1])

            return retry_if_cuda_oom(self.inference_video)(
                mask_cls_result,
                mask_pred_result,
                image_size,
                height,
                width,
                first_resize_size)

    def frame_decoder_loss_reshape(self, outputs, targets):
        outputs['pred_masks'] = einops.rearrange(outputs['pred_masks'], 'b q t h w -> (b t) q () h w')
        outputs['pred_logits'] = einops.rearrange(outputs['pred_logits'], 'b t q c -> (b t) q c')
        if 'aux_outputs' in outputs:
            for i in range(len(outputs['aux_outputs'])):
                outputs['aux_outputs'][i]['pred_masks'] = einops.rearrange(
                    outputs['aux_outputs'][i]['pred_masks'], 'b q t h w -> (b t) q () h w'
                )
                outputs['aux_outputs'][i]['pred_logits'] = einops.rearrange(
                    outputs['aux_outputs'][i]['pred_logits'], 'b t q c -> (b t) q c'
                )
        gt_instances = []
        for targets_per_video in targets:
            num_labeled_frames = targets_per_video['ids'].shape[1]
            for f in range(num_labeled_frames):
                labels = targets_per_video['labels']
                ids = targets_per_video['ids'][:, [f]]
                masks = targets_per_video['masks'][:, [f], :, :]
                gt_instances.append({"labels": labels, "ids": ids, "masks": masks})
        return outputs, gt_instances

    def match_from_embds(self, tgt_embds, cur_embds):
        cur_embds = cur_embds / cur_embds.norm(dim=1)[:, None]
        tgt_embds = tgt_embds / tgt_embds.norm(dim=1)[:, None]
        cos_sim = torch.mm(cur_embds, tgt_embds.transpose(0, 1))
        cost_embd = 1 - cos_sim
        C = 1.0 * cost_embd
        C = C.cpu()
        indices = linear_sum_assignment(C.transpose(0, 1))  # target x current
        indices = indices[1]  # permutation that makes current aligns to target
        return indices

    def post_processing(self, outputs):
        pred_logits, pred_masks, pred_embds = outputs['pred_logits'], outputs['pred_masks'], outputs['pred_embds']

        pred_logits = pred_logits[0]
        pred_masks = einops.rearrange(pred_masks[0], 'q t h w -> t q h w')
        pred_embds = einops.rearrange(pred_embds[0], 'c t q -> t q c')

        pred_logits = list(torch.unbind(pred_logits))
        pred_masks = list(torch.unbind(pred_masks))
        pred_embds = list(torch.unbind(pred_embds))

        out_logits = []
        out_masks = []
        out_embds = []
        out_logits.append(pred_logits[0])
        out_masks.append(pred_masks[0])
        out_embds.append(pred_embds[0])

        # match the instances frame by frame
        for i in range(1, len(pred_logits)):
            indices = self.match_from_embds(out_embds[-1], pred_embds[i])

            out_logits.append(pred_logits[i][indices, :])
            out_masks.append(pred_masks[i][indices, :, :])
            out_embds.append(pred_embds[i][indices, :])

        out_logits = sum(out_logits)/len(out_logits)
        out_masks = torch.stack(out_masks, dim=1)  # q h w -> q t h w

        out_logits = out_logits.unsqueeze(0)
        out_masks = out_masks.unsqueeze(0)

        outputs['pred_logits'] = out_logits
        outputs['pred_masks'] = out_masks

        return outputs

    def run_window_inference(self, images_tensor, window_size=30):
        iters = len(images_tensor) // window_size
        if len(images_tensor) % window_size != 0:
            iters += 1
        out_list = []
        for i in range(iters):
            start_idx = i * window_size
            end_idx = (i+1) * window_size

            features = self.backbone(images_tensor[start_idx:end_idx])
            out = self.sem_seg_head(features)
            del features['res2'], features['res3'], features['res4'], features['res5']
            for j in range(len(out['aux_outputs'])):
                del out['aux_outputs'][j]['pred_masks'], out['aux_outputs'][j]['pred_logits']
            out['pred_masks'] = out['pred_masks'].detach().cpu().to(torch.float32)
            out_list.append(out)

        # merge outputs
        outputs = {}
        outputs['pred_logits'] = torch.cat([x['pred_logits'] for x in out_list], dim=1).detach()
        outputs['pred_masks'] = torch.cat([x['pred_masks'] for x in out_list], dim=2).detach()
        outputs['pred_embds'] = torch.cat([x['pred_embds'] for x in out_list], dim=2).detach()

        return outputs

    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        gt_instances = []
        for targets_per_video in targets:
            _num_instance = len(targets_per_video["instances"][0])
            mask_shape = [_num_instance, self.num_frames, h_pad, w_pad]
            gt_masks_per_video = torch.zeros(mask_shape, dtype=torch.bool, device=self.device)

            gt_ids_per_video = []
            gt_classes_per_video = []
            for f_i, targets_per_frame in enumerate(targets_per_video["instances"]):
                targets_per_frame = targets_per_frame.to(self.device)
                h, w = targets_per_frame.image_size

                gt_ids_per_video.append(targets_per_frame.gt_ids[:, None])
                gt_classes_per_video.append(targets_per_frame.gt_classes[:, None])
                if isinstance(targets_per_frame.gt_masks, BitMasks):
                    gt_masks_per_video[:, f_i, :h, :w] = targets_per_frame.gt_masks.tensor
                else:  # polygon
                    gt_masks_per_video[:, f_i, :h, :w] = targets_per_frame.gt_masks

            gt_ids_per_video = torch.cat(gt_ids_per_video, dim=1)
            gt_classes_per_video = torch.cat(gt_classes_per_video, dim=1).max(dim=1)[0]
            valid_idx = (gt_ids_per_video != -1).any(dim=-1)

            gt_classes_per_video = gt_classes_per_video[valid_idx]  # N,
            gt_ids_per_video = gt_ids_per_video[valid_idx]  # N, num_frames

            gt_instances.append({"labels": gt_classes_per_video, "ids": gt_ids_per_video})
            gt_masks_per_video = gt_masks_per_video[valid_idx].float()  # N, num_frames, H, W
            gt_instances[-1].update({"masks": gt_masks_per_video})

        return gt_instances

    def inference_video(self, pred_cls, pred_masks, img_size, output_height, output_width, first_resize_size):
        if len(pred_cls) > 0:
            scores = F.softmax(pred_cls, dim=-1)[:, :-1]
            labels = torch.arange(
                self.sem_seg_head.num_classes,
                device=self.device
            ).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
            # keep top-10 predictions
            scores_per_image, topk_indices = scores.flatten(0, 1).topk(10, sorted=False)
            labels_per_image = labels[topk_indices]
            topk_indices = topk_indices // self.sem_seg_head.num_classes
            pred_masks = pred_masks[topk_indices]

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
            out_masks = [m for m in masks.cpu()]
        else:
            out_scores = []
            out_labels = []
            out_masks = []

        video_output = {
            "image_size": (output_height, output_width),
            "pred_scores": out_scores,
            "pred_labels": out_labels,
            "pred_masks": out_masks,
        }

        return video_output

@META_ARCH_REGISTRY.register()
class VISeg(MinVIS):
    """
    Online version of DVIS, including a segmenter and a referring tracker.
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
        tracker,
        num_frames,
        window_inference,
        max_num,
        max_iter_num,
        window_size,
        task,
        # use_cl
        use_cl,
        # for track
        track_embed,
        track_pos_embed,
        track_static_cross_attention_layers,
        track_self_attention_layers,
        track_static_self_attention_layers,
        track_ffn_layers,
        track_static_ffn_layers,
        track_pos_cross_attention_layers,
        track_pos_self_attention_layers,
        track_pos_ffn_layers,
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
            # video
            tracker: a tracker module, e.g. ReferringTracker
            num_frames: number of frames sampled during training
            window_inference: if the GPU memory is insufficient to predict the entire video at
                once, inference needs to be performed clip by clip
            num_class: the categories number of the dataset
            max_num: the maximum number of instances retained for a video, only used in VIS
            max_iter_num: the iter nums
            window_size: the number of images processed by the segmenter at a time
            task: VIS, VSS or VPS
        """
        super().__init__(
            backbone=backbone,
            sem_seg_head=sem_seg_head,
            criterion=criterion,
            num_queries=num_queries,
            object_mask_threshold=object_mask_threshold,
            overlap_threshold=overlap_threshold,
            metadata=metadata,
            size_divisibility=size_divisibility,
            sem_seg_postprocess_before_inference=sem_seg_postprocess_before_inference,
            pixel_mean=pixel_mean,
            pixel_std=pixel_std,
            # video
            num_frames=num_frames,
            window_inference=window_inference,
        )
        # frozen the segmenter
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        for p in self.sem_seg_head.parameters():
            p.requires_grad_(False)

        self.max_num = max_num
        self.iter = 0
        self.max_iter_num = max_iter_num

        self.window_size = window_size
        self.task = task
        assert self.task in ['vis', 'vss', 'vps'], "Only support vis, vss and vps !"
        inference_dict = {
            'vis': self.inference_video_vis,
            'vss': self.inference_video_vss,
            'vps': self.inference_video_vps,
        }
        self.inference_video_task = inference_dict[self.task]
        self.use_cl = use_cl

        # tracker
        self.track_embed = track_embed
        self.track_pos_embed = track_pos_embed
        self.track_static_cross_attention_layers = track_static_cross_attention_layers
        self.track_self_attention_layers = track_self_attention_layers
        self.track_static_self_attention_layers = track_static_self_attention_layers
        self.track_ffn_layers = track_ffn_layers
        self.track_static_ffn_layers = track_static_ffn_layers
        self.track_pos_cross_attention_layers = track_pos_cross_attention_layers
        self.track_pos_self_attention_layers = track_pos_self_attention_layers
        self.track_pos_ffn_layers = track_pos_ffn_layers

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

        # building criterion
        matcher = VideoHungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )

        weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}

        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        if cfg.MODEL.TRACKER.USE_CL:
            weight_dict.update({'loss_reid': 2})

        losses = ["labels", "masks"]

        criterion = VideoSetCriterion(
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
        )

        if cfg.MODEL.MASK_FORMER.REID_BRANCH:
            hidden_channel = cfg.MODEL.MASK_FORMER.HIDDEN_DIM * 2
        else:
            hidden_channel = cfg.MODEL.MASK_FORMER.HIDDEN_DIM

        tracker = None

        max_iter_num = cfg.SOLVER.MAX_ITER


        # for track
        num_track_layers = 3
        track_static_cross_attention_layers = nn.ModuleList()
        track_self_attention_layers = nn.ModuleList()
        track_ffn_layers = nn.ModuleList()
        track_static_self_attention_layers = nn.ModuleList()
        track_static_ffn_layers = nn.ModuleList()

        track_pos_cross_attention_layers = nn.ModuleList()
        track_pos_self_attention_layers = nn.ModuleList()
        track_pos_ffn_layers = nn.ModuleList()

        hidden_dim = cfg.MODEL.MASK_FORMER.HIDDEN_DIM
        n_heads = cfg.MODEL.MASK_FORMER.NHEADS
        pre_norm = cfg.MODEL.MASK_FORMER.PRE_NORM
        for _ in range(num_track_layers):
            track_static_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=n_heads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )
            track_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=n_heads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )
            track_static_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=n_heads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            track_pos_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=n_heads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )
            track_pos_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=n_heads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            track_pos_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=hidden_channel,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            track_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=hidden_channel,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )
            track_static_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=hidden_channel,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )
        track_embed = nn.Embedding(2, hidden_dim)
        track_pos_embed = nn.Embedding(1, hidden_dim)

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
            "tracker": tracker,
            "num_frames": cfg.INPUT.SAMPLING_FRAME_NUM,
            "window_inference": cfg.MODEL.MASK_FORMER.TEST.WINDOW_INFERENCE,
            "max_num": cfg.MODEL.MASK_FORMER.TEST.MAX_NUM,
            "max_iter_num": max_iter_num,
            "window_size": cfg.MODEL.MASK_FORMER.TEST.WINDOW_SIZE,
            "task": cfg.MODEL.MASK_FORMER.TEST.TASK,
            "use_cl": cfg.MODEL.REFINER.USE_CL,

            # for track
            "track_embed": track_embed,
            "track_pos_embed": track_pos_embed,
            "track_static_cross_attention_layers": track_static_cross_attention_layers,
            "track_self_attention_layers": track_self_attention_layers,
            "track_static_self_attention_layers": track_static_self_attention_layers,
            "track_ffn_layers": track_ffn_layers,
            "track_static_ffn_layers": track_static_ffn_layers,
            "track_pos_cross_attention_layers": track_pos_cross_attention_layers,
            "track_pos_self_attention_layers": track_pos_self_attention_layers,
            "track_pos_ffn_layers": track_pos_ffn_layers,

        }

    def mix_videos(self, gt_instances, n_batches=None, mode='clip'):
        assert n_batches is not None
        if n_batches == 1:
            return gt_instances

        clip_length = len(gt_instances) // n_batches

        id_base = 0
        for i in range(n_batches):
            clip_ids = gt_instances[clip_length * (i + 1) - 1]["ids"]  # (q, 1)
            if clip_ids.shape[0] == 0:
                continue
            max_ids = torch.max(clip_ids[:, 0], dim=0)[0]
            if i != 0:
                for j in range(clip_length):
                    gt_instances[clip_length * (i + 1) - 1 - j]["ids"] =\
                        gt_instances[clip_length * (i + 1) - 1 - j]["ids"] + id_base
            id_base = id_base + max_ids + 1

        if mode == 'clip':
            for i in range(clip_length, len(gt_instances)):
                if i % clip_length == 0:
                    additional_annos = {'ids': gt_instances[i - 1]['ids'],
                                        'labels': torch.zeros_like(gt_instances[i - 1]['labels']) + self.sem_seg_head.num_classes}
                    additional_annos.update({"masks": torch.zeros_like(gt_instances[i - 1]['masks'])})

                gt_instances[i]['ids'] = torch.cat([additional_annos['ids'], gt_instances[i]['ids'], ], dim=0)
                gt_instances[i]['labels'] = torch.cat([additional_annos['labels'], gt_instances[i]['labels'], ], dim=0)
                gt_instances[i]['masks'] = torch.cat([additional_annos['masks'], gt_instances[i]['masks'], ], dim=0)
        elif mode == 'inter_clip':
            pass
        else:
            raise NotImplementedError
        return gt_instances

    def targets_reshape(self, targets, mix_videos=True, n_batches=None):
        gt_instances = []
        for targets_per_video in targets:
            num_labeled_frames = targets_per_video['ids'].shape[1]
            for f in range(num_labeled_frames):
                labels = targets_per_video['labels']
                ids = targets_per_video['ids'][:, [f]]
                # fileter the have not appeared objects
                ids_history = targets_per_video['ids'][:, :f+1]
                max_ids_history = torch.max(ids_history, dim=-1)[0]
                valid = max_ids_history != -1
                is_bg = ids[:, 0] == -1
                labels[is_bg] = self.sem_seg_head.num_classes

                # valid = ids[:, 0] != -1
                masks = targets_per_video['masks'][:, [f], :, :]
                # gt_instances.append({"labels": labels[valid], "ids": ids[valid], "masks": masks[valid]})
                gt_instances.append({"labels": labels[valid], "ids": max_ids_history[valid].unsqueeze(1), "masks": masks[valid]})

        if mix_videos:
            return self.mix_videos(gt_instances, n_batches=n_batches)
        return gt_instances

    def pre_match(self, image_outputs, targets):
        matched_indexes, new_track_ids = [], []
        keep_track_ids = []

        pred_logits = image_outputs['pred_logits']  # (t, q, cls)
        pred_masks = image_outputs['pred_masks'].unsqueeze(2)  # (t, q, h, w)
        n_t, n_q = pred_masks.shape[:2]

        image_matched_indices = []
        for i in range(n_t):
            frame_pred_logits = pred_logits[i: i + 1]
            frame_pred_masks = pred_masks[i: i + 1]

            frame_matched_indices = self.criterion.matcher({'pred_logits': frame_pred_logits,
                                                            'pred_masks': frame_pred_masks},
                                                           [targets[i]])
            image_matched_indices.append(frame_matched_indices[0])

        gt_idx2id = []
        gt_id2idx = []

        exhibit_gt_ids = []
        for i, (frame_target, frame_matched_indices) in enumerate(zip(targets, image_matched_indices)):
            frame_target_ids = targets[i]['ids'][:, 0]
            frame_gt_idx2id = {}
            frame_gt_id2idx = {}
            for idx, id in enumerate(frame_target_ids.cpu().numpy()):
                frame_gt_id2idx[id] = idx
                frame_gt_idx2id[idx] = id
                gt_idx2id.append(frame_gt_idx2id)
                gt_id2idx.append(frame_gt_id2idx)

            frame_new_track_ids = []
            matched_pred_idxs, matched_gt_idxs = frame_matched_indices
            matched_pred_idxs, matched_gt_idxs = matched_pred_idxs.cpu().numpy(), matched_gt_idxs.cpu().numpy()
            if i == 0:
                # all new appera
                frame_new_track_ids += list(matched_pred_idxs)
                matched_indexes.append(frame_matched_indices)
                exhibit_gt_ids += [frame_gt_idx2id[idx] for idx in matched_gt_idxs]
            else:
                ret_frame_macthed_indxes = [[], []]

                random_delete_ids = []
                for off_idx in range(len(exhibit_gt_ids)):
                    if random.random() < 0.2:
                        random_delete_ids.append(off_idx)
                frame_keep_track_ids = [idx for idx in range(len(exhibit_gt_ids)) if idx not in random_delete_ids]
                for off_idx in random_delete_ids[::-1]:
                    del exhibit_gt_ids[off_idx]
                keep_track_ids.append(frame_keep_track_ids)

                for off_idx, exhibit_gt_id in enumerate(exhibit_gt_ids):
                    # gt id must in current frame
                    if exhibit_gt_id in frame_gt_id2idx.keys():
                        ret_frame_macthed_indxes[0].append(n_q + off_idx)
                        ret_frame_macthed_indxes[1].append(frame_gt_id2idx[exhibit_gt_id])

                for matched_pred_idx, mactched_gt_idx in zip(matched_pred_idxs, matched_gt_idxs):
                    if frame_gt_idx2id[mactched_gt_idx] not in exhibit_gt_ids:
                        frame_new_track_ids.append(matched_pred_idx)
                        exhibit_gt_ids.append(frame_gt_idx2id[mactched_gt_idx])
                        ret_frame_macthed_indxes[0].append(matched_pred_idx)
                        ret_frame_macthed_indxes[1].append(mactched_gt_idx)

                # print('ret_frame_macthed_indxes', ret_frame_macthed_indxes[0], [frame_gt_idx2id[idx] for idx in ret_frame_macthed_indxes[1]])
                ret_frame_macthed_indxes = (torch.as_tensor(ret_frame_macthed_indxes[0], dtype=torch.int64),
                                            torch.as_tensor(ret_frame_macthed_indxes[1], dtype=torch.int64))
                matched_indexes.append(ret_frame_macthed_indxes)
            new_track_ids.append(frame_new_track_ids)

        return matched_indexes, new_track_ids, keep_track_ids

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
            dict:
                For specific task, the dict contains the following keys:
                * For VIS:
                    "image_size": (output_height, output_width).
                    "pred_scores": score for per instance.
                    "pred_labels": class for per instance.
                    "pred_masks": list[Tensor], bit-masks for per instance, Tensor shape is (t, h, w).
                    "pred_ids": list, query ids for per instance, list length is N.
                    "task": "vis",
                * For VSS:
                    "image_size": (output_height, output_width).
                    "pred_masks": A Tensor that represents the
                        per-pixel segmentation prediced by the head.
                        The prediction has shape (t, h, w) that represents
                        the category ID for each pixel.
                    "task": "vss".
                * For VPS:
                    "image_size": (output_height, output_width).
                    "pred_masks": Tensor, shape is (t, h, w),
                        that represents the unique ID for the object which each pixel belong to.
                    "segments_infos": list[dict], info dicts for per object.
                        Info dict including unique ID, category ID and isthing.
                    "pred_ids": list, query ids for per thing and stuff, list length is N.
                    "task": "vps".
        """
        # for running demo on very long videos
        if 'keep' in batched_inputs[0].keys():
            self.keep = batched_inputs[0]['keep']
        else:
            self.keep = False

        images = []
        n_batches = len(batched_inputs)
        for video in batched_inputs:
            for frame in video["image"]:
                images.append(frame.to(self.device))
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        if not self.training:
            first_resize_size = (images.tensor.shape[-2], images.tensor.shape[-1])

            input_per_image = batched_inputs[0]
            image_size = images.image_sizes[0]  # image size without padding after data augmentation

            height = input_per_image.get("height", image_size[0])  # raw image size before data augmentation
            width = input_per_image.get("width", image_size[1])
            outputs = self.frame_by_frame_inference(images.tensor, first_resize_size,
                                                    image_size, height, width)
        else:
            self.backbone.eval()
            self.sem_seg_head.eval()
            with torch.no_grad():
                # first get the image outputs
                features = self.backbone(images.tensor)
                mask_features, transformer_encoder_features, multi_scale_features = \
                    self.sem_seg_head.pixel_decoder.forward_features(features)
                image_outputs = self.sem_seg_head(features, mask_features=mask_features,
                                                  transformer_encoder_features=transformer_encoder_features,
                                                  multi_scale_features=multi_scale_features)
                del image_outputs['aux_outputs']
                torch.cuda.empty_cache()

                targets = self.prepare_targets(batched_inputs, images)
                targets = self.targets_reshape(targets, mix_videos=True, n_batches=n_batches)
                matched_indexes, new_track_ids, keep_track_ids = self.pre_match(image_outputs, targets)
            outputs = []
            for i, frame_new_track_idx in enumerate(new_track_ids[:-1]):
                if i == 0:
                    n_q = image_outputs['pred_queries'].shape[0]
                    track_queries = image_outputs['pred_queries'][:, 0:1][frame_new_track_idx][keep_track_ids[i]]
                else:
                    new_track_queries = outputs[-1]['pred_queries'][frame_new_track_idx]
                    track_queries = torch.cat([outputs[-1]['pred_queries'][n_q:][keep_track_ids][i],
                                               new_track_queries]).detach()

                track_infos = {
                    'track_queries': track_queries, 'track_queries_pos': self.track_pos_embed.weight,
                    'track_embed': self.track_embed.weight,
                    'attention_layers': {
                        "track_static_cross_attention_layers": self.track_static_cross_attention_layers,
                        "track_self_attention_layers": self.track_self_attention_layers,
                        "track_static_self_attention_layers": self.track_static_self_attention_layers,
                        "track_ffn_layers": self.track_ffn_layers,
                        "track_static_ffn_layers": self.track_static_ffn_layers,
                        "track_pos_cross_attention_layers": self.track_pos_cross_attention_layers,
                        "track_pos_self_attention_layers": self.track_pos_self_attention_layers,
                        "track_pos_ffn_layers": self.track_pos_ffn_layers,
                    },
                }
                outputs.append(self.sem_seg_head(None, track_infos=track_infos,
                                                 mask_features=mask_features[i+1:i+2],
                                                 transformer_encoder_features=transformer_encoder_features[i+1:i+2],
                                                 multi_scale_features=[x[i+1:i+2] for x in multi_scale_features]))

        if self.training:
            frames_losses = []
            for i, output in enumerate(outputs):
                for aux_output in output['aux_outputs']:
                    aux_output['pred_masks'] = aux_output['pred_masks'].unsqueeze(2)
                output['pred_masks'] = output['pred_masks'].unsqueeze(2)

                frames_losses.append(self.criterion(output, targets[i + 1: i + 2],
                                                    match_indices=matched_indexes[i + 1: i + 2]))
            losses = {}
            n_frames = len(frames_losses)
            for key in frames_losses[0].keys():
                losses[key] = sum([frame_loss[key] for frame_loss in frames_losses]) / n_frames

            self.iter += 1

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
            return losses
        else:

            return retry_if_cuda_oom(self.inference_video_task)(
                outputs, images.tensor.shape[0], height, width
            )

    def frame_by_frame_inference(self, images, first_resize_size,
                                 image_size, height, width):
        self.backbone.eval()
        self.sem_seg_head.eval()
        out_list = []
        with torch.no_grad():
            for i in range(len(images)):
                features = self.backbone(images[i: i+1])
                if i == 0:
                    image_outputs = self.sem_seg_head(features)
                    n_q = image_outputs['pred_queries'].shape[0]
                    track_infos = self.extract_track_infos(image_outputs, out_list,
                                                           n_q, first_resize_size,
                                                           image_size, height, width)
                else:
                    image_outputs = self.sem_seg_head(features, track_infos=track_infos)
                    track_infos = self.extract_track_infos(image_outputs, out_list,
                                                           n_q, first_resize_size,
                                                           image_size, height, width)
        return out_list

    def extract_track_infos(self, image_outputs, out_list, n_q, first_resize_size,
                            img_size, output_height, output_width):

        def _process_track_embeds(pred_logits, pred_masks, out_list,
                                  first_resize_size, img_size, output_height, output_width):
            if pred_logits.shape[0] == 0:
                return
            scores = F.softmax(pred_logits, dim=-1)[:, :-1]
            max_scores = scores.max(dim=-1)[0]

            pred_masks = F.interpolate(
                pred_masks.unsqueeze(0), size=first_resize_size, mode="bilinear", align_corners=False
            )[0]
            pred_masks = pred_masks[:, : img_size[0], : img_size[1]]
            pred_masks = F.interpolate(
                pred_masks.unsqueeze(0), size=(output_height, output_width), mode="bilinear", align_corners=False
            )[0]
            masks = pred_masks > 0.
            del pred_masks
            masks = masks.cpu()

            for i in range(max_scores.shape[0]):
                if max_scores[i] < 0.3:
                    out_list[i]['pred_logits'].append(None)
                    out_list[i]['pred_masks'].append(torch.zeros_like(masks[i]))
                else:
                    out_list[i]['pred_logits'].append(pred_logits[i].to(torch.float32).cpu())
                    out_list[i]['pred_masks'].append(masks[i])
                #out_list[i]['pred_logits'].append(pred_logits[i].to(torch.float32).cpu())

                # out_list[i]['pred_masks'].append(masks[i])
            return

        def _process_new_embeds(pred_logits, pred_masks, out_list,
                                first_resize_size, img_size, output_height, output_width):
            scores = F.softmax(pred_logits, dim=-1)[:, :-1]
            max_scores = scores.max(dim=-1)[0]

            pred_masks = F.interpolate(
                pred_masks.unsqueeze(0), size=first_resize_size, mode="bilinear", align_corners=False
            )[0]
            pred_masks = pred_masks[:, :img_size[0], : img_size[1]]
            pred_masks = F.interpolate(
                pred_masks.unsqueeze(0), size=(output_height, output_width), mode="bilinear", align_corners=False
            )[0]
            masks = pred_masks > 0.
            del pred_masks
            masks = masks.cpu()

            keep_indexes = []
            for i in range(max_scores.shape[0]):
                if max_scores[i] > 0.8:
                    keep_indexes.append(i)
                    out_list.append({'pred_logits': [pred_logits[i].to(torch.float32).cpu()],
                                     'pred_masks': [masks[i]]})
            return keep_indexes

        del image_outputs['aux_outputs']
        pred_logits = image_outputs['pred_logits'][0]  # (q, c)
        pred_masks = image_outputs['pred_masks'][0]  # (q, h, w)
        pred_queries = image_outputs['pred_queries']  # (q, b, c)

        new_indices = _process_new_embeds(pred_logits[:n_q], pred_masks[:n_q], out_list,
                                          first_resize_size, img_size, output_height, output_width)
        _process_track_embeds(pred_logits[n_q:], pred_masks[n_q:], out_list,
                              first_resize_size, img_size, output_height, output_width)

        track_queries_1 = pred_queries[n_q:]
        track_queries_2 = pred_queries[new_indices]

        track_queries = torch.cat([track_queries_1, track_queries_2], dim=0)

        track_infos = {
            'track_queries': track_queries, 'track_queries_pos': self.track_pos_embed.weight,
            'track_embed': self.track_embed.weight,
            'attention_layers': {
                "track_static_cross_attention_layers": self.track_static_cross_attention_layers,
                "track_self_attention_layers": self.track_self_attention_layers,
                "track_static_self_attention_layers": self.track_static_self_attention_layers,
                "track_ffn_layers": self.track_ffn_layers,
                "track_static_ffn_layers": self.track_static_ffn_layers,
                "track_pos_cross_attention_layers": self.track_pos_cross_attention_layers,
                "track_pos_self_attention_layers": self.track_pos_self_attention_layers,
                "track_pos_ffn_layers": self.track_pos_ffn_layers,
            },
        }
        return track_infos

    def inference_video_vis(
        self, outputs, n_frames, output_height, output_width
    ):
        out_scores = []
        out_labels = []
        out_masks = []
        out_ids = []

        for i, output in enumerate(outputs):
            out_ids.append(i)

            pred_masks = output['pred_masks']
            pred_masks = torch.stack(pred_masks, dim=0)

            pre_masks = torch.zeros((n_frames - pred_masks.shape[0], pred_masks.shape[1], pred_masks.shape[2]),
                                    dtype=pred_masks.dtype, device=pred_masks.device)
            pred_masks = torch.cat([pre_masks, pred_masks], dim=0)

            out_masks.append(pred_masks)

            _num = 0
            pred_logits = 0
            for frame_pred_logits in output['pred_logits']:
                if frame_pred_logits is None:
                    pass
                else:
                    _num += 1
                    pred_logits = pred_logits + frame_pred_logits
            pred_logits = pred_logits / _num

            score, label = torch.max(torch.softmax(pred_logits, dim=0)[:-1], dim=0)
            out_scores.append(score.item())
            out_labels.append(label.item())

        video_output = {
            "image_size": (output_height, output_width),
            "pred_scores": out_scores,
            "pred_labels": out_labels,
            "pred_masks": out_masks,
            "pred_ids": out_ids,
            "task": "vis",
        }

        return video_output


    def inference_video_vps(
        self, pred_cls, pred_masks, img_size, output_height, output_width,
        first_resize_size, pred_id, aux_pred_cls=None,
    ):
        pred_cls = F.softmax(pred_cls, dim=-1)
        if aux_pred_cls is not None:
            aux_pred_cls = F.softmax(aux_pred_cls, dim=-1)[:, :-1]
            pred_cls[:, :-1] = torch.maximum(pred_cls[:, :-1], aux_pred_cls.to(pred_cls))
        mask_pred = pred_masks
        scores, labels = pred_cls.max(-1)

        # filter out the background prediction
        keep = labels.ne(self.sem_seg_head.num_classes) & (scores > self.object_mask_threshold)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_ids = pred_id[keep]
        cur_masks = mask_pred[keep]

        # interpolation to original image size
        cur_masks = F.interpolate(
            cur_masks, size=first_resize_size, mode="bilinear", align_corners=False
        )
        cur_masks = cur_masks[:, :, :img_size[0], :img_size[1]].sigmoid()
        cur_masks = F.interpolate(
            cur_masks, size=(output_height, output_width), mode="bilinear", align_corners=False
        )
        cur_prob_masks = cur_scores.view(-1, 1, 1, 1).to(cur_masks.device) * cur_masks

        # initial panoptic_seg and segments infos
        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((cur_masks.size(1), h, w), dtype=torch.int32, device=cur_masks.device)
        segments_infos = []
        out_ids = []
        current_segment_id = 0

        if cur_masks.shape[0] == 0:
            # We didn't detect any mask
            return {
                "image_size": (output_height, output_width),
                "pred_masks": panoptic_seg.cpu(),
                "segments_infos": segments_infos,
                "pred_ids": out_ids,
                "task": "vps",
            }
        else:
            # take argmax
            cur_mask_ids = cur_prob_masks.argmax(0)  # (t, h, w)
            stuff_memory_list = {}
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                isthing = pred_class < len(self.metadata.thing_dataset_id_to_contiguous_id)
                # filter out the unstable segmentation results
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
                "pred_masks": panoptic_seg.cpu(),
                "segments_infos": segments_infos,
                "pred_ids": out_ids,
                "task": "vps",
            }

    def inference_video_vss(
        self, pred_cls, pred_masks, img_size, output_height, output_width,
        first_resize_size, pred_id, aux_pred_cls=None,
    ):
        mask_cls = F.softmax(pred_cls, dim=-1)[..., :-1]
        if aux_pred_cls is not None:
            aux_pred_cls = F.softmax(aux_pred_cls, dim=-1)[..., :-1]
            mask_cls = torch.maximum(mask_cls, aux_pred_cls.to(mask_cls))
        mask_pred = pred_masks
        # interpolation to original image size
        cur_masks = F.interpolate(
            mask_pred, size=first_resize_size, mode="bilinear", align_corners=False
        )
        cur_masks = cur_masks[:, :, :img_size[0], :img_size[1]].sigmoid()
        cur_masks = F.interpolate(
            cur_masks, size=(output_height, output_width), mode="bilinear", align_corners=False
        )

        semseg = torch.einsum("qc,qthw->cthw", mask_cls, cur_masks)
        sem_score, sem_mask = semseg.max(0)
        sem_mask = sem_mask
        return {
                "image_size": (output_height, output_width),
                "pred_masks": sem_mask.cpu(),
                "task": "vss",
            }