from mask2former_video.modeling.transformer_decoder.video_mask2former_transformer_decoder import SelfAttentionLayer,\
    CrossAttentionLayer, FFNLayer, MLP, _get_activation_fn
from .utils import Noiser
from typing import Tuple
import einops

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
from .meta_architecture import MinVIS
import fvcore.nn.weight_init as weight_init
import random

class ReferringCrossAttentionLayer(nn.Module):

    def __init__(
        self,
        d_model,
        nhead,
        dropout=0.0,
        activation="relu",
        normalize_before=False,
        standard=False
    ):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.standard = standard
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        indentify,
        tgt,
        key,
        memory,
        memory_mask=None,
        memory_key_padding_mask=None,
        pos=None,
        query_pos=None
    ):
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(key, pos),
            value=memory, attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask)[0]
        tgt = indentify + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(
        self,
        indentify,
        tgt,
        key,
        memory,
        memory_mask=None,
        memory_key_padding_mask=None,
        pos=None,
        query_pos=None
    ):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt2, query_pos),
            key=self.with_pos_embed(key, pos),
            value=memory, attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask)[0]
        tgt = indentify + self.dropout(tgt2)

        return tgt

    def forward(
        self,
        indentify,
        tgt,
        key,
        memory,
        memory_mask=None,
        memory_key_padding_mask=None,
        pos=None,
        query_pos=None
    ):
        # when set "indentify = tgt", ReferringCrossAttentionLayer is same as CrossAttentionLayer
        if self.standard:
            tgt = tgt * 0.0 + indentify

        if self.normalize_before:
            return self.forward_pre(indentify, tgt, key, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(indentify, tgt, key, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)

class ClReferringTracker_noiser(torch.nn.Module):
    def __init__(
        self,
        hidden_channel=256,
        feedforward_channel=2048,
        num_head=8,
        decoder_layer_num=6,
        mask_dim=256,
        class_num=25,
        noise_mode='hard',
    ):
        super(ClReferringTracker_noiser, self).__init__()

        # init transformer layers
        self.num_heads = num_head
        self.num_layers = decoder_layer_num
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        for _ in range(self.num_layers):

            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_channel,
                    nhead=num_head,
                    dropout=0.0,
                    normalize_before=False,
                )
            )

            self.transformer_cross_attention_layers.append(
                ReferringCrossAttentionLayer(
                    d_model=hidden_channel,
                    nhead=num_head,
                    dropout=0.0,
                    normalize_before=False,
                    standard=True
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

        self.use_memory = False
        if self.use_memory:
            self.memory_cross_attn = CrossAttentionLayer(
                d_model=hidden_channel,
                nhead=num_head,
                dropout=0.0,
                normalize_before=False,)
            self.references_memory = None

        self.decoder_norm = nn.LayerNorm(hidden_channel)

        # init heads
        self.class_embed = nn.Linear(2 * hidden_channel, class_num + 1)
        self.mask_embed = MLP(hidden_channel, hidden_channel, mask_dim, 3)

        # for cl learning
        self.ref_proj = MLP(hidden_channel, hidden_channel, hidden_channel, 3)

        for layer in self.ref_proj.layers:
            weight_init.c2_xavier_fill(layer)

        # mask features projection
        self.mask_feature_proj = nn.Conv2d(
            mask_dim,
            mask_dim,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        # record previous frame information
        self.last_outputs = None
        self.last_frame_embeds = None
        self.last_reference = None

        self.noiser = Noiser(noise_ratio=0.8, mode=noise_mode)

    def _clear_memory(self):
        del self.last_outputs
        self.last_outputs = None
        self.last_reference = None
        return

    def forward(self, frame_embeds, mask_features, resume=False,
                return_indices=False, frame_classes=None,
                frame_embeds_no_norm=None):
        """
        :param frame_embeds: the instance queries output by the segmenter
        :param mask_features: the mask features output by the segmenter
        :param resume: whether the first frame is the start of the video
        :param return_indices: whether return the match indices
        :return: output dict, including masks, classes, embeds.
        """
        # mask feature projection
        mask_features_shape = mask_features.shape
        mask_features = self.mask_feature_proj(mask_features.flatten(0, 1)).reshape(*mask_features_shape)  # (b, t, c, h, w)

        frame_embeds = frame_embeds.permute(2, 3, 0, 1)  # t, q, b, c
        if frame_embeds_no_norm is not None:
            frame_embeds_no_norm = frame_embeds_no_norm.permute(2, 3, 0, 1)  # t, q, b, c
        n_frame, n_q, bs, _ = frame_embeds.size()
        outputs = []
        ret_indices = []

        all_frames_references = []

        for i in range(n_frame):
            ms_output = []
            single_frame_embeds = frame_embeds[i]  # q b c
            if frame_embeds_no_norm is not None:
                single_frame_embeds_no_norm = frame_embeds_no_norm[i]
            else:
                single_frame_embeds_no_norm = single_frame_embeds
            if frame_classes is None:
                single_frame_classes = None
            else:
                single_frame_classes = frame_classes[i]

            frame_key = single_frame_embeds_no_norm

            # the first frame of a video
            if i == 0 and resume is False:
                self._clear_memory()
                for j in range(self.num_layers):
                    if j == 0:
                        indices, noised_init = self.noiser(
                            single_frame_embeds,
                            single_frame_embeds,
                            cur_embeds_no_norm=single_frame_embeds_no_norm,
                            activate=False,
                            cur_classes=single_frame_classes,
                        )

                        # for reference as init value
                        noised_init = frame_key

                        ms_output.append(single_frame_embeds_no_norm[indices])
                        self.last_frame_embeds = single_frame_embeds[indices]
                        ret_indices.append(indices)
                        output = self.transformer_cross_attention_layers[j](
                            noised_init, self.ref_proj(frame_key),
                            frame_key, single_frame_embeds_no_norm,
                            memory_mask=None,
                            memory_key_padding_mask=None,
                            pos=None, query_pos=None
                        )

                        output = self.transformer_self_attention_layers[j](
                            output, tgt_mask=None,
                            tgt_key_padding_mask=None,
                            query_pos=None
                        )
                        # FFN
                        output = self.transformer_ffn_layers[j](
                            output
                        )
                        ms_output.append(output)
                    else:
                        output = self.transformer_cross_attention_layers[j](
                            ms_output[-1], self.ref_proj(ms_output[-1]),
                            frame_key, single_frame_embeds_no_norm,
                            memory_mask=None,
                            memory_key_padding_mask=None,
                            pos=None, query_pos=None
                        )

                        output = self.transformer_self_attention_layers[j](
                            output, tgt_mask=None,
                            tgt_key_padding_mask=None,
                            query_pos=None
                        )
                        # FFN
                        output = self.transformer_ffn_layers[j](
                            output
                        )
                        ms_output.append(output)
                self.last_reference = self.ref_proj(frame_key)
            else:
                reference = self.ref_proj(self.last_outputs[-1])
                self.last_reference = reference

                for j in range(self.num_layers):
                    if j == 0:
                        indices, noised_init = self.noiser(
                            self.last_frame_embeds,
                            single_frame_embeds,
                            cur_embeds_no_norm=single_frame_embeds_no_norm,
                            activate=self.training,
                            cur_classes=single_frame_classes,
                        )

                        # for reference as init value
                        noised_init = self.last_outputs[-1]

                        ms_output.append(single_frame_embeds_no_norm[indices])
                        self.last_frame_embeds = single_frame_embeds[indices]
                        ret_indices.append(indices)
                        output = self.transformer_cross_attention_layers[j](
                            noised_init, reference, frame_key,
                            single_frame_embeds_no_norm,
                            memory_mask=None,
                            memory_key_padding_mask=None,
                            pos=None, query_pos=None
                        )

                        output = self.transformer_self_attention_layers[j](
                            output, tgt_mask=None,
                            tgt_key_padding_mask=None,
                            query_pos=None
                        )
                        # FFN
                        output = self.transformer_ffn_layers[j](
                            output
                        )
                        ms_output.append(output)
                    else:
                        output = self.transformer_cross_attention_layers[j](
                            ms_output[-1], reference, frame_key,
                            single_frame_embeds_no_norm,
                            memory_mask=None,
                            memory_key_padding_mask=None,
                            pos=None, query_pos=None
                        )

                        output = self.transformer_self_attention_layers[j](
                            output, tgt_mask=None,
                            tgt_key_padding_mask=None,
                            query_pos=None
                        )
                        # FFN
                        output = self.transformer_ffn_layers[j](
                            output
                        )
                        ms_output.append(output)

            all_frames_references.append(self.last_reference)

            ms_output = torch.stack(ms_output, dim=0)  # (1 + layers, q, b, c)
            self.last_outputs = ms_output
            outputs.append(ms_output[1:])
        outputs = torch.stack(outputs, dim=0)  # (t, l, q, b, c)

        all_frames_references = torch.stack(all_frames_references, dim=0)  # (t, q, b, c)

        mask_features_ = mask_features
        if not self.training:
            outputs = outputs[:, -1:]
            del mask_features
        outputs_class, outputs_masks = self.prediction(outputs, mask_features_, all_frames_references)
        #outputs = self.decoder_norm(outputs)
        out = {
           'pred_logits': outputs_class[-1].transpose(1, 2),  # (b, t, q, c)
           'pred_masks': outputs_masks[-1],  # (b, q, t, h, w)
           'aux_outputs': self._set_aux_loss(
               outputs_class, outputs_masks
           ),
           'pred_embds': outputs[:, -1].permute(2, 3, 0, 1),  # (b, c, t, q),
           'pred_references': all_frames_references.permute(2, 3, 0, 1),  # (b, c, t, q),
        }
        if return_indices:
            return out, ret_indices
        else:
            return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{"pred_logits": a.transpose(1, 2), "pred_masks": b}
                for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
                ]

    def prediction(self, outputs, mask_features, references):
        # outputs (t, l, q, b, c)
        # mask_features (b, t, c, h, w)
        # references (t, q, b, c)
        decoder_output = self.decoder_norm(outputs)
        decoder_output = decoder_output.permute(1, 3, 0, 2, 4)  # (l, b, t, q, c)

        references = references.unsqueeze(1).repeat(1, decoder_output.size(0), 1, 1, 1).permute(1, 3, 0, 2, 4)  # (l, b, t, q, c)
        decoder_output_cls = torch.cat([references, decoder_output], dim=-1)
        outputs_class = self.class_embed(decoder_output_cls).transpose(2, 3)  # (l, b, q, t, cls+1)
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("lbtqc,btchw->lbqthw", mask_embed, mask_features)
        return outputs_class, outputs_mask


@META_ARCH_REGISTRY.register()
class ClDVIS_online(MinVIS):
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

        self.tracker = tracker
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

        self.classes_references_memory = Classes_References_Memory(max_len=20)

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
        matcher = VideoHungarianMatcher_Consistent(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            frames=cfg.INPUT.SAMPLING_FRAME_NUM
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

        tracker = ClReferringTracker_noiser(
            hidden_channel=cfg.MODEL.MASK_FORMER.HIDDEN_DIM * 2,
            feedforward_channel=cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD,
            num_head=cfg.MODEL.MASK_FORMER.NHEADS,
            decoder_layer_num=cfg.MODEL.TRACKER.DECODER_LAYERS,
            noise_mode=cfg.MODEL.TRACKER.NOISE_MODE,
            mask_dim=cfg.MODEL.MASK_FORMER.HIDDEN_DIM,
            class_num=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
        )

        max_iter_num = cfg.SOLVER.MAX_ITER

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
        }

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
        for video in batched_inputs:
            for frame in video["image"]:
                images.append(frame.to(self.device))
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        if not self.training and self.window_inference:
            outputs = self.run_window_inference(images.tensor, window_size=self.window_size)
        else:
            self.backbone.eval()
            self.sem_seg_head.eval()
            with torch.no_grad():
                features = self.backbone(images.tensor)
                image_outputs = self.sem_seg_head(features)
                object_labels = self._get_instance_labels(image_outputs['pred_logits'])
                frame_embds = image_outputs['pred_embds'].clone().detach()  # (b, c, t, q)
                frame_embds_no_norm = image_outputs['pred_embds_without_norm'].clone().detach()  # (b, c, t, q)
                mask_features = image_outputs['mask_features'].clone().detach().unsqueeze(0)
                del image_outputs['mask_features']
                torch.cuda.empty_cache()
            outputs, indices = self.tracker(frame_embds, mask_features, return_indices=True,
                                            resume=self.keep, frame_classes=object_labels,
                                            frame_embeds_no_norm=frame_embds_no_norm)
            image_outputs = self.reset_image_output_order(image_outputs, indices)


        if self.training:
            targets = self.prepare_targets(batched_inputs, images)
            # use the segmenter prediction results to guide the matching process during early training phase
            image_outputs, outputs, targets = self.frame_decoder_loss_reshape(
                outputs, targets, image_outputs=image_outputs
            )
            if self.iter < self.max_iter_num // 2:
                losses, reference_match_result = self.criterion(outputs, targets, matcher_outputs=image_outputs, ret_match_result=True)
            else:
                losses, reference_match_result = self.criterion(outputs, targets, matcher_outputs=None, ret_match_result=True)
            losses_cl = self.get_cl_loss_ref(outputs, reference_match_result)
            # losses_cl = self.get_cl_loss_ref_with_memory(outputs, reference_match_result, targets=targets)
            losses.update(losses_cl)

            self.iter += 1

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
            pred_ids = outputs["ids"]

            mask_cls_result = mask_cls_results[0]
            mask_pred_result = mask_pred_results[0]
            pred_id = pred_ids[0]
            first_resize_size = (images.tensor.shape[-2], images.tensor.shape[-1])

            input_per_image = batched_inputs[0]
            image_size = images.image_sizes[0]  # image size without padding after data augmentation

            height = input_per_image.get("height", image_size[0])  # raw image size before data augmentation
            width = input_per_image.get("width", image_size[1])

            return retry_if_cuda_oom(self.inference_video_task)(
                mask_cls_result, mask_pred_result, image_size, height, width, first_resize_size, pred_id
            )

    def _get_instance_labels(self, pred_logits):
        # b, t, q, c
        pred_logits = pred_logits[0]  # (t, q, c)
        scores = F.softmax(pred_logits, dim=-1)
        labels = torch.argmax(scores, dim=2)  # (t, q)
        labels[labels == pred_logits.size(2) - 1] = -1
        return labels

    def frame_decoder_loss_reshape(self, outputs, targets, image_outputs=None):
        outputs['pred_masks'] = einops.rearrange(outputs['pred_masks'], 'b q t h w -> (b t) q () h w')
        outputs['pred_logits'] = einops.rearrange(outputs['pred_logits'], 'b t q c -> (b t) q c')
        outputs['pred_references'] = einops.rearrange(outputs['pred_references'], 'b c t q -> (b t) q c')

        if image_outputs is not None:
            image_outputs['pred_masks'] = einops.rearrange(image_outputs['pred_masks'], 'b q t h w -> (b t) q () h w')
            image_outputs['pred_logits'] = einops.rearrange(image_outputs['pred_logits'], 'b t q c -> (b t) q c')
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
        return image_outputs, outputs, gt_instances

    def reset_image_output_order(self, output, indices):
        """
        in order to maintain consistency between the initial query and the guided results (segmenter prediction)
        :param output: segmenter prediction results (image-level segmentation results)
        :param indices: matched indicates
        :return: reordered outputs
        """
        # pred_keys, (b, c, t, q)
        indices = torch.Tensor(indices).to(torch.int64)  # (t, q)
        frame_indices = torch.range(0, indices.shape[0] - 1).to(indices).unsqueeze(1).repeat(1, indices.shape[1])
        # pred_masks, shape is (b, q, t, h, w)
        output['pred_masks'][0] = output['pred_masks'][0][indices, frame_indices].transpose(0, 1)
        # pred logits, shape is (b, t, q, c)
        output['pred_logits'][0] = output['pred_logits'][0][frame_indices, indices]
        return output

    def post_processing(self, outputs, aux_logits=None):
        """
        average the class logits and append query ids
        """
        pred_logits = outputs['pred_logits']
        pred_logits = pred_logits[0]  # (t, q, c)
        out_logits = torch.mean(pred_logits, dim=0).unsqueeze(0)
        if aux_logits is not None:
            aux_logits = aux_logits[0]
            aux_logits = torch.mean(aux_logits, dim=0)  # (q, c)
        outputs['pred_logits'] = out_logits
        outputs['ids'] = [torch.arange(0, outputs['pred_masks'].size(1))]
        if aux_logits is not None:
            return outputs, aux_logits
        return outputs

    def run_window_inference(self, images_tensor, window_size=30):
        iters = len(images_tensor) // window_size
        if len(images_tensor) % window_size != 0:
            iters += 1
        out_list = []
        for i in range(iters):
            start_idx = i * window_size
            end_idx = (i+1) * window_size
            # segmeter inference
            features = self.backbone(images_tensor[start_idx:end_idx])
            out = self.sem_seg_head(features)
            # remove unnecessary variables to save GPU memory
            del features['res2'], features['res3'], features['res4'], features['res5']
            for j in range(len(out['aux_outputs'])):
                del out['aux_outputs'][j]['pred_masks'], out['aux_outputs'][j]['pred_logits']
            # referring tracker inference
            frame_embds = out['pred_embds']  # (b, c, t, q)
            frame_embds_no_norm = out['pred_embds_without_norm']
            mask_features = out['mask_features'].unsqueeze(0)
            if i != 0 or self.keep:
                track_out = self.tracker(frame_embds, mask_features,
                                         resume=True, frame_embeds_no_norm=frame_embds_no_norm)
            else:
                track_out = self.tracker(frame_embds, mask_features, frame_embeds_no_norm=frame_embds_no_norm)
            # remove unnecessary variables to save GPU memory
            del mask_features
            for j in range(len(track_out['aux_outputs'])):
                del track_out['aux_outputs'][j]['pred_masks'], track_out['aux_outputs'][j]['pred_logits']
            track_out['pred_logits'] = track_out['pred_logits'].to(torch.float32).detach().cpu()
            track_out['pred_masks'] = track_out['pred_masks'].to(torch.float32).detach().cpu()
            track_out['pred_embds'] = track_out['pred_embds'].to(torch.float32).detach().cpu()
            # track_out['pred_logits'] = track_out['pred_logits'].detach()
            # track_out['pred_masks'] = track_out['pred_masks'].detach()
            # track_out['pred_embds'] = track_out['pred_embds'].detach()
            out_list.append(track_out)

        # merge outputs
        outputs = {}
        outputs['pred_logits'] = torch.cat([x['pred_logits'] for x in out_list], dim=1)
        outputs['pred_masks'] = torch.cat([x['pred_masks'] for x in out_list], dim=2)
        outputs['pred_embds'] = torch.cat([x['pred_embds'] for x in out_list], dim=2)

        return outputs

    def inference_video_vis(
        self, pred_cls, pred_masks, img_size, output_height, output_width,
        first_resize_size, pred_id, aux_pred_cls=None,
    ):
        if len(pred_cls) > 0:
            scores = F.softmax(pred_cls, dim=-1)[:, :-1]
            if aux_pred_cls is not None:
                aux_pred_cls = F.softmax(aux_pred_cls, dim=-1)[:, :-1]
                scores = torch.maximum(scores, aux_pred_cls.to(scores))
            labels = torch.arange(
                self.sem_seg_head.num_classes, device=self.device
            ).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
            # keep top-K predictions
            scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.max_num, sorted=False)
            labels_per_image = labels[topk_indices]
            topk_indices = topk_indices // self.sem_seg_head.num_classes
            pred_masks = pred_masks[topk_indices]
            pred_ids = pred_id[topk_indices]

            # interpolation to original image size
            pred_masks = F.interpolate(
                pred_masks, size=first_resize_size, mode="bilinear", align_corners=False
            )
            pred_masks = pred_masks[:, :, : img_size[0], : img_size[1]]
            pred_masks = F.interpolate(
                pred_masks, size=(output_height, output_width), mode="bilinear", align_corners=False
            )
            masks = pred_masks > 0.
            del pred_masks

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

    def get_cl_loss_ref(self, outputs, referecne_match_result):
        # outputs['pred_keys'] = (b t) q c
        # outputs['pred_references'] = (b t) q c
        references = outputs['pred_references']

        # per frame
        contrastive_items = []
        for i in range(references.size(0)):
            if i == 0:
                continue
            frame_reference = references[i]  # (q, c)
            frame_reference_ = references[i - 1]  # (q, c)

            if i != references.size(0) - 1:
                frame_reference_next = references[i + 1]
            else:
                frame_reference_next = None

            frame_ref_gt_indices = referecne_match_result[i]

            gt2ref = {}
            for i_ref, i_gt in zip(frame_ref_gt_indices[0], frame_ref_gt_indices[1]):
                gt2ref[i_gt.item()] = i_ref.item()
            # per instance
            for i_gt in gt2ref.keys():
                i_ref = gt2ref[i_gt]

                anchor_embeds = frame_reference[[i_ref]]
                pos_embeds = frame_reference_[[i_ref]]
                neg_range = list(range(0, i_ref)) + list(range(i_ref + 1, frame_reference.size(0)))
                neg_embeds = frame_reference_[neg_range]

                num_positive = pos_embeds.shape[0]
                # concate pos and neg to get whole constractive samples
                pos_neg_embedding = torch.cat(
                    [pos_embeds, neg_embeds], dim=0)
                # generate label, pos is 1, neg is 0
                pos_neg_label = pos_neg_embedding.new_zeros((pos_neg_embedding.shape[0],),
                                                            dtype=torch.int64)  # noqa
                pos_neg_label[:num_positive] = 1.

                # dot product
                dot_product = torch.einsum(
                    'ac,kc->ak', [pos_neg_embedding, anchor_embeds])
                aux_normalize_pos_neg_embedding = nn.functional.normalize(
                    pos_neg_embedding, dim=1)
                aux_normalize_anchor_embedding = nn.functional.normalize(
                    anchor_embeds, dim=1)

                aux_cosine_similarity = torch.einsum('ac,kc->ak', [aux_normalize_pos_neg_embedding,
                                                                   aux_normalize_anchor_embedding])
                contrastive_items.append({
                    'dot_product': dot_product,
                    'cosine_similarity': aux_cosine_similarity,
                    'label': pos_neg_label})

                if frame_reference_next is not None:
                    pos_embeds = frame_reference_next[[i_ref]]
                    neg_range = list(range(0, i_ref)) + list(range(i_ref + 1, frame_reference.size(0)))
                    # print(neg_range, '---------', i_key)
                    neg_embeds = frame_reference_next[neg_range]

                    num_positive = pos_embeds.shape[0]
                    # concate pos and neg to get whole constractive samples
                    pos_neg_embedding = torch.cat(
                        [pos_embeds, neg_embeds], dim=0)
                    # generate label, pos is 1, neg is 0
                    pos_neg_label = pos_neg_embedding.new_zeros((pos_neg_embedding.shape[0],),
                                                                dtype=torch.int64)  # noqa
                    pos_neg_label[:num_positive] = 1.

                    # dot product
                    dot_product = torch.einsum(
                        'ac,kc->ak', [pos_neg_embedding, anchor_embeds])
                    aux_normalize_pos_neg_embedding = nn.functional.normalize(
                        pos_neg_embedding, dim=1)
                    aux_normalize_anchor_embedding = nn.functional.normalize(
                        anchor_embeds, dim=1)

                    aux_cosine_similarity = torch.einsum('ac,kc->ak', [aux_normalize_pos_neg_embedding,
                                                                       aux_normalize_anchor_embedding])
                    contrastive_items.append({
                        'dot_product': dot_product,
                        'cosine_similarity': aux_cosine_similarity,
                        'label': pos_neg_label})

        losses = loss_reid(contrastive_items, outputs)
        return losses

    def get_cl_loss_ref_with_memory(self, outputs, referecne_match_result, targets):
        # outputs['pred_keys'] = (b t) q c
        # outputs['pred_references'] = (b t) q c
        references = outputs['pred_references']

        # per frame
        contrastive_items = []
        for i in range(references.size(0)):
            if i == 0:
                continue

            frame_reference = references[i]  # (q, c)
            frame_reference_ = references[i - 1]  # (q, c)

            if i != references.size(0) - 1:
                frame_reference_next = references[i + 1]
            else:
                frame_reference_next = None

            frame_ref_gt_indices = referecne_match_result[i]

            gt2ref = {}
            for i_ref, i_gt in zip(frame_ref_gt_indices[0], frame_ref_gt_indices[1]):
                gt2ref[i_gt.item()] = i_ref.item()
            # per instance
            for i_gt in gt2ref.keys():
                i_ref = gt2ref[i_gt]

                anchor_embeds = frame_reference[[i_ref]]
                pos_embeds = frame_reference_[[i_ref]]
                neg_range = list(range(0, i_ref)) + list(range(i_ref + 1, frame_reference.size(0)))
                neg_embeds = frame_reference_[neg_range]

                num_positive = pos_embeds.shape[0]
                # concate pos and neg to get whole constractive samples
                pos_neg_embedding = torch.cat(
                    [pos_embeds, neg_embeds], dim=0)
                # generate label, pos is 1, neg is 0
                pos_neg_label = pos_neg_embedding.new_zeros((pos_neg_embedding.shape[0],),
                                                            dtype=torch.int64)  # noqa
                pos_neg_label[:num_positive] = 1.

                # dot product
                dot_product = torch.einsum(
                    'ac,kc->ak', [pos_neg_embedding, anchor_embeds])
                aux_normalize_pos_neg_embedding = nn.functional.normalize(
                    pos_neg_embedding, dim=1)
                aux_normalize_anchor_embedding = nn.functional.normalize(
                    anchor_embeds, dim=1)

                aux_cosine_similarity = torch.einsum('ac,kc->ak', [aux_normalize_pos_neg_embedding,
                                                                   aux_normalize_anchor_embedding])
                contrastive_items.append({
                    'dot_product': dot_product,
                    'cosine_similarity': aux_cosine_similarity,
                    'label': pos_neg_label})

                if frame_reference_next is not None:
                    pos_embeds = frame_reference_next[[i_ref]]
                    neg_range = list(range(0, i_ref)) + list(range(i_ref + 1, frame_reference.size(0)))
                    # print(neg_range, '---------', i_key)
                    neg_embeds = frame_reference_next[neg_range]

                    num_positive = pos_embeds.shape[0]
                    # concate pos and neg to get whole constractive samples
                    pos_neg_embedding = torch.cat(
                        [pos_embeds, neg_embeds], dim=0)
                    # generate label, pos is 1, neg is 0
                    pos_neg_label = pos_neg_embedding.new_zeros((pos_neg_embedding.shape[0],),
                                                                dtype=torch.int64)  # noqa
                    pos_neg_label[:num_positive] = 1.

                    # dot product
                    dot_product = torch.einsum(
                        'ac,kc->ak', [pos_neg_embedding, anchor_embeds])
                    aux_normalize_pos_neg_embedding = nn.functional.normalize(
                        pos_neg_embedding, dim=1)
                    aux_normalize_anchor_embedding = nn.functional.normalize(
                        anchor_embeds, dim=1)

                    aux_cosine_similarity = torch.einsum('ac,kc->ak', [aux_normalize_pos_neg_embedding,
                                                                       aux_normalize_anchor_embedding])
                    contrastive_items.append({
                        'dot_product': dot_product,
                        'cosine_similarity': aux_cosine_similarity,
                        'label': pos_neg_label})

                # cls cl
                cls = targets[i]['labels'][i_gt].item()
                # anchor_embeds = frame_reference[[i_ref]]
                anchor_embeds = anchor_embeds * 0.1 + anchor_embeds.detach() * 0.9
                if frame_reference_next is None:
                    pos_embeds = frame_reference_[[i_ref]].detach()
                else:
                    pos_embeds = torch.cat([frame_reference_[[i_ref]], frame_reference_next[[i_ref]]], dim=0).detach()
                neg_embeds = self.classes_references_memory.get_items(cls)
                if len(neg_embeds) != 0:
                    num_positive = pos_embeds.shape[0]
                    # concate pos and neg to get whole constractive samples
                    pos_neg_embedding = torch.cat(
                        [pos_embeds, neg_embeds], dim=0)
                    # generate label, pos is 1, neg is 0
                    pos_neg_label = pos_neg_embedding.new_zeros((pos_neg_embedding.shape[0],),
                                                                dtype=torch.int64)  # noqa
                    pos_neg_label[:num_positive] = 1.

                    # dot product
                    dot_product = torch.einsum(
                        'ac,kc->ak', [pos_neg_embedding, anchor_embeds])
                    aux_normalize_pos_neg_embedding = nn.functional.normalize(
                        pos_neg_embedding, dim=1)
                    aux_normalize_anchor_embedding = nn.functional.normalize(
                        anchor_embeds, dim=1)

                    aux_cosine_similarity = torch.einsum('ac,kc->ak', [aux_normalize_pos_neg_embedding,
                                                                       aux_normalize_anchor_embedding])
                    contrastive_items.append({
                        'dot_product': dot_product,
                        'cosine_similarity': aux_cosine_similarity,
                        'label': pos_neg_label})

        self.classes_references_memory.push(references, targets, referecne_match_result)

        losses = loss_reid(contrastive_items, outputs)
        return losses

class Classes_References_Memory:
    def __init__(self, max_len=100,):
        self.class_references = {}
        self.max_len = max_len

    def push(self, references, targets, referecne_match_result):
        references = references.detach()
        for i in range(len(targets)):
            classes = targets[i]['labels']  # (N, )
            frame_match_result = referecne_match_result[i]
            frame_reference = references[i]
            for i_ref, i_gt in zip(frame_match_result[0], frame_match_result[1]):
                cls = classes[i_gt].item()
                if cls in self.class_references.keys():
                    self.class_references[cls].append(frame_reference[i_ref])
                else:
                    self.class_references[cls] = [frame_reference[i_ref]]
        for cls in self.class_references.keys():
            if len(self.class_references[cls]) > self.max_len:
                self.class_references[cls] = self.class_references[cls][-self.max_len:]
        return

    def push_refiner(self, references, targets, referecne_match_result):
        # (t q c)
        references = references.clone().detach()
        classes = targets['labels']  # (N, )
        for i_ref, i_gt in zip(referecne_match_result[0], referecne_match_result[1]):
            cls = classes[i_gt].item()
            if cls in self.class_references.keys():
                self.class_references[cls].extend(list(torch.unbind(references[:, i_ref], dim=0)))
            else:
                self.class_references[cls] = list(torch.unbind(references[:, i_ref], dim=0))

        for cls in self.class_references.keys():
            if len(self.class_references[cls]) > self.max_len:
                random.shuffle(self.class_references[cls])
                self.class_references[cls] = self.class_references[cls][-self.max_len:]
        return

    def get_items(self, cls):
        if cls not in self.class_references.keys():
            return []
        else:
            cls_ref = torch.stack(self.class_references[cls], dim=0)
            return cls_ref

def loss_reid(qd_items, outputs):
    # outputs only using when have not contrastive items
    # compute two loss, contrastive loss & similarity loss
    contras_loss = 0
    aux_loss = 0

    num_qd_items = len(qd_items) # n_instances * frames

    # if none items, return 0 loss
    if len(qd_items) == 0:
        if 'pred_references' in outputs.keys():
            losses = {'loss_reid': outputs['pred_references'].sum() * 0,
                      'loss_aux_reid': outputs['pred_references'].sum() * 0}
        else:
            losses = {'loss_reid': outputs['pred_embds'].sum() * 0,
                      'loss_aux_reid': outputs['pred_embds'].sum() * 0}
        return losses

    for qd_item in qd_items:
        # (n_pos, n_anchor) -> (n_anchor, n_pos)
        pred = qd_item['dot_product'].permute(1, 0)
        label = qd_item['label'].unsqueeze(0)
        # contrastive loss
        pos_inds = (label == 1)
        neg_inds = (label == 0)
        pred_pos = pred * pos_inds.float()
        pred_neg = pred * neg_inds.float()
        # use -inf to mask out unwanted elements.
        pred_pos[neg_inds] = pred_pos[neg_inds] + float('inf')
        pred_neg[pos_inds] = pred_neg[pos_inds] + float('-inf')

        _pos_expand = torch.repeat_interleave(pred_pos, pred.shape[1], dim=1)
        _neg_expand = pred_neg.repeat(1, pred.shape[1])
        # [bz,N], N is all pos and negative samples on reference frame, label indicate it's pos or negative
        x = torch.nn.functional.pad(
            (_neg_expand - _pos_expand), (0, 1), "constant", 0)
        contras_loss += torch.logsumexp(x, dim=1)

        aux_pred = qd_item['cosine_similarity'].permute(1, 0)
        aux_label = qd_item['label'].unsqueeze(0)

        aux_loss += (torch.abs(aux_pred - aux_label) ** 2).mean()


    losses = {'loss_reid': contras_loss.sum() / num_qd_items,
              'loss_aux_reid': aux_loss / num_qd_items}
    return losses

# dvis offline
class TemporalRefiner(torch.nn.Module):
    def __init__(
        self,
        hidden_channel=256,
        feedforward_channel=2048,
        num_head=8,
        decoder_layer_num=6,
        mask_dim=256,
        class_num=25,
        windows=5,
        mask_agu=False,
        mask_ratio=0.4,
    ):
        super(TemporalRefiner, self).__init__()

        self.windows = windows

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
                    nn.Conv1d(hidden_channel, hidden_channel,
                              kernel_size=5, stride=1,
                              padding='same', padding_mode='replicate'),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(hidden_channel, hidden_channel,
                              kernel_size=3, stride=1,
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

        self.activation_proj = nn.Linear(hidden_channel, 1)

        # mask agumentation
        self.mask_agu = mask_agu
        self.mask_ratio = mask_ratio

    def forward(self, instance_embeds, frame_embeds, mask_features):
        """
        :param instance_embeds: the aligned instance queries output by the tracker, shape is (b, c, t, q)
        :param frame_embeds: the instance queries processed by the tracker.frame_forward function, shape is (b, c, t, q)
        :param mask_features: the mask features output by the segmenter, shape is (b, t, c, h, w)
        :return: output dict, including masks, classes, embeds.
        """
        n_batch, n_channel, n_frames, n_instance = instance_embeds.size()

        if self.training and self.mask_agu:
            temporal_mask = torch.rand(n_frames, n_frames).to(instance_embeds)
            temporal_mask = torch.maximum(temporal_mask, torch.eye(n_frames).to(instance_embeds))
            temporal_mask = temporal_mask <= self.mask_ratio
        else:
            temporal_mask = None

        outputs = []
        output = instance_embeds
        frame_embeds = frame_embeds.permute(3, 0, 2, 1).flatten(1, 2)

        for i in range(self.num_layers):
            output = output.permute(2, 0, 3, 1)  # (t, b, q, c)
            output = output.flatten(1, 2)  # (t, bq, c)

            # do long temporal attention
            output = self.transformer_time_self_attention_layers[i](
                output, tgt_mask=temporal_mask,
                tgt_key_padding_mask=None,
                query_pos=None
            )

            # do short temporal conv
            output = output.permute(1, 2, 0)  # (bq, c, t)
            output = self.conv_norms[i](
                (self.conv_short_aggregate_layers[i](output) + output).transpose(1, 2)
            ).transpose(1, 2)
            output = output.reshape(
                n_batch, n_instance, n_channel, n_frames
            ).permute(1, 0, 3, 2).flatten(1, 2)  # (q, bt, c)

            # do objects self attention
            output = self.transformer_obj_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=None
            )

            # do cross attention
            output = self.transformer_cross_attention_layers[i](
                output, frame_embeds,
                memory_mask=None,
                memory_key_padding_mask=None,
                pos=None, query_pos=None
            )

            # FFN
            output = self.transformer_ffn_layers[i](
                output
            )

            output = output.reshape(n_instance, n_batch, n_frames, n_channel).permute(1, 3, 2, 0)  # (b, c, t, q)
            outputs.append(output)

        outputs = torch.stack(outputs, dim=0).permute(3, 0, 4, 1, 2)  # (l, b, c, t, q) -> (t, l, q, b, c)
        outputs_class, outputs_masks = self.prediction(outputs, mask_features)
        outputs = self.decoder_norm(outputs)
        out = {
           'pred_logits': outputs_class[-1].transpose(1, 2),  # (b, t, q, c)
           'pred_masks': outputs_masks[-1],  # (b, q, t, h, w)
           'aux_outputs': self._set_aux_loss(
               outputs_class, outputs_masks
           ),
           'pred_embds': outputs[:, -1].permute(2, 3, 0, 1)  # (b, c, t, q)
        }
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
        """
        for windows prediction, because mask features consumed too much GPU memory
        """
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
            decoder_output = decoder_output.permute(1, 3, 0, 2, 4)  # (l, b, t, q, c)
            mask_embed = self.mask_embed(decoder_output)
            outputs_mask = torch.einsum(
                "lbtqc,btchw->lbqthw",
                mask_embed,
                mask_features[:, start_idx:end_idx].to(mask_embed.device)
            )
            outputs_classes.append(decoder_output)
            outputs_masks.append(outputs_mask.cpu().to(torch.float32))
        outputs_classes = torch.cat(outputs_classes, dim=2)
        outputs_classes = self.pred_class(outputs_classes)
        return outputs_classes.cpu().to(torch.float32), torch.cat(outputs_masks, dim=3)

    def pred_class(self, decoder_output):
        """
        fuse the objects queries of all frames and predict an overall score based on the fused objects queries
        :param decoder_output: instance queries, shape is (l, b, t, q, c)
        """
        T = decoder_output.size(2)

        # compute the weighted average of the decoder_output
        activation = self.activation_proj(decoder_output).softmax(dim=2)  # (l, b, t, q, 1)
        class_output = (decoder_output * activation).sum(dim=2, keepdim=True)  # (l, b, 1, q, c)

        # to unify the output format, duplicate the fused features T times
        class_output = class_output.repeat(1, 1, T, 1, 1)
        outputs_class = self.class_embed(class_output).transpose(2, 3)
        return outputs_class

    def prediction(self, outputs, mask_features):
        """
        :param outputs: instance queries, shape is (t, l, q, b, c)
        :param mask_features: mask features, shape is (b, t, c, h, w)
        :return: pred class and pred masks
        """
        if self.training:
            decoder_output = self.decoder_norm(outputs)
            decoder_output = decoder_output.permute(1, 3, 0, 2, 4)  # (l, b, t, q, c)
            outputs_class = self.pred_class(decoder_output)
            mask_embed = self.mask_embed(decoder_output)
            outputs_mask = torch.einsum("lbtqc,btchw->lbqthw", mask_embed, mask_features)
        else:
            outputs = outputs[:, -1:]
            outputs_class, outputs_mask = self.windows_prediction(outputs, mask_features, windows=self.windows)
        return outputs_class, outputs_mask

@META_ARCH_REGISTRY.register()
class ClDVIS_offline(ClDVIS_online):
    """
    Offline version of DVIS, including a segmenter, a referring tracker and a temporal refiner.
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
        refiner,
        num_frames,
        window_inference,
        max_num,
        max_iter_num,
        window_size,
        task,
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
            refiner: a refiner module, e.g. TemporalRefiner
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
            tracker=tracker,
            num_frames=num_frames,
            window_inference=window_inference,
            max_num=max_num,
            max_iter_num=max_iter_num,
            window_size=window_size,
            task=task,
        )

        # frozen the referring tracker
        for p in self.tracker.parameters():
            p.requires_grad_(False)

        self.refiner = refiner

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
            # since when calculating the loss, the t frames of a video are flattened into a image with size of (th, w),
            # the number of sampling points is increased t times accordingly.
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS * cfg.INPUT.SAMPLING_FRAME_NUM,
        )

        weight_dict = {
            "loss_ce": class_weight,
            "loss_mask": mask_weight,
            "loss_dice": dice_weight,
        }

        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        #weight_dict.update({'loss_reid': 2})
        losses = ["labels", "masks"]

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

        tracker = ClReferringTracker_noiser(
            hidden_channel=cfg.MODEL.MASK_FORMER.HIDDEN_DIM * 2,
            feedforward_channel=cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD,
            num_head=cfg.MODEL.MASK_FORMER.NHEADS,
            decoder_layer_num=cfg.MODEL.TRACKER.DECODER_LAYERS,
            noise_mode=cfg.MODEL.TRACKER.NOISE_MODE,
            mask_dim=cfg.MODEL.MASK_FORMER.HIDDEN_DIM,
            class_num=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
        )

        refiner = TemporalRefiner(
            hidden_channel=cfg.MODEL.MASK_FORMER.HIDDEN_DIM * 2,
            feedforward_channel=cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD,
            num_head=cfg.MODEL.MASK_FORMER.NHEADS,
            decoder_layer_num=cfg.MODEL.REFINER.DECODER_LAYERS,
            mask_dim=cfg.MODEL.MASK_FORMER.HIDDEN_DIM,
            class_num=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            windows=cfg.MODEL.MASK_FORMER.TEST.WINDOW_SIZE,
            mask_agu=cfg.MODEL.REFINER.MASK_AGU,
            mask_ratio=cfg.MODEL.REFINER.MASK_RATIO,
        )

        max_iter_num = cfg.SOLVER.MAX_ITER

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
            "refiner": refiner,
            "num_frames": cfg.INPUT.SAMPLING_FRAME_NUM,
            "window_inference": cfg.MODEL.MASK_FORMER.TEST.WINDOW_INFERENCE,
            "max_num": cfg.MODEL.MASK_FORMER.TEST.MAX_NUM,
            "max_iter_num": max_iter_num,
            "window_size": cfg.MODEL.MASK_FORMER.TEST.WINDOW_SIZE,
            "task": cfg.MODEL.MASK_FORMER.TEST.TASK,
        }

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
            outputs, online_pred_logits = self.run_window_inference(images.tensor, window_size=self.window_size)
        else:
            with torch.no_grad():
                # due to GPU memory limitations, the segmenter processes the video clip by clip.
                image_outputs = self.segmentor_windows_inference(images.tensor, window_size=21)
                object_labels = self._get_instance_labels(image_outputs['pred_logits'])
                frame_embds = image_outputs['pred_embds'].clone().detach()  # (b, c, t, q)
                frame_embds_no_norm = image_outputs['pred_embds_without_norm'].clone().detach()  # (b, c, t, q)
                mask_features = image_outputs['mask_features'].clone().detach().unsqueeze(0)
                del image_outputs['mask_features'], image_outputs['pred_embds_without_norm'],\
                    image_outputs['pred_logits'], image_outputs['pred_embds']

                # perform tracker/alignment
                image_outputs = self.tracker(
                    frame_embds, mask_features,
                    resume=self.keep, frame_classes=object_labels,
                    frame_embeds_no_norm=frame_embds_no_norm
                )
                online_pred_logits = image_outputs['pred_logits']  # (b, t, q, c)
                # frame_embds_ = self.tracker.frame_forward(frame_embds)
                frame_embds_ = frame_embds_no_norm.clone().detach()
                instance_embeds = image_outputs['pred_embds'].clone().detach()

                del frame_embds, frame_embds_no_norm
                del image_outputs['pred_embds']
                for j in range(len(image_outputs['aux_outputs'])):
                    del image_outputs['aux_outputs'][j]['pred_masks'], image_outputs['aux_outputs'][j]['pred_logits']
                torch.cuda.empty_cache()
            # do temporal refine
            outputs = self.refiner(instance_embeds, frame_embds_, mask_features)

        if self.training:
            # mask classification target
            targets = self.prepare_targets(batched_inputs, images)
            # use the online prediction results to guide the matching process during early training phase
            if self.iter < self.max_iter_num // 2:
                image_outputs, outputs, targets = self.frame_decoder_loss_reshape(
                    outputs, targets, image_outputs=image_outputs
                )
            else:
                image_outputs, outputs, targets = self.frame_decoder_loss_reshape(
                    outputs, targets, image_outputs=None
                )
            self.iter += 1

            # bipartite matching-based loss
            losses, matching_result = self.criterion(outputs, targets,
                                                     matcher_outputs=image_outputs, ret_match_result=True)
            # cl_loss = self.get_cl_loss(outputs, matching_result)
            cl_loss = self.get_cl_loss_with_memory(outputs, matching_result, targets)
            losses.update(cl_loss)

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
            return losses
        else:
            outputs, aux_pred_logits = self.post_processing(outputs, aux_logits=online_pred_logits)
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

            return retry_if_cuda_oom(self.inference_video_task)(
                mask_cls_result, mask_pred_result, image_size, height, width,
                first_resize_size, pred_id, aux_pred_cls=aux_pred_logits,
            )

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
            del out['pred_masks']
            for j in range(len(out['aux_outputs'])):
                del out['aux_outputs'][j]['pred_masks'], out['aux_outputs'][j]['pred_logits']
            outs_list.append(out)

        image_outputs['pred_embds'] = torch.cat([x['pred_embds'] for x in outs_list], dim=2).detach()
        image_outputs['mask_features'] = torch.cat([x['mask_features'] for x in outs_list], dim=0).detach()
        image_outputs['pred_logits'] = torch.cat([x['pred_logits'] for x in outs_list], dim=1).detach()
        image_outputs['pred_embds_without_norm'] = torch.cat([x['pred_embds_without_norm'] for x in outs_list], dim=2).detach()
        return image_outputs

    def frame_decoder_loss_reshape(self, outputs, targets, image_outputs=None):
        # flatten the t frames as an image with size of (th, w)
        outputs['pred_masks'] = einops.rearrange(outputs['pred_masks'], 'b q t h w -> b q () (t h) w')
        outputs['pred_logits'] = outputs['pred_logits'][:, 0, :, :]
        if image_outputs is not None:
            image_outputs['pred_masks'] = einops.rearrange(image_outputs['pred_masks'], 'b q t h w -> b q () (t h) w')
            image_outputs['pred_logits'] = image_outputs['pred_logits'].mean(dim=1)
        if 'aux_outputs' in outputs:
            for i in range(len(outputs['aux_outputs'])):
                outputs['aux_outputs'][i]['pred_masks'] = einops.rearrange(
                    outputs['aux_outputs'][i]['pred_masks'], 'b q t h w -> b q () (t h) w'
                )
                outputs['aux_outputs'][i]['pred_logits'] = outputs['aux_outputs'][i]['pred_logits'][:, 0, :, :]

        gt_instances = []
        for targets_per_video in targets:
            targets_per_video['masks'] = einops.rearrange(
                targets_per_video['masks'], 'q t h w -> q () (t h) w'
                )
            gt_instances.append(targets_per_video)
        return image_outputs, outputs, gt_instances

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

            # sementer inference
            features = self.backbone(images_tensor[start_idx:end_idx])
            out = self.sem_seg_head(features)

            del features['res2'], features['res3'], features['res4'], features['res5']
            del out['pred_masks']
            for j in range(len(out['aux_outputs'])):
                del out['aux_outputs'][j]['pred_masks'], out['aux_outputs'][j]['pred_logits']

            object_labels = self._get_instance_labels(out['pred_logits'])
            frame_embds = out['pred_embds']  # (b, c, t, q)
            frame_embds_no_norm = out['pred_embds_without_norm']
            mask_features = out['mask_features'].unsqueeze(0)
            overall_mask_features.append(mask_features.cpu())
            overall_frame_embds.append(frame_embds_no_norm)

            # referring tracker inference
            if i != 0:
                track_out = self.tracker(frame_embds, mask_features, resume=True,
                                         frame_classes=object_labels,
                                         frame_embeds_no_norm=frame_embds_no_norm)
            else:
                track_out = self.tracker(frame_embds, mask_features, frame_classes=object_labels,
                                         frame_embeds_no_norm=frame_embds_no_norm)
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
        #overall_frame_embds_ = self.tracker.frame_forward(overall_frame_embds)
        #del overall_frame_embds

        # temporal refiner inference
        outputs = self.refiner(overall_instance_embds, overall_frame_embds, overall_mask_features)
        return outputs, online_pred_logits

    def get_cl_loss(self, outputs_, matching_result):
        # outputs['pred_keys'] = (b t) q c
        # outputs['pred_references'] = (b t) q c
        assert outputs_['pred_embds'].shape[0] == len(matching_result) == 1
        outputs = outputs_['pred_embds'][0].permute(1, 2, 0)  # (t q c)
        matching_result = matching_result[0]

        # per frame
        contrastive_items = []
        for i in range(outputs.size(0)):

            gt2ref = {}
            for i_ref, i_gt in zip(matching_result[0], matching_result[1]):
                gt2ref[i_gt.item()] = i_ref.item()
            # per instance
            for i_gt in gt2ref.keys():
                i_ref = gt2ref[i_gt]
                anchor_embeds = outputs[i][[i_ref]]  # (1, c)
                pos_embeds = outputs[:, i_ref]  # (t, c)
                neg_range = list(range(0, i_ref)) + list(range(i_ref + 1, outputs.size(1)))
                neg_embeds = outputs[i][neg_range] # (q - 1, c)

                num_positive = pos_embeds.shape[0]
                # concate pos and neg to get whole constractive samples
                pos_neg_embedding = torch.cat(
                    [pos_embeds, neg_embeds], dim=0)
                # generate label, pos is 1, neg is 0
                pos_neg_label = pos_neg_embedding.new_zeros((pos_neg_embedding.shape[0],),
                                                            dtype=torch.int64)  # noqa
                pos_neg_label[:num_positive] = 1.

                # dot product
                dot_product = torch.einsum(
                    'ac,kc->ak', [pos_neg_embedding, anchor_embeds])
                aux_normalize_pos_neg_embedding = nn.functional.normalize(
                    pos_neg_embedding, dim=1)
                aux_normalize_anchor_embedding = nn.functional.normalize(
                    anchor_embeds, dim=1)

                aux_cosine_similarity = torch.einsum('ac,kc->ak', [aux_normalize_pos_neg_embedding,
                                                                   aux_normalize_anchor_embedding])
                contrastive_items.append({
                    'dot_product': dot_product,
                    'cosine_similarity': aux_cosine_similarity,
                    'label': pos_neg_label})

        losses = loss_reid(contrastive_items, outputs_)
        return losses

    def get_cl_loss_with_memory(self, outputs_, matching_result, targets):
        # outputs['pred_keys'] = (b t) q c
        # outputs['pred_references'] = (b t) q c
        assert outputs_['pred_embds'].shape[0] == len(matching_result) == len(targets) == 1
        outputs = outputs_['pred_embds'][0].permute(1, 2, 0)  # (t q c)
        matching_result = matching_result[0]
        targets = targets[0]

        # per frame
        contrastive_items = []
        for i in range(outputs.size(0)):

            gt2ref = {}
            for i_ref, i_gt in zip(matching_result[0], matching_result[1]):
                gt2ref[i_gt.item()] = i_ref.item()
            # per instance
            for i_gt in gt2ref.keys():
                i_ref = gt2ref[i_gt]
                anchor_embeds = outputs[i][[i_ref]]  # (1, c)
                pos_embeds = outputs[:, i_ref]  # (t, c)
                neg_range = list(range(0, i_ref)) + list(range(i_ref + 1, outputs.size(1)))
                neg_embeds = outputs[i][neg_range] # (q - 1, c)

                num_positive = pos_embeds.shape[0]
                # concate pos and neg to get whole constractive samples
                pos_neg_embedding = torch.cat(
                    [pos_embeds, neg_embeds], dim=0)
                # generate label, pos is 1, neg is 0
                pos_neg_label = pos_neg_embedding.new_zeros((pos_neg_embedding.shape[0],),
                                                            dtype=torch.int64)  # noqa
                pos_neg_label[:num_positive] = 1.

                # dot product
                dot_product = torch.einsum(
                    'ac,kc->ak', [pos_neg_embedding, anchor_embeds])
                aux_normalize_pos_neg_embedding = nn.functional.normalize(
                    pos_neg_embedding, dim=1)
                aux_normalize_anchor_embedding = nn.functional.normalize(
                    anchor_embeds, dim=1)

                aux_cosine_similarity = torch.einsum('ac,kc->ak', [aux_normalize_pos_neg_embedding,
                                                                   aux_normalize_anchor_embedding])
                contrastive_items.append({
                    'dot_product': dot_product,
                    'cosine_similarity': aux_cosine_similarity,
                    'label': pos_neg_label})

                # cls cl
                cls = targets['labels'][i_gt].item()
                anchor_embeds = outputs[i][[i_ref]]
                pos_embeds = outputs[:, i_ref]  # (t, c)
                neg_embeds = self.classes_references_memory.get_items(cls)
                if len(neg_embeds) != 0:
                    num_positive = pos_embeds.shape[0]
                    # concate pos and neg to get whole constractive samples
                    pos_neg_embedding = torch.cat(
                        [pos_embeds, neg_embeds], dim=0)
                    # generate label, pos is 1, neg is 0
                    pos_neg_label = pos_neg_embedding.new_zeros((pos_neg_embedding.shape[0],),
                                                                dtype=torch.int64)  # noqa
                    pos_neg_label[:num_positive] = 1.

                    # dot product
                    dot_product = torch.einsum(
                        'ac,kc->ak', [pos_neg_embedding, anchor_embeds])
                    aux_normalize_pos_neg_embedding = nn.functional.normalize(
                        pos_neg_embedding, dim=1)
                    aux_normalize_anchor_embedding = nn.functional.normalize(
                        anchor_embeds, dim=1)

                    aux_cosine_similarity = torch.einsum('ac,kc->ak', [aux_normalize_pos_neg_embedding,
                                                                       aux_normalize_anchor_embedding])
                    contrastive_items.append({
                        'dot_product': dot_product,
                        'cosine_similarity': aux_cosine_similarity,
                        'label': pos_neg_label})

        self.classes_references_memory.push_refiner(outputs, targets, matching_result)
        losses = loss_reid(contrastive_items, outputs_)
        return losses