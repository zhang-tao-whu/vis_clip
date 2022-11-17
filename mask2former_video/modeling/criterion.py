# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/detr.py
"""
MaskFormer criterion.
"""
import logging

import torch
import torch.nn.functional as F
from torch import nn

from detectron2.utils.comm import get_world_size
from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)

from mask2former.utils.misc import is_dist_avail_and_initialized


def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


dice_loss_jit = torch.jit.script(
    dice_loss
)  # type: torch.jit.ScriptModule


def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    return loss.mean(1).sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(
    sigmoid_ce_loss
)  # type: torch.jit.ScriptModule


def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))


class VideoSetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses,
                 num_points, oversample_ratio, importance_sample_ratio, frames=2):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

        # pointwise mask loss parameters
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        self.frames = frames

    def loss_labels(self, outputs, targets, indices, num_masks):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"].float()

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {"loss_ce": loss_ce}
        return losses
    
    def loss_masks(self, outputs, targets, indices, num_masks):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        # Modified to handle video
        target_masks = torch.cat([t['masks'][i] for t, (_, i) in zip(targets, indices)]).to(src_masks)

        # No need to upsample predictions as we are using normalized coordinates :)
        # NT x 1 x H x W
        src_masks = src_masks.flatten(0, 1)[:, None]
        target_masks = target_masks.flatten(0, 1)[:, None]

        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        losses = {
            "loss_mask": sigmoid_ce_loss_jit(point_logits, point_labels, num_masks),
            "loss_dice": dice_loss_jit(point_logits, point_labels, num_masks),
        }

        del src_masks
        del target_masks
        return losses

    def loss_contrast(self, outputs, targets, indices, num_masks, neg_num=3):
        refer_embeds = []
        pos_embeds = []
        neg_embeds = []

        pred_embeds = outputs['pred_embds'].permute(0, 2, 3, 1).flatten(0, 1)  # pred_embeds: (bt, q, c)

        num_video = pred_embeds.size(0) // self.frames

        for i in range(num_video):
            frame_masks = []
            frame_embeds = []
            frame_ids = []
            for j in range(self.frames):
                idx = i * self.frames + j
                indice = indices[idx]
                frame_masks.append(targets[idx]['masks'][indice[1]].squeeze(1))  # (n, h, w)
                frame_ids.append(targets[idx]['ids'][indice[1]].squeeze(1))  # (n)
                frame_embeds.append(pred_embeds[idx][indice[0]])  # (n, c)
            self._select_pos_neg_embeds(frame_masks, frame_ids, frame_embeds, refer_embeds,
                                        pos_embeds, neg_embeds, neg_num=neg_num)
        if len(refer_embeds) == 0:
            return {"loss_ce": pred_embeds.sum() * 0.0}
        refer_embeds = torch.cat(refer_embeds, dim=0)
        pos_embeds = torch.cat(pos_embeds, dim=0)
        neg_embeds = torch.cat(neg_embeds, dim=0)
        targets_embeds = torch.cat([pos_embeds, neg_embeds], dim=1)

        refer_embeds = refer_embeds / refer_embeds.norm(dim=2)[:, :, None]  # (n, 1, c)
        targets_embeds = targets_embeds / targets_embeds.norm(dim=2)[:, :, None]  # (n, 1+neg_num, c)

        cos_sim = (refer_embeds * targets_embeds).sum(dim=2)  # (n, 1+neg_num)
        target_classes = torch.full(
            cos_sim.shape[:1], 0, dtype=torch.int64, device=cos_sim.device
        )
        empty_weight = torch.ones(neg_num + 1) / neg_num
        empty_weight[0] = 1
        loss_contrast = F.cross_entropy(cos_sim, target_classes, empty_weight.to(cos_sim.device))
        return {"loss_ce": loss_contrast}

    def get_bounding_boxes(self, masks):
        """
        Returns:
            Boxes: tight bounding boxes around bitmasks.
            If a mask is empty, it's bounding box will be all zero.
        """
        boxes = torch.zeros(masks.shape[0], 4, dtype=torch.float32)
        x_any = torch.any(masks, dim=1)
        y_any = torch.any(masks, dim=2)
        for idx in range(masks.shape[0]):
            x = torch.where(x_any[idx, :])[0]
            y = torch.where(y_any[idx, :])[0]
            if len(x) > 0 and len(y) > 0:
                boxes[idx, :] = torch.as_tensor(
                    [x[0], y[0], x[-1] + 1, y[-1] + 1], dtype=torch.float32
                )
        return boxes

    def _select_pos_neg_embeds(self, frame_masks, frame_ids, frame_embeds, refer_embeds, pos_embeds, neg_embeds, neg_num=3):
        assert len(frame_masks) == len(frame_embeds) == len(frame_ids) == self.frames
        refer_idx = self.frames // 2
        refer_id = frame_ids[refer_idx]
        valid = refer_id != -1
        refer_boxes = self.get_bounding_boxes(frame_masks[refer_idx][valid]) # (n, 4) [minx, miny, maxx, maxy]
        refer_embed = frame_embeds[refer_idx][valid]
        refer_id = refer_id[valid] # remove not avaliable in refer frame

        for i in range(self.frames):
            target_id = frame_ids[i]
            target_embed = frame_embeds[i]
            target_boxes = self.get_bounding_boxes(frame_masks[i])
            #select pos
            is_same_id = ((refer_id.unsqueeze(1) - target_id.unsqueeze(0)) == 0).to(torch.float32)
            id_refer, id_target = torch.nonzero(is_same_id, as_tuple=True)
            refer_embeds.append(refer_embed[id_refer].unsqueeze(1))
            pos_embeds.append(target_embed[id_target].unsqueeze(1))
            #select neg
            distance = torch.sum((refer_boxes.unsqueeze(1) - target_boxes.unsqueeze(0)) ** 2, dim=-1) ** 0.5
            distance = is_same_id * 1e6 + distance.to(is_same_id.device)
            _neg_num = min(neg_num, distance.size(1) - 1)
            # if none neg, pass
            if _neg_num == 0:
                refer_embeds.pop()
                pos_embeds.pop()
                continue
            _, id_neg = torch.topk(distance, k=_neg_num, dim=1, largest=False)
            # is real neg num < neg_num, repeat to neg_num
            if _neg_num != neg_num:
                id_neg = torch.cat([id_neg, torch.flip(id_neg, [1])] * neg_num, dim=1)[:, :neg_num]
            neg_embeds.append(target_embed[id_neg])
            if neg_embeds[-1].size(0) != pos_embeds[-1].size(0):
                print(neg_embeds[-1].size(0), pos_embeds[-1].size(0))
        return

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_masks):
        loss_map = {
            'labels': self.loss_labels,
            'masks': self.loss_masks,
            'contrast': self.loss_contrast
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks)

    def forward(self, outputs, targets, use_contrast=False):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)
        # [per image indicates], per image indicates -> (pred inds, gt inds)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            if loss == 'contrast' and use_contrast is False:
                continue
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'contrast':
                        continue
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_masks)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "losses: {}".format(self.losses),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "eos_coef: {}".format(self.eos_coef),
            "num_points: {}".format(self.num_points),
            "oversample_ratio: {}".format(self.oversample_ratio),
            "importance_sample_ratio: {}".format(self.importance_sample_ratio),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
