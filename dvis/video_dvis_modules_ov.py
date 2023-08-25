import torch
from torch import nn
from mask2former_video.modeling.transformer_decoder.video_mask2former_transformer_decoder import SelfAttentionLayer,\
    CrossAttentionLayer, FFNLayer, MLP, _get_activation_fn
from scipy.optimize import linear_sum_assignment
from .utils import Noiser
import random
import numpy as np
import torch.nn.functional as F

def get_classification_logits(x, text_classifier, logit_scale, num_templates=None):
    # x in shape of [B, *, C]
    # text_classifier in shape of [num_classes, C]
    # logit_scale is a learnable scalar https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/model.py#L201
    # return: [B, *, num_classes]
    x = F.normalize(x, dim=-1)
    logit_scale = torch.clamp(logit_scale.exp(), max=100)
    pred_logits = logit_scale * x @ text_classifier.T # B, *, N + 1
    # max ensembel as in OpenSeg/ODISE
    final_pred_logits = []
    cur_idx = 0
    for num_t in num_templates[:-1]:
        final_pred_logits.append(pred_logits[..., cur_idx: cur_idx + num_t].max(-1).values)
        cur_idx += num_t
    # final_pred_logits.append(pred_logits[:, :, -1]) # the last classifier is for void
    final_pred_logits.append(pred_logits[..., -num_templates[-1]:].max(-1).values)
    final_pred_logits = torch.stack(final_pred_logits, dim=-1)
    return final_pred_logits

class ReferringCrossAttentionLayer(nn.Module):

    def __init__(
        self,
        d_model,
        nhead,
        dropout=0.0,
        activation="relu",
        normalize_before=False
    ):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
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
        memory,
        memory_mask=None,
        memory_key_padding_mask=None,
        pos=None,
        query_pos=None
    ):
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory, attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask)[0]
        tgt = indentify + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(
        self,
        indentify,
        tgt,
        memory,
        memory_mask=None,
        memory_key_padding_mask=None,
        pos=None,
        query_pos=None
    ):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt2, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory, attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask)[0]
        tgt = indentify + self.dropout(tgt2)

        return tgt

    def forward(
        self,
        indentify,
        tgt,
        memory,
        memory_mask=None,
        memory_key_padding_mask=None,
        pos=None,
        query_pos=None
    ):
        # when set "indentify = tgt", ReferringCrossAttentionLayer is same as CrossAttentionLayer
        if self.normalize_before:
            return self.forward_pre(indentify, tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(indentify, tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)


class ReferringTracker_noiser_OV(torch.nn.Module):
    def __init__(
        self,
        hidden_channel=256,
        feedforward_channel=2048,
        num_head=8,
        decoder_layer_num=6,
        mask_dim=256,
        class_num=25,
        noise_mode='hard',
        feature_refusion=False,
        multi_layer_noise=False,
        use_memory=False,
        memory_length=4,
        # frozen fc-clip head
        mask_pooling=None,
        mask_pooling_proj=None,
        class_embed=None,
        logit_scale=None,
    ):
        super(ReferringTracker_noiser_OV, self).__init__()

        # init transformer layers
        self.num_heads = num_head
        self.num_layers = decoder_layer_num
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()
        if feature_refusion:
            self.memory_feature = None
            self.feature2query_fusion_layers = nn.ModuleList()

            self.mem_cur_feature_fusion = SelfAttentionLayer(
                d_model=hidden_channel,
                nhead=num_head,
                dropout=0.0,
                normalize_before=False,
            )

            self.feature_proj = nn.Conv2d(
                mask_dim,
                mask_dim,
                kernel_size=1,
                stride=1,
                padding=0,
            )

        for _ in range(self.num_layers):
            if feature_refusion:
                self.feature2query_fusion_layers.append(
                    ReferringCrossAttentionLayer(
                        d_model=hidden_channel,
                        nhead=num_head,
                        dropout=0.0,
                        normalize_before=False,
                    )
                )

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
                # GatedReferringCrossAttentionLayer(
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
        self.mask_embed = MLP(hidden_channel, hidden_channel, mask_dim, 3)
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

        self.noiser = Noiser(noise_ratio=0.8, mode=noise_mode)

        self.feature_refusion = feature_refusion
        self.multi_layer_noise = multi_layer_noise

        self.use_memory = use_memory
        if use_memory:
            self.memory_length = memory_length
            self.none_embed = nn.Embedding(1, hidden_channel)
            self.cur_pos = nn.Embedding(1, hidden_channel)
            self.temporal_pos_embed = nn.Embedding(memory_length, hidden_channel)
            self.transformer_cross_attention_layers_memory = nn.ModuleList()
            for _ in range(self.num_layers):
                self.transformer_cross_attention_layers_memory.append(
                    CrossAttentionLayer(
                        d_model=hidden_channel,
                        nhead=num_head,
                        dropout=0.0,
                        normalize_before=False,
                    )
                )
            self.memory = None

        # FC-CLIP
        self.mask_pooling = mask_pooling
        self._mask_pooling_proj = mask_pooling_proj
        self.class_embed = class_embed
        self.logit_scale = logit_scale

    def get_memory(self, bs):
        if self.memory is None:
            self.memory = self.none_embed.weight.unsqueeze(0).repeat(self.memory_length, bs, 1)
        return self.memory

    def push_memory(self, query):
        query = query.flatten(0, 1).unsqueeze(0)
        self.memory = torch.cat([self.memory, query], dim=0)[1:]
        return

    def _clear_memory(self):
        del self.last_outputs
        self.last_outputs = None
        if self.feature_refusion:
            del self.memory_feature
            self.memory_feature = None
        if self.use_memory:
            self.memory = None
        return

    def feature_query_fusion(self, feature, mask_features, memory_feature=None):
        # query (q, b, c)
        # feature (b, c, h, w)
        b, c, hf, wf = feature.size()
        _, _, hm, wm = mask_features.size()

        feature = feature.flatten(2).permute(2, 0, 1)  # (hw, b, c)
        if memory_feature is None:
            memory_feature = feature

        # do mem_cur_feature_fusion
        mem_cur_feature = torch.cat([memory_feature, feature], dim=0)  # (2hw, b, c)
        mem_cur_feature = self.mem_cur_feature_fusion(
            mem_cur_feature, tgt_mask=None,
            tgt_key_padding_mask=None,
            query_pos=None
        )
        feature_ = mem_cur_feature[hf*wf:]

        feature = feature_.reshape(hf, wf, b, c).permute(2, 3, 0, 1)
        feature = F.interpolate(feature, size=(hm, wm), mode='bilinear', align_corners=True)
        mask_features = mask_features + self.feature_proj(feature)
        return mask_features, feature_


    def forward(self, frame_embeds, mask_features, resume=False,
                return_indices=False, frame_classes=None,
                frame_embeds_no_norm=None, cur_feature=None,
                text_classifier=None, num_templates=None,
                ):
        """
        :param frame_embeds: the instance queries output by the segmenter
        :param mask_features: the mask features output by the segmenter
        :param resume: whether the first frame is the start of the video
        :param return_indices: whether return the match indices
        :return: output dict, including masks, classes, embeds.
        """
        # mask feature projection
        if self.feature_refusion:
            assert cur_feature is not None
        mask_features_shape = mask_features.shape
        mask_features = self.mask_feature_proj(mask_features.flatten(0, 1)).reshape(*mask_features_shape)  # (b, t, c, h, w)
        mask_features_ = []

        frame_embeds = frame_embeds.permute(2, 3, 0, 1)  # t, q, b, c
        if frame_embeds_no_norm is not None:
            frame_embeds_no_norm = frame_embeds_no_norm.permute(2, 3, 0, 1)  # t, q, b, c
        n_frame, n_q, bs, _ = frame_embeds.size()
        outputs = []
        ret_indices = []

        if self.use_memory:
            temporal_pos = self.temporal_pos_embed.weight.unsqueeze(1).repeat(1, n_q * bs, 1)
            cur_pos = self.cur_pos.weight.unsqueeze(1).repeat(1, n_q * bs, 1)

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
            if i == 0 and resume is False:
                self._clear_memory()
            # do fusion
            if self.feature_refusion:
                single_frame_feature = cur_feature[i: i + 1]  # (1, c, h, w)
                single_frame_mask_feature, self.memory_feature = self.feature_query_fusion(
                    single_frame_feature, mask_features[0, i: i + 1], self.memory_feature
                )
                mask_features_.append(single_frame_mask_feature)
            if self.feature_refusion:
                single_frame_feature = cur_feature[i: i + 1].flatten(2).permute(2, 0, 1)
            if self.use_memory:
                memory = self.get_memory(bs=n_q * bs)
            # the first frame of a video
            if i == 0 and resume is False:
                # self._clear_memory()
                for j in range(self.num_layers):
                    if j == 0:
                        indices, noised_init = self.noiser(
                            single_frame_embeds,
                            single_frame_embeds,
                            cur_embeds_no_norm=single_frame_embeds_no_norm,
                            activate=False,
                            cur_classes=single_frame_classes,
                        )
                        ms_output.append(single_frame_embeds_no_norm[indices])
                        self.last_frame_embeds = single_frame_embeds[indices]
                        ret_indices.append(indices)
                        output = self.transformer_cross_attention_layers[j](
                            noised_init, single_frame_embeds_no_norm, single_frame_embeds_no_norm,
                            memory_mask=None,
                            memory_key_padding_mask=None,
                            pos=None, query_pos=None
                        )
                        # refusion
                        if self.feature_refusion:
                            output = self.feature2query_fusion_layers[j](
                                output, single_frame_embeds_no_norm, single_frame_feature
                            )

                        if self.use_memory:
                            output = output.flatten(0, 1).unsqueeze(0)
                            output = self.transformer_cross_attention_layers_memory[j](
                                output, self.memory,
                                memory_mask=None,
                                memory_key_padding_mask=None,
                                pos=temporal_pos, query_pos=cur_pos
                            )
                            output = output.reshape(n_q, bs, -1)

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
                            ms_output[-1], ms_output[-1], single_frame_embeds_no_norm,
                            memory_mask=None,
                            memory_key_padding_mask=None,
                            pos=None, query_pos=None
                        )
                        # refusion
                        if self.feature_refusion:
                            output = self.feature2query_fusion_layers[j](
                                output, ms_output[-1], single_frame_feature
                            )

                        if self.use_memory:
                            output = output.flatten(0, 1).unsqueeze(0)
                            output = self.transformer_cross_attention_layers_memory[j](
                                output, self.memory,
                                memory_mask=None,
                                memory_key_padding_mask=None,
                                pos=temporal_pos, query_pos=cur_pos
                            )
                            output = output.reshape(n_q, bs, -1)

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
                for j in range(self.num_layers):
                    if j == 0:
                        indices, noised_init = self.noiser(
                            self.last_frame_embeds,
                            single_frame_embeds,
                            cur_embeds_no_norm=single_frame_embeds_no_norm,
                            activate=self.training,
                            cur_classes=single_frame_classes,
                        )
                        ms_output.append(single_frame_embeds_no_norm[indices])
                        self.last_frame_embeds = single_frame_embeds[indices]
                        ret_indices.append(indices)
                        output = self.transformer_cross_attention_layers[j](
                            noised_init, self.last_outputs[-1], single_frame_embeds_no_norm,
                            memory_mask=None,
                            memory_key_padding_mask=None,
                            pos=None, query_pos=None
                        )
                        # refusion
                        if self.feature_refusion:
                            output = self.feature2query_fusion_layers[j](
                                output, self.last_outputs[-1], single_frame_feature
                            )

                        if self.use_memory:
                            output = output.flatten(0, 1).unsqueeze(0)
                            output = self.transformer_cross_attention_layers_memory[j](
                                output, self.memory,
                                memory_mask=None,
                                memory_key_padding_mask=None,
                                pos=temporal_pos, query_pos=cur_pos
                            )
                            output = output.reshape(n_q, bs, -1)

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
                        if self.multi_layer_noise and self.training:
                            output = self.transformer_cross_attention_layers[j](
                                self.soft_noise(ms_output[-1], ratio=0.5 / j), self.last_outputs[-1], single_frame_embeds_no_norm,
                                memory_mask=None,
                                memory_key_padding_mask=None,
                                pos=None, query_pos=None
                            )
                        else:
                            output = self.transformer_cross_attention_layers[j](
                                ms_output[-1], self.last_outputs[-1], single_frame_embeds_no_norm,
                                memory_mask=None,
                                memory_key_padding_mask=None,
                                pos=None, query_pos=None
                            )
                        # refusion
                        if self.feature_refusion:
                            output = self.feature2query_fusion_layers[j](
                                output, self.last_outputs[-1], single_frame_feature
                            )

                        if self.use_memory:
                            output = output.flatten(0, 1).unsqueeze(0)
                            output = self.transformer_cross_attention_layers_memory[j](
                                output, self.memory,
                                memory_mask=None,
                                memory_key_padding_mask=None,
                                pos=temporal_pos, query_pos=cur_pos
                            )
                            output = output.reshape(n_q, bs, -1)

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
            if self.use_memory:
                self.push_memory(output)
            ms_output = torch.stack(ms_output, dim=0)  # (1 + layers, q, b, c)
            self.last_outputs = ms_output
            outputs.append(ms_output[1:])
        outputs = torch.stack(outputs, dim=0)  # (t, l, q, b, c)
        if self.feature_refusion:
            mask_features_ = torch.cat(mask_features_, dim=0).unsqueeze(0)
        else:
            mask_features_ = mask_features
        if not self.training:
            outputs = outputs[:, -1:]
            del mask_features
        outputs_class, outputs_masks = self.prediction(outputs, mask_features_, text_classifier, num_templates)
        #outputs = self.decoder_norm(outputs)
        out = {
           'pred_logits': outputs_class[-1].transpose(1, 2),  # (b, t, q, c)
           'pred_masks': outputs_masks[-1],  # (b, q, t, h, w)
           'aux_outputs': self._set_aux_loss(
               outputs_class, outputs_masks
           ),
           'pred_embds': outputs[:, -1].permute(2, 3, 0, 1)  # (b, c, t, q)
        }
        if return_indices:
            return out, ret_indices
        else:
            return out

    def soft_noise(self, queries, ratio=0.5):
        # queries (q, b, c)
        indices = list(range(queries.shape[0]))
        np.random.shuffle(indices)
        noise = queries[indices]
        ratio = ratio * random.random()
        queries = queries * (1 - ratio) + noise * ratio
        return queries

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{"pred_logits": a.transpose(1, 2), "pred_masks": b}
                for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
                ]

    def prediction(self, outputs, mask_features, text_classifier, num_templates):
        # outputs (t, l, q, b, c)
        # mask_features (b, t, c, h, w)
        decoder_output = self.decoder_norm(outputs)
        decoder_output = decoder_output.permute(1, 3, 0, 2, 4)  # (l, b, t, q, c)
        # outputs_class = self.class_embed(decoder_output).transpose(2, 3)  # (l, b, q, t, cls+1)
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("lbtqc,btchw->lbqthw", mask_embed, mask_features)

        # fc-clip class head forward
        # mean pooling
        b, t, c, _, _ = mask_features.shape
        l, b, q, t, _, _ = outputs_mask.shape
        mask_features = mask_features.unsqueeze(0).repeat(l, 1, 1, 1, 1, 1).flatten(0, 2)  # lbt, c, h, w
        outputs_mask_ = outputs_mask.permute(0, 1, 3, 2, 4, 5).flatten(0, 2)  # (lbt, q, h, w)
        maskpool_embeddings = self.mask_pooling(x=mask_features, mask=outputs_mask_)  # [lbt, q, c]
        maskpool_embeddings = maskpool_embeddings.reshape(l, b, t, *maskpool_embeddings.shape[-2:]) # (l b t q c)
        maskpool_embeddings = self._mask_pooling_proj(maskpool_embeddings)
        class_embed = self.class_embed(maskpool_embeddings + decoder_output)
        outputs_class = get_classification_logits(class_embed, text_classifier, self.logit_scale, num_templates)
        outputs_class = outputs_class.transpose(2, 3)  # (l, b, q, t, cls+1)

        return outputs_class, outputs_mask

# class TemporalRefiner(torch.nn.Module):
#     def __init__(
#         self,
#         hidden_channel=256,
#         feedforward_channel=2048,
#         num_head=8,
#         decoder_layer_num=6,
#         mask_dim=256,
#         class_num=25,
#         windows=5,
#         mask_agu=False,
#         mask_ratio=0.4,
#     ):
#         super(TemporalRefiner, self).__init__()
#
#         self.windows = windows
#
#         # init transformer layers
#         self.num_heads = num_head
#         self.num_layers = decoder_layer_num
#         self.transformer_obj_self_attention_layers = nn.ModuleList()
#         self.transformer_time_self_attention_layers = nn.ModuleList()
#         self.transformer_cross_attention_layers = nn.ModuleList()
#         self.transformer_ffn_layers = nn.ModuleList()
#
#         self.conv_short_aggregate_layers = nn.ModuleList()
#         self.conv_norms = nn.ModuleList()
#
#         for _ in range(self.num_layers):
#             self.transformer_time_self_attention_layers.append(
#                 SelfAttentionLayer(
#                     d_model=hidden_channel,
#                     nhead=num_head,
#                     dropout=0.0,
#                     normalize_before=False,
#                 )
#             )
#
#             self.conv_short_aggregate_layers.append(
#                 nn.Sequential(
#                     nn.Conv1d(hidden_channel, hidden_channel,
#                               kernel_size=5, stride=1,
#                               padding='same', padding_mode='replicate'),
#                     nn.ReLU(inplace=True),
#                     nn.Conv1d(hidden_channel, hidden_channel,
#                               kernel_size=3, stride=1,
#                               padding='same', padding_mode='replicate'),
#                 )
#             )
#
#             self.conv_norms.append(nn.LayerNorm(hidden_channel))
#
#             self.transformer_obj_self_attention_layers.append(
#                 SelfAttentionLayer(
#                     d_model=hidden_channel,
#                     nhead=num_head,
#                     dropout=0.0,
#                     normalize_before=False,
#                 )
#             )
#
#             self.transformer_cross_attention_layers.append(
#                 CrossAttentionLayer(
#                     d_model=hidden_channel,
#                     nhead=num_head,
#                     dropout=0.0,
#                     normalize_before=False,
#                 )
#             )
#
#             self.transformer_ffn_layers.append(
#                 FFNLayer(
#                     d_model=hidden_channel,
#                     dim_feedforward=feedforward_channel,
#                     dropout=0.0,
#                     normalize_before=False,
#                 )
#             )
#
#         self.decoder_norm = nn.LayerNorm(hidden_channel)
#
#         # init heads
#         self.class_embed = nn.Linear(hidden_channel, class_num + 1)
#         self.mask_embed = MLP(hidden_channel, hidden_channel, mask_dim, 3)
#
#         self.activation_proj = nn.Linear(hidden_channel, 1)
#
#         # mask agumentation
#         self.mask_agu = mask_agu
#         self.mask_ratio = mask_ratio
#
#     def forward(self, instance_embeds, frame_embeds, mask_features):
#         """
#         :param instance_embeds: the aligned instance queries output by the tracker, shape is (b, c, t, q)
#         :param frame_embeds: the instance queries processed by the tracker.frame_forward function, shape is (b, c, t, q)
#         :param mask_features: the mask features output by the segmenter, shape is (b, t, c, h, w)
#         :return: output dict, including masks, classes, embeds.
#         """
#         n_batch, n_channel, n_frames, n_instance = instance_embeds.size()
#
#         if self.training and self.mask_agu:
#             temporal_mask = torch.rand(n_frames, n_frames).to(instance_embeds)
#             temporal_mask = torch.maximum(temporal_mask, torch.eye(n_frames).to(instance_embeds))
#             temporal_mask = temporal_mask <= self.mask_ratio
#         else:
#             temporal_mask = None
#
#         outputs = []
#         output = instance_embeds
#         frame_embeds = frame_embeds.permute(3, 0, 2, 1).flatten(1, 2)
#
#         for i in range(self.num_layers):
#             output = output.permute(2, 0, 3, 1)  # (t, b, q, c)
#             output = output.flatten(1, 2)  # (t, bq, c)
#
#             # do long temporal attention
#             output = self.transformer_time_self_attention_layers[i](
#                 output, tgt_mask=temporal_mask,
#                 tgt_key_padding_mask=None,
#                 query_pos=None
#             )
#
#             # do short temporal conv
#             output = output.permute(1, 2, 0)  # (bq, c, t)
#             output = self.conv_norms[i](
#                 (self.conv_short_aggregate_layers[i](output) + output).transpose(1, 2)
#             ).transpose(1, 2)
#             output = output.reshape(
#                 n_batch, n_instance, n_channel, n_frames
#             ).permute(1, 0, 3, 2).flatten(1, 2)  # (q, bt, c)
#
#             # do objects self attention
#             output = self.transformer_obj_self_attention_layers[i](
#                 output, tgt_mask=None,
#                 tgt_key_padding_mask=None,
#                 query_pos=None
#             )
#
#             # do cross attention
#             output = self.transformer_cross_attention_layers[i](
#                 output, frame_embeds,
#                 memory_mask=None,
#                 memory_key_padding_mask=None,
#                 pos=None, query_pos=None
#             )
#
#             # FFN
#             output = self.transformer_ffn_layers[i](
#                 output
#             )
#
#             output = output.reshape(n_instance, n_batch, n_frames, n_channel).permute(1, 3, 2, 0)  # (b, c, t, q)
#             outputs.append(output)
#
#         outputs = torch.stack(outputs, dim=0).permute(3, 0, 4, 1, 2)  # (l, b, c, t, q) -> (t, l, q, b, c)
#         outputs_class, outputs_masks = self.prediction(outputs, mask_features)
#         outputs = self.decoder_norm(outputs)
#         out = {
#            'pred_logits': outputs_class[-1].transpose(1, 2),  # (b, t, q, c)
#            'pred_masks': outputs_masks[-1],  # (b, q, t, h, w)
#            'aux_outputs': self._set_aux_loss(
#                outputs_class, outputs_masks
#            ),
#            'pred_embds': outputs[:, -1].permute(2, 3, 0, 1)  # (b, c, t, q)
#         }
#         return out
#
#     @torch.jit.unused
#     def _set_aux_loss(self, outputs_class, outputs_seg_masks):
#         # this is a workaround to make torchscript happy, as torchscript
#         # doesn't support dictionary with non-homogeneous values, such
#         # as a dict having both a Tensor and a list.
#         return [{"pred_logits": a.transpose(1, 2), "pred_masks": b}
#                 for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
#                 ]
#
#     def windows_prediction(self, outputs, mask_features, windows=5):
#         """
#         for windows prediction, because mask features consumed too much GPU memory
#         """
#         iters = outputs.size(0) // windows
#         if outputs.size(0) % windows != 0:
#             iters += 1
#         outputs_classes = []
#         outputs_masks = []
#         for i in range(iters):
#             start_idx = i * windows
#             end_idx = (i + 1) * windows
#             clip_outputs = outputs[start_idx:end_idx]
#             decoder_output = self.decoder_norm(clip_outputs)
#             decoder_output = decoder_output.permute(1, 3, 0, 2, 4)  # (l, b, t, q, c)
#             mask_embed = self.mask_embed(decoder_output)
#             outputs_mask = torch.einsum(
#                 "lbtqc,btchw->lbqthw",
#                 mask_embed,
#                 mask_features[:, start_idx:end_idx].to(mask_embed.device)
#             )
#             outputs_classes.append(decoder_output)
#             outputs_masks.append(outputs_mask.cpu().to(torch.float32))
#         outputs_classes = torch.cat(outputs_classes, dim=2)
#         outputs_classes = self.pred_class(outputs_classes)
#         return outputs_classes.cpu().to(torch.float32), torch.cat(outputs_masks, dim=3)
#
#     def pred_class(self, decoder_output):
#         """
#         fuse the objects queries of all frames and predict an overall score based on the fused objects queries
#         :param decoder_output: instance queries, shape is (l, b, t, q, c)
#         """
#         T = decoder_output.size(2)
#
#         # compute the weighted average of the decoder_output
#         activation = self.activation_proj(decoder_output).softmax(dim=2)  # (l, b, t, q, 1)
#         class_output = (decoder_output * activation).sum(dim=2, keepdim=True)  # (l, b, 1, q, c)
#
#         # to unify the output format, duplicate the fused features T times
#         class_output = class_output.repeat(1, 1, T, 1, 1)
#         outputs_class = self.class_embed(class_output).transpose(2, 3)
#         return outputs_class
#
#     def prediction(self, outputs, mask_features):
#         """
#         :param outputs: instance queries, shape is (t, l, q, b, c)
#         :param mask_features: mask features, shape is (b, t, c, h, w)
#         :return: pred class and pred masks
#         """
#         if self.training:
#             decoder_output = self.decoder_norm(outputs)
#             decoder_output = decoder_output.permute(1, 3, 0, 2, 4)  # (l, b, t, q, c)
#             outputs_class = self.pred_class(decoder_output)
#             mask_embed = self.mask_embed(decoder_output)
#             outputs_mask = torch.einsum("lbtqc,btchw->lbqthw", mask_embed, mask_features)
#         else:
#             outputs = outputs[:, -1:]
#             outputs_class, outputs_mask = self.windows_prediction(outputs, mask_features, windows=self.windows)
#         return outputs_class, outputs_mask

