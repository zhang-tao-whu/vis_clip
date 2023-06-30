import torch
import random
import numpy as np
from scipy.optimize import linear_sum_assignment

class Noiser:
    def __init__(self, noise_ratio=0.8, mode='hard', memory_max_len=100):
        assert mode in ['none', 'hard', 'object_hard', 'overall_class_hard']
        self.mode = mode
        self.noise_ratio = noise_ratio

        if self.mode == 'overall_class_hard':
            self.memory_bank = {}
            self.memory_max_len = memory_max_len

    def _hard_noise_forward(self, cur_embeds):
        indices = list(range(cur_embeds.shape[0]))
        np.random.shuffle(indices)
        noise_init = cur_embeds[indices]
        return indices, noise_init

    def _object_hard_noise_forward(self, cur_embeds, cur_classes):

        assert cur_classes is not None
        # embeds (q, b, c), classes (q)
        mask = cur_classes != -1
        indices = np.array(list(range(cur_embeds.shape[0])))
        indices = indices[mask.cpu().numpy()]
        if len(indices) == 0:
            indices = list(range(cur_embeds.shape[0]))
            np.random.shuffle(indices)
            return indices, cur_embeds[indices]
        rand_indices = torch.randint(low=0, high=len(indices), size=(cur_embeds.shape[0],))
        indices = list(indices[rand_indices])
        noise_init = cur_embeds[indices]
        return indices, noise_init

    def _push_new_embeds(self, cur_embeds, cur_classes):
        unique_cls = list(torch.unique(cur_classes, sorted=False).cpu().numpy())
        unique_cls = [int(item) for item in unique_cls]
        for _cls in unique_cls:
            if _cls == -1:
                pass
            else:
                if _cls not in self.memory_bank.keys():
                    _cls_embeds = cur_embeds[cur_classes == _cls]
                    rand_indices = torch.randint(low=0, high=_cls_embeds.size(0), size=(self.memory_max_len,))
                    self.memory_bank[_cls] = _cls_embeds[rand_indices]
                else:
                    self.memory_bank[_cls] = torch.cat([self.memory_bank[_cls], cur_embeds[cur_classes == _cls]], dim=0)
                    indices = list(range(self.memory_bank[_cls].shape[0]))
                    np.random.shuffle(indices)
                    self.memory_bank[_cls] = self.memory_bank[_cls][indices]
        return

    def _overall_class_hard_forward(self, cur_embeds, cur_classes):
        self._push_new_embeds(cur_embeds, cur_classes)
        rand_indices = torch.randint(low=0, high=self.memory_max_len, size=(cur_embeds.shape[0],))
        noise_init = torch.zeros_like(cur_embeds)
        unique_cls = list(torch.unique(cur_classes).cpu().numpy())
        unique_cls = [int(item) for item in unique_cls]
        for _cls in unique_cls:
            if _cls == -1:
                if len(unique_cls) == 1:
                    noise_init[cur_classes == _cls] = cur_embeds[cur_classes == _cls]
                else:
                    rand_cls = torch.randint(low=0, high=len(self.memory_bank), size=(1,))
                    rand_cls = list(self.memory_bank.keys())[rand_cls[0]]
                    noise_init[cur_classes == _cls] = self.memory_bank[rand_cls][rand_indices[cur_classes == _cls]]
            else:
                noise_init[cur_classes == _cls] = self.memory_bank[_cls][rand_indices[cur_classes == _cls]]
        return None, noise_init

    def match_embds(self, ref_embds, cur_embds):
        #  embeds (q, b, c)
        ref_embds, cur_embds = ref_embds.detach()[:, 0, :], cur_embds.detach()[:, 0, :]
        ref_embds = ref_embds / (ref_embds.norm(dim=1)[:, None] + 1e-6)
        cur_embds = cur_embds / (cur_embds.norm(dim=1)[:, None] + 1e-6)
        cos_sim = torch.mm(cur_embds, ref_embds.transpose(0, 1))
        C = 1 - cos_sim

        C = C.cpu()
        C = torch.where(torch.isnan(C), torch.full_like(C, 0), C)

        indices = linear_sum_assignment(C.transpose(0, 1))
        indices = indices[1]
        return indices

    def __call__(self, ref_embeds, cur_embeds, cur_embeds_no_norm=None, activate=False, cur_classes=None):
        if cur_embeds_no_norm is None:
            cur_embeds_no_norm = cur_embeds
        matched_indices = self.match_embds(ref_embeds, cur_embeds)
        if activate and random.random() < self.noise_ratio:
            if self.mode == 'hard':
                indices, noise_init = self._hard_noise_forward(cur_embeds_no_norm)
                return indices, noise_init
            elif self.mode == 'object_hard':
                indices, noise_init = self._object_hard_noise_forward(cur_embeds_no_norm, cur_classes)
                return indices, noise_init
            elif self.mode == 'overall_class_hard':
                indices, noise_init = self._overall_class_hard_forward(cur_embeds_no_norm, cur_classes)
                return matched_indices, noise_init
            elif self.mode == 'none':
                return matched_indices, cur_embeds_no_norm[matched_indices]
            else:
                raise NotImplementedError
        else:
            return matched_indices, cur_embeds_no_norm[matched_indices]
