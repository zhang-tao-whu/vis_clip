import random
import torch
import torch.nn.functional as F

class ReferencesMemory:
    def __init__(self, max_len=5,):
        self.references = []
        self.weighted_references = []
        self.exist_frames = 0
        self.max_len = max_len

    def append(self, references):
        # references (q, b, c)
        if len(self.references) == 0:
            self.references.append(references)
            self.similarity_guided_references = references
            self.weighted_references.append(references)
        else:
            # Similarity-Guided Feature Fusion
            # https://arxiv.org/abs/2203.14208v1
            all_references = torch.stack(self.references[-self.max_len:], dim=0)  #(t, q, b, c)
            # mean similarity with all previous embed
            similarity = torch.sum(torch.einsum("tqbc,qbc->tqb",
                                                F.normalize(
                                                    all_references, dim=-1),
                                                F.normalize(references,
                                                            dim=-1)), dim=0) / min(self.exist_frames, self.max_len)  # (q, b)
            beta = torch.clamp(similarity, 0, 1).unsqueeze(2)
            # momentum update similarity_guided_reid_embed, maye avoid the negative effect of occlusion
            self.similarity_guided_references = (1 - beta) * self.similarity_guided_references + beta * references  # noqa
            self.weighted_references.append(
                self.similarity_guided_references)
            self.references.append(references)

        self.exist_frames += 1
        return self.similarity_guided_references

    def get_items(self):
        references = torch.stack(self.references, dim=0)
        weighted_references = torch.stack(self.weighted_references, dim=0)
        return references, weighted_references  # (t, q, b, c)