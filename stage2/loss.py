"""Stage 2 distillation loss: MSE + cosine over per-frame pix_feat_with_mem.

Inputs are per-frame memory-conditioned feature maps from student and teacher,
shape [B, T, C, H, W]. The per-frame `attention_mask` [B, T] flags which frames
are real vs. zero-padded; padded frames contribute 0 loss.

Cosine distance is computed over the channel dim per spatial token, then mean-
pooled over (H, W) and averaged over valid frames.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DistillLossOutput:
    total: torch.Tensor
    mse: torch.Tensor
    cosine: torch.Tensor


class DistillLoss(nn.Module):
    def __init__(self, mse_weight: float = 1.0, cosine_weight: float = 1.0, eps: float = 1e-6):
        super().__init__()
        self.mse_weight = mse_weight
        self.cosine_weight = cosine_weight
        self.eps = eps

    def forward(
        self,
        student_feat: torch.Tensor,
        teacher_feat: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> DistillLossOutput:
        """
        student_feat / teacher_feat: [B, T, C, H, W]
        attention_mask:              [B, T] bool — True = real frame
        """
        assert student_feat.shape == teacher_feat.shape, \
            f"shape mismatch: {student_feat.shape} vs {teacher_feat.shape}"

        # match teacher dtype to student for stable AMP loss
        teacher_feat = teacher_feat.to(student_feat.dtype)

        B, T, C, H, W = student_feat.shape
        if attention_mask is None:
            attention_mask = torch.ones(B, T, dtype=torch.bool, device=student_feat.device)

        # per-frame masks broadcast over channel + spatial
        m = attention_mask.to(student_feat.dtype)        # [B, T]
        m_full = m.view(B, T, 1, 1, 1).expand_as(student_feat)
        n_valid_elems = m_full.sum().clamp_min(self.eps)

        # L2-normalise along channel dim so MSE is scale-invariant [0, 4]
        # Teacher feature magnitude varies 30-50x across SA-V videos; raw MSE
        # would scale as magnitude^2, causing loss to range 2->2700.
        s_norm = F.normalize(student_feat, p=2, dim=2)  # [B, T, C, H, W]
        t_norm = F.normalize(teacher_feat, p=2, dim=2)

        # ---- MSE on normalised features ----
        mse = ((s_norm - t_norm) ** 2 * m_full).sum() / n_valid_elems

        # ---- cosine (per spatial token, then mean) ----
        s = s_norm.permute(0, 1, 3, 4, 2).reshape(B * T * H * W, C)
        t = t_norm.permute(0, 1, 3, 4, 2).reshape(B * T * H * W, C)
        cos = F.cosine_similarity(s, t, dim=-1, eps=self.eps)  # [B*T*H*W]
        token_mask = m.view(B, T, 1, 1).expand(B, T, H, W).reshape(-1)
        n_valid_tokens = token_mask.sum().clamp_min(self.eps)
        cosine_loss = ((1.0 - cos) * token_mask).sum() / n_valid_tokens

        total = self.mse_weight * mse + self.cosine_weight * cosine_loss
        return DistillLossOutput(total=total, mse=mse.detach(), cosine=cosine_loss.detach())


def build_distill_loss(config) -> DistillLoss:
    return DistillLoss(
        mse_weight=config.DISTILL.MSE_WEIGHT,
        cosine_weight=config.DISTILL.COSINE_WEIGHT,
    )
