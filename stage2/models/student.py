"""Student wrapper: RepViT M0.9 image encoder + TemporalPerceiver.

Loads the Stage 1 student checkpoint (efficient_sam3_repvit_s.pt) via the
existing `build_efficientsam3_video_model` factory. Only the image encoder
path (`detector.backbone.vision_backbone`) and the tracker's FPN projections
(`tracker.sam_mask_decoder.conv_s0/s1`) are used here; the full tracker
memory bank is replaced by our TemporalPerceiver.

Per-frame path:
    img [B,3,H,W]
      → detector.backbone.forward_image(img)
      → sam2_backbone_out["backbone_fpn"] = [feat_0, feat_1, feat_2]  (stage 1/2/3)
      → conv_s0(feat_0), conv_s1(feat_1), feat_2 as-is  (all 256-d)
      → pix_feat = feat_2 [B,256,H',W']           (top-level feature)
      → perceiver(frame_tokens, prev_latents) → pix_feat_with_mem [B,256,H',W']
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import torch
import torch.nn as nn

# locate the vendored sam3 package without leaving the project root
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
_SAM3_PKG = os.path.join(_PROJECT_ROOT, 'sam3')
if _SAM3_PKG not in sys.path:
    sys.path.insert(0, _SAM3_PKG)

from sam3.model_builder import build_efficientsam3_video_model  # noqa: E402

from .perceiver import TemporalPerceiver


class StudentTemporalModel(nn.Module):
    """Student = (RepViT M0.9 backbone, frozen) + (TemporalPerceiver, trainable)."""

    def __init__(
        self,
        checkpoint_path: str,
        perceiver: TemporalPerceiver,
        backbone_type: str = 'repvit',
        model_name: str = 'm0.9',
        freeze_backbone: bool = True,
    ):
        super().__init__()

        base = build_efficientsam3_video_model(
            checkpoint_path=checkpoint_path,
            load_from_HF=False,
            backbone_type=backbone_type,
            model_name=model_name,
            strict_state_dict_loading=False,
        )
        # Keep only what we need: the detector's vision backbone + tracker's
        # FPN projection convs. Everything else (text encoder, decoder heads,
        # inference wrapper) is dead weight at training time.
        self.vision_backbone = base.detector.backbone
        self.conv_s0 = base.tracker.sam_mask_decoder.conv_s0
        self.conv_s1 = base.tracker.sam_mask_decoder.conv_s1

        self.perceiver = perceiver

        if freeze_backbone:
            for p in self.vision_backbone.parameters():
                p.requires_grad_(False)
            for p in self.conv_s0.parameters():
                p.requires_grad_(False)
            for p in self.conv_s1.parameters():
                p.requires_grad_(False)
            self.vision_backbone.eval()
            self.conv_s0.eval()
            self.conv_s1.eval()

        # free unused base submodules to reclaim VRAM
        del base

    # ------------------------------------------------------------------
    # Per-frame feature extraction
    # ------------------------------------------------------------------

    def encode_frame(self, img: torch.Tensor) -> torch.Tensor:
        """Run backbone + projection, return top-level pix_feat [B,256,H,W]."""
        out = self.vision_backbone.forward_image(img)
        sam2_out = out['sam2_backbone_out']
        # backbone_fpn: [feat_0 (highest res), feat_1, feat_2 (lowest res)]
        # conv_s0/s1 match the tracker's forward_image projection order.
        feat_top = sam2_out['backbone_fpn'][-1]
        return feat_top

    def forward(
        self,
        frames: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        frames: [B, T, 3, H, W]  — T = MAX_FRAMES (= 8)
        attention_mask: [B, T] bool — currently unused (padding handled by loss)

        Returns per-frame memory-conditioned features [B, T, 256, H', W'].
        """
        B, T, C, H, W = frames.shape
        outs = []
        latents = self.perceiver.init_memory(B, device=frames.device, dtype=frames.dtype)

        for t in range(T):
            pix_feat = self.encode_frame(frames[:, t])          # [B, 256, H', W']
            Bp, Cp, Hp, Wp = pix_feat.shape
            tokens = pix_feat.flatten(2).transpose(1, 2)        # [B, H'W', 256]
            latents, mem_feat = self.perceiver(tokens, latents, frame_idx=t)
            outs.append(mem_feat)

        return torch.stack(outs, dim=1)  # [B, T, 256, H', W']


def build_student_model(config) -> StudentTemporalModel:
    perceiver = TemporalPerceiver(
        dim=config.MODEL.PERCEIVER_DIM,
        depth=config.MODEL.PERCEIVER_DEPTH,
        num_latents=config.MODEL.PERCEIVER_LATENTS,
        num_heads=config.MODEL.PERCEIVER_HEADS,
        mlp_ratio=config.MODEL.PERCEIVER_MLP_RATIO,
        dropout=config.MODEL.PERCEIVER_DROPOUT,
    )
    model = StudentTemporalModel(
        checkpoint_path=config.MODEL.STUDENT_CKPT,
        perceiver=perceiver,
        backbone_type='repvit',
        model_name='m0.9' if 'm0_9' in config.MODEL.BACKBONE else 'm1.1',
        freeze_backbone=True,
    )
    return model
