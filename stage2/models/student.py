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
import torch.nn.functional as F

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
        prompt_conditioning: bool = False,
        img_size: int = 1008,
        feat_size: int = 72,
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

        # ---- mask-decode head (Milestone 1): turn pix_feat_with_mem -> masks ----
        # Reuse SAM3's frozen tracker so the student can emit masks for the mask
        # loss (and at inference). We keep a reference to the whole tracker and
        # call its `_forward_sam_heads`; it owns the SAM prompt-encoder + mask
        # decoder. Frozen — only the perceiver + our prompt path are trained.
        self.decode_masks_enabled = prompt_conditioning
        if self.decode_masks_enabled:
            self.tracker = base.tracker
            for p in self.tracker.parameters():
                p.requires_grad_(False)
            self.tracker.eval()
        else:
            self.tracker = None

        # ---- prompt-conditioning: encode a frame-0 mask/point/box into latents ----
        # Reuses SAM3's PromptEncoder (same embed_dim=256, 72x72 grid). The sparse
        # (point/box) + dense (mask) embeddings are flattened to prompt tokens that
        # seed the perceiver's initial latents toward the SELECTED object.
        self.prompt_conditioning = prompt_conditioning
        self.feat_size = feat_size
        if prompt_conditioning:
            from sam3.sam.prompt_encoder import PromptEncoder
            self.prompt_encoder = PromptEncoder(
                embed_dim=perceiver.dim,
                image_embedding_size=(feat_size, feat_size),
                input_image_size=(img_size, img_size),
                mask_in_chans=16,
            )
        else:
            self.prompt_encoder = None

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

        self._freeze_backbone = freeze_backbone

        # free unused base submodules to reclaim VRAM
        del base

    def train(self, mode: bool = True):
        """Keep frozen backbone modules in eval mode regardless of student mode.

        PyTorch's Module.train() recurses into all children, which would override
        the eval() calls set during __init__ for the frozen backbone. This override
        re-applies eval() after the recursive call so BN running stats stay fixed
        at their initial (Stage 1) values — making training-time val and standalone
        --eval produce consistent results.
        """
        super().train(mode)
        if self._freeze_backbone:
            self.vision_backbone.eval()
            self.conv_s0.eval()
            self.conv_s1.eval()
        if self.tracker is not None:
            self.tracker.eval()  # frozen SAM3 mask decoder stays in eval
        return self

    # ------------------------------------------------------------------
    # Per-frame feature extraction
    # ------------------------------------------------------------------

    def encode_frame(self, img: torch.Tensor, want_high_res: bool = False):
        """Run backbone + projection, return top-level pix_feat [B,256,H,W].

        If want_high_res, also returns the two higher-res FPN levels after the
        tracker's conv_s0/conv_s1 projection (needed by the SAM mask decoder):
        [feat_s0 (288x288), feat_s1 (144x144)].
        """
        out = self.vision_backbone.forward_image(img)
        sam2_out = out['sam2_backbone_out']
        # backbone_fpn: [feat_0 (highest res), feat_1, feat_2 (lowest res)]
        # conv_s0/s1 match the tracker's forward_image projection order.
        fpn = sam2_out['backbone_fpn']
        feat_top = fpn[-1]
        if not want_high_res:
            return feat_top
        high_res = [self.conv_s0(fpn[0]), self.conv_s1(fpn[1])]  # [288x288, 144x144]
        return feat_top, high_res

    def decode_masks(self, mem_feats: torch.Tensor,
                     high_res_per_frame: list) -> torch.Tensor:
        """Decode per-frame memory features into low-res mask logits.

        mem_feats: [B, T, 256, 72, 72] (output of forward()).
        high_res_per_frame: list of T [feat_s0, feat_s1] pairs from encode_frame.
        Returns:   [B, T, 1, 288, 288] mask logits (4x the feature stride).
        Uses the frozen SAM3 mask decoder via `_forward_sam_heads` with no prompt
        (pure memory decode — matches propagated t>0 inference).
        """
        assert self.tracker is not None, "decode_masks needs prompt_conditioning=True"
        B, T = mem_feats.shape[:2]
        logits = []
        for t in range(T):
            sam_outs = self.tracker._forward_sam_heads(
                backbone_features=mem_feats[:, t],          # [B,256,72,72]
                point_inputs=None,
                mask_inputs=None,
                high_res_features=high_res_per_frame[t],    # [feat_s0, feat_s1]
                multimask_output=False,
            )
            logits.append(sam_outs[3])  # low_res_masks [B,1,288,288]
        return torch.stack(logits, dim=1)  # [B, T, 1, 288, 288]

    def encode_prompt(self, prompt: dict, B: int, device, dtype) -> torch.Tensor:
        """Turn a frame-0 prompt into perceiver prompt tokens [B, P, 256].

        prompt keys (all optional, at least one expected):
          - 'mask'   : [B, 1, Hm, Wm] float in {0,1} (will be resized to PromptEncoder's
                       expected mask_input_size = 4*feat_size)
          - 'points' : [B, N, 2] in PIXEL coords on the padded image, + 'labels' [B, N]
          - 'box'    : [B, 4] xyxy in PIXEL coords on the padded image
        Returns prompt tokens = sparse (point/box) embeddings concatenated with the
        flattened dense (mask) embedding.
        """
        pe = self.prompt_encoder
        mask = prompt.get('mask') if prompt else None
        points = None
        if prompt and prompt.get('points') is not None:
            points = (prompt['points'].to(device=device, dtype=torch.float32),
                      prompt['labels'].to(device=device))
        box = prompt.get('box') if prompt else None
        if box is not None:
            box = box.to(device=device, dtype=torch.float32)

        if mask is not None:
            mask = mask.to(device=device, dtype=torch.float32)
            target = pe.mask_input_size  # (4*feat, 4*feat)
            if mask.shape[-2:] != target:
                mask = F.interpolate(mask, size=target, mode='bilinear', align_corners=False)

        # PromptEncoder runs in fp32 (random PE matrix); cast result to model dtype.
        with torch.autocast(device_type='cuda', enabled=False):
            sparse, dense = pe(points=points, boxes=box, masks=mask)
        # sparse: [B, Ns, C] ; dense: [B, C, feat, feat] -> flatten to tokens [B, feat*feat, C]
        dense_tokens = dense.flatten(2).transpose(1, 2)
        if sparse.shape[1] > 0:
            tokens = torch.cat([sparse, dense_tokens], dim=1)
        else:
            tokens = dense_tokens
        return tokens.to(dtype=dtype)

    def forward(
        self,
        frames: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        prompt: Optional[dict] = None,
        return_high_res: bool = False,
    ):
        """
        frames: [B, T, 3, H, W]  — T = MAX_FRAMES (= 8)
        attention_mask: [B, T] bool — currently unused (padding handled by loss)
        prompt: optional frame-0 prompt dict (mask/points/box) used to STEER the
                memory toward the selected object. Requires prompt_conditioning=True.
                When None, behaves object-agnostically (original behavior).
        return_high_res: if True, also returns the per-frame high-res FPN features
                needed by decode_masks (list of T [feat_s0, feat_s1] pairs).

        Returns per-frame memory-conditioned features [B, T, 256, H', W']
        (or (features, high_res_per_frame) when return_high_res=True).
        """
        B, T, C, H, W = frames.shape
        outs = []
        high_res_per_frame = [] if return_high_res else None

        prompt_tokens = None
        if self.prompt_conditioning and self.prompt_encoder is not None and prompt is not None:
            prompt_tokens = self.encode_prompt(prompt, B, frames.device, frames.dtype)

        latents = self.perceiver.init_memory(
            B, device=frames.device, dtype=frames.dtype, prompt_tokens=prompt_tokens)

        for t in range(T):
            if return_high_res:
                pix_feat, high_res = self.encode_frame(frames[:, t], want_high_res=True)
                high_res_per_frame.append(high_res)
            else:
                pix_feat = self.encode_frame(frames[:, t])      # [B, 256, H', W']
            tokens = pix_feat.flatten(2).transpose(1, 2)        # [B, H'W', 256]
            latents, mem_feat = self.perceiver(tokens, latents, frame_idx=t)
            outs.append(mem_feat)

        feats = torch.stack(outs, dim=1)  # [B, T, 256, H', W']
        if return_high_res:
            return feats, high_res_per_frame
        return feats


def build_student_model(config) -> StudentTemporalModel:
    prompt_conditioning = bool(getattr(config.MODEL, 'PROMPT_CONDITIONING', False))
    perceiver = TemporalPerceiver(
        dim=config.MODEL.PERCEIVER_DIM,
        depth=config.MODEL.PERCEIVER_DEPTH,
        num_latents=config.MODEL.PERCEIVER_LATENTS,
        num_heads=config.MODEL.PERCEIVER_HEADS,
        mlp_ratio=config.MODEL.PERCEIVER_MLP_RATIO,
        dropout=config.MODEL.PERCEIVER_DROPOUT,
        prompt_conditioning=prompt_conditioning,
    )
    model = StudentTemporalModel(
        checkpoint_path=config.MODEL.STUDENT_CKPT,
        perceiver=perceiver,
        backbone_type='repvit',
        model_name='m0.9' if 'm0_9' in config.MODEL.BACKBONE else 'm1.1',
        freeze_backbone=True,
        prompt_conditioning=prompt_conditioning,
        img_size=config.DATA.IMG_SIZE,
        feat_size=int(getattr(config.MODEL, 'FEAT_SIZE', 72)),
    )
    return model
