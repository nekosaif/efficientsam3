"""Teacher wrapper: frozen SAM3 ViT-H video model.

Per-frame distillation target: tracker `pix_feat_with_mem`.

Two paths:
  * `forward(frames)`            → first-frame path for every t (legacy Step 2)
  * `forward(frames, masks=...)` → full sequential memory-bank path that
    consumes GT masklets to populate prior-frame `maskmem_features` /
    `obj_ptr` (Step 3, #7).

Teacher is frozen + eval mode always; entire forward runs under no_grad.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import torch
import torch.nn as nn

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
_SAM3_PKG = os.path.join(_PROJECT_ROOT, 'sam3')
if _SAM3_PKG not in sys.path:
    sys.path.insert(0, _SAM3_PKG)

from sam3.model_builder import build_sam3_video_model  # noqa: E402

from stage2.config import MAX_FRAMES  # noqa: E402


class TeacherModel(nn.Module):
    def __init__(self, checkpoint_path: str):
        super().__init__()

        base = build_sam3_video_model(
            checkpoint_path=checkpoint_path,
            load_from_HF=False,
            strict_state_dict_loading=False,
        )
        self.vision_backbone = base.detector.backbone
        self.tracker = base.tracker
        del base

        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):  # teacher stays in eval mode
        return super().train(False)

    # -------------------------------------------------------------------- utils

    @torch.no_grad()
    def encode_frame(self, img: torch.Tensor) -> torch.Tensor:
        """Per-frame top-level pix_feat [B,256,H,W] after tracker FPN projection."""
        out = self.vision_backbone.forward_image(img)
        sam2_out = out['sam2_backbone_out']
        return sam2_out['backbone_fpn'][-1]

    @torch.no_grad()
    def pix_feat_with_mem_first_frame(self, img: torch.Tensor) -> torch.Tensor:
        """First-frame pix_feat_with_mem = pix_feat + no_mem_embed."""
        pix_feat = self.encode_frame(img)
        B, C, H, W = pix_feat.shape
        no_mem = self.tracker.no_mem_embed                # [1,1,256]
        flat = pix_feat.flatten(2).transpose(1, 2)        # [B,HW,256]
        flat = flat + no_mem
        return flat.transpose(1, 2).reshape(B, C, H, W).contiguous()

    # ------------------------------------------------------------ memory path

    @torch.no_grad()
    def _backbone_per_frame(self, frames: torch.Tensor):
        """Run SAM3 backbone once over [B*T,3,H,W]; split per frame.

        Mirrors `tracker.forward_image` but calls vision_backbone directly:
        in the composed video model `tracker.backbone` is None (image encoder
        lives on detector). Then pre-applies sam_mask_decoder.conv_s0/conv_s1
        to backbone_fpn[0:2] exactly like tracker.forward_image does.
        """
        B, T = frames.shape[:2]
        flat = frames.reshape(B * T, *frames.shape[2:])
        bo = self.vision_backbone.forward_image(flat)["sam2_backbone_out"]
        bo["backbone_fpn"][0] = self.tracker.sam_mask_decoder.conv_s0(bo["backbone_fpn"][0])
        bo["backbone_fpn"][1] = self.tracker.sam_mask_decoder.conv_s1(bo["backbone_fpn"][1])
        per_frame = []
        for t in range(T):
            sl = slice(t * B, (t + 1) * B)
            bo_t = {
                "backbone_fpn": [x[sl] for x in bo["backbone_fpn"]],
                "vision_pos_enc": [x[sl] for x in bo["vision_pos_enc"]],
            }
            _, vf, vpe, fs = self.tracker._prepare_backbone_features(bo_t)
            per_frame.append((vf, vpe, fs))
        return per_frame

    @torch.no_grad()
    def _forward_with_memory(
        self,
        frames: torch.Tensor,        # [B,T,3,H,W]
        masks: torch.Tensor,         # [B,T,H,W] binary float
    ) -> torch.Tensor:
        B, T = frames.shape[:2]
        per_frame = self._backbone_per_frame(frames)

        output_dict = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
        targets = []  # pix_feat_with_mem per frame

        for t in range(T):
            vf, vpe, fs = per_frame[t]
            img_t = frames[:, t]
            is_init = (t == 0)

            # high-res FPN features for SAM head (already conv_s0/conv_s1 projected)
            if len(vf) > 1:
                high_res_features = [
                    x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
                    for x, s in zip(vf[:-1], fs[:-1])
                ]
            else:
                high_res_features = None

            # (a) memory-conditioned features → DISTILLATION TARGET
            pix_feat_with_mem = self.tracker._prepare_memory_conditioned_features(
                frame_idx=t,
                is_init_cond_frame=is_init,
                current_vision_feats=vf[-1:],
                current_vision_pos_embeds=vpe[-1:],
                feat_sizes=fs[-1:],
                output_dict=output_dict,
                num_frames=T,
            )
            targets.append(pix_feat_with_mem)

            # (b) derive obj_ptr / object_score_logits from GT mask
            gt_mask_t = masks[:, t:t + 1]  # [B,1,H,W]
            sam_outs = self.tracker._use_mask_as_output(
                backbone_features=pix_feat_with_mem,
                high_res_features=high_res_features,
                mask_inputs=gt_mask_t,
            )
            _, high_res_masks, _, _, _, obj_ptr, object_score_logits = sam_outs

            # (c) encode new memory for future frames
            maskmem_features, maskmem_pos_enc = self.tracker._encode_new_memory(
                image=img_t,
                current_vision_feats=vf,
                feat_sizes=fs,
                pred_masks_high_res=high_res_masks,
                object_score_logits=object_score_logits,
                is_mask_from_pts=False,
                output_dict=output_dict,
                is_init_cond_frame=is_init,
            )

            out_t = {
                "maskmem_features": maskmem_features,
                "maskmem_pos_enc": maskmem_pos_enc,
                "obj_ptr": obj_ptr,
                # frame_filter() needs eff_iou_score on non-cond entries when
                # use_memory_selection=True. GT mask → trust it.
                "eff_iou_score": torch.ones((), device=frames.device, dtype=torch.float32),
            }
            if is_init:
                output_dict["cond_frame_outputs"][t] = out_t
            else:
                output_dict["non_cond_frame_outputs"][t] = out_t

        return torch.stack(targets, dim=1)  # [B,T,256,H',W']

    # ------------------------------------------------------------------ entry

    @torch.no_grad()
    def forward(
        self,
        frames: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
        mask_valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        frames:         [B,T,3,H,W]
        attention_mask: [B,T] (currently unused; SA-V loader picks valid frames)
        masks:          [B,T,H,W] binary GT masklet (required for memory path)
        mask_valid:     [B,T] bool (currently must be all True)
        Returns:        [B,T,256,H',W']
        """
        B, T = frames.shape[:2]
        assert T == MAX_FRAMES

        if masks is None:
            return torch.stack(
                [self.pix_feat_with_mem_first_frame(frames[:, t]) for t in range(T)],
                dim=1,
            )

        # mask_valid==False just means the object isn't present in that annotated
        # frame — _use_mask_as_output / _encode_new_memory handle the "no-obj"
        # branch via is_obj_appearing on the GT mask itself, so we can pass through.
        del mask_valid
        return self._forward_with_memory(frames, masks)


def build_teacher_model(config) -> TeacherModel:
    return TeacherModel(checkpoint_path=config.MODEL.TEACHER_CKPT)
