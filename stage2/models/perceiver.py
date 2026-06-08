"""TemporalPerceiver — fixed-latent cross-attention memory module.

Replaces SAM3's dense sequential memory bank (up to 7 past frames + object
pointers) with a fixed set of learnable latents that attend over the current
frame tokens. Same latents are carried across frames, giving the model
temporal memory while keeping shapes static for ONNX export.

Shapes (per frame):
    frame_tokens: [B, HW, C]          — e.g. C=256, HW=64*64=4096
    prev_latents: [B, L, C]           — L = num_latents (default 64)

Returned:
    new_latents:  [B, L, C]           — carried to next frame
    pix_feat:     [B, C, H, W]        — memory-conditioned features (distill target)
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


class _CrossAttnBlock(nn.Module):
    """Cross-attn(latents ← tokens) + self-attn(latents) + MLP."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.norm_q1 = nn.LayerNorm(dim)
        self.norm_kv1 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True,
        )

        self.norm2 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True,
        )

        hidden = int(dim * mlp_ratio)
        self.norm3 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, latents: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        q = self.norm_q1(latents)
        kv = self.norm_kv1(tokens)
        x, _ = self.cross_attn(q, kv, kv, need_weights=False)
        latents = latents + x

        s = self.norm2(latents)
        x, _ = self.self_attn(s, s, s, need_weights=False)
        latents = latents + x

        latents = latents + self.mlp(self.norm3(latents))
        return latents


class _PromptConditionBlock(nn.Module):
    """Seed the latents from a frame-0 prompt: cross-attn(latents <- prompt) + MLP.

    Makes the carried memory OBJECT-SELECTIVE — the same frames yield different
    latents (and thus different memory features / masks) for different prompts.
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True,
        )
        hidden = int(dim * mlp_ratio)
        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim),
        )

    def forward(self, latents: torch.Tensor, prompt_tokens: torch.Tensor) -> torch.Tensor:
        q = self.norm_q(latents)
        kv = self.norm_kv(prompt_tokens)
        x, _ = self.cross_attn(q, kv, kv, need_weights=False)
        latents = latents + x
        latents = latents + self.mlp(self.norm_mlp(latents))
        return latents


class TemporalPerceiver(nn.Module):
    def __init__(
        self,
        dim: int = 256,
        depth: int = 4,
        num_latents: int = 64,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        prompt_conditioning: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.num_latents = num_latents
        self.prompt_conditioning = prompt_conditioning

        # initial latents carried at t=0
        self.init_latents = nn.Parameter(torch.zeros(1, num_latents, dim))
        nn.init.trunc_normal_(self.init_latents, std=0.02)

        # optional prompt-conditioning: seed latents from a frame-0 prompt embedding
        # so the memory is steered toward the selected object (mask/point/box).
        self.prompt_block = (
            _PromptConditionBlock(dim, num_heads, mlp_ratio, dropout)
            if prompt_conditioning else None
        )

        # per-step temporal embedding added to latents (supports up to 8 frames)
        self.temporal_embed = nn.Parameter(torch.zeros(8, 1, dim))
        nn.init.trunc_normal_(self.temporal_embed, std=0.02)

        self.blocks = nn.ModuleList([
            _CrossAttnBlock(dim, num_heads, mlp_ratio, dropout) for _ in range(depth)
        ])

        # project latents back onto frame tokens to produce pix_feat_with_mem
        self.norm_q_out = nn.LayerNorm(dim)
        self.norm_kv_out = nn.LayerNorm(dim)
        self.out_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True,
        )
        self.norm_out = nn.LayerNorm(dim)
        self.mlp_out = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def init_memory(self, batch_size: int, device=None, dtype=None,
                    prompt_tokens: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Initial latents at t=0.

        If prompt_conditioning is on and prompt_tokens [B, P, C] are given, the
        latents are seeded toward the prompted object via cross-attention; this
        is what makes the model object-selective. Otherwise returns the learned
        constant latents (object-agnostic, original behavior).
        """
        x = self.init_latents.expand(batch_size, -1, -1).contiguous()
        if device is not None or dtype is not None:
            x = x.to(device=device, dtype=dtype)
        if self.prompt_block is not None and prompt_tokens is not None:
            x = self.prompt_block(x, prompt_tokens.to(dtype=x.dtype, device=x.device))
        return x

    def forward(
        self,
        frame_tokens: torch.Tensor,
        prev_latents: torch.Tensor,
        frame_idx: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        frame_tokens: [B, HW, C]
        prev_latents: [B, L, C]
        """
        B, HW, C = frame_tokens.shape
        H = W = int(math.isqrt(HW))
        assert H * W == HW, f"frame_tokens HW={HW} not square"

        tpe = self.temporal_embed[frame_idx % self.temporal_embed.shape[0]]
        latents = prev_latents + tpe

        for blk in self.blocks:
            latents = blk(latents, frame_tokens)

        q = self.norm_q_out(frame_tokens)
        kv = self.norm_kv_out(latents)
        delta, _ = self.out_attn(q, kv, kv, need_weights=False)
        tokens = frame_tokens + delta
        tokens = tokens + self.mlp_out(self.norm_out(tokens))

        pix_feat = tokens.transpose(1, 2).reshape(B, C, H, W).contiguous()
        return latents, pix_feat
