# Archived: Stage-2 ep50_run4 (object-agnostic) — pre-Milestone-1 backup

Backed up 2026-06-04 before starting the mask-prompted (prompt-conditioned)
redesign (Milestone 1 in plan lexical-seeking-spring.md).

These are the ORIGINAL Stage-2 weights: object-AGNOSTIC TemporalPerceiver trained
with the loader hardcoding obj_idx=0 and no prompt input. forward(frames) -> 
[B,8,256,72,72] features (not masks), NOT steerable to a chosen object.

- ckpt_best.pth: best object-agnostic checkpoint (epoch ~22, val ~0.2896).
- ckpt_epoch_*.pth: per-epoch.
- ckpt_running.ep27it18000.bak: mid-epoch running save preserved earlier.
- A separate curated snapshot also exists at repo best_checkpoint/ckpt_best_ep10_val0.2920.pth.

Kept as: (a) safety, (b) possible warm-start to A/B vs random init for the
prompt-conditioned Perceiver (review judged it NOT a reliable init, so test before trusting).
