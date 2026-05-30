# Stage 2 best checkpoint (ep10)

Snapshot copied from `output/efficient_sam3_stage2/ep50_run4/ckpt_best.pth` while
ep50_run4 was still training.

## Contents

| file | bytes | what |
|---|---|---|
| `ckpt_best_ep10_val0.2920.pth` | ~77 MB | full payload: perceiver weights + backbone BN buffers + optimizer + EMA + config |
| `sav_repvit_m0_9.yaml` | <1 KB | training config used to produce this checkpoint |

## Metadata

- **epoch_done**: 10
- **global_step**: 272,767
- **best_val (total)**: 0.292039
- **EMA shadow included**: yes
- **trainable keys** (perceiver only): 96
- **backbone BN buffer keys**: 321
- **autocast cache_enabled=False** fix was active when this checkpoint was produced

## Load for inference

The checkpoint format is the one used by `stage2/utils/checkpoint.py:load_checkpoint`.
You also need the Stage 1 frozen backbone checkpoint at
`checkpoints/efficient_sam3_repvit_s.pt` and the SAM3 teacher ckpt at
`/mnt/hdd/checkpoints/sam3/sam3.pt` (the latter only if you want to compare to teacher).

```python
from stage2.config import get_config
from stage2.models import build_student_model, ModelEma
from stage2.utils import load_checkpoint
import torch

class _Args:
    cfg = 'best_checkpoint/sav_repvit_m0_9.yaml'
    opts = None
    batch_size = data_path = student_ckpt = teacher_ckpt = resume = output = tag = None
    accumulation_steps = None
    use_checkpoint = disable_amp = only_cpu = eval = throughput = False
    local_rank = 0

config = get_config(_Args)
device = torch.device('cuda')
model = build_student_model(config).to(device)

# build EMA placeholder so load_checkpoint also restores the EMA shadow
ema = ModelEma(model.perceiver, decay=config.TRAIN.EMA_DECAY, device=device)

meta = load_checkpoint(
    config=config,
    path='best_checkpoint/ckpt_best_ep10_val0.2920.pth',
    model=model,
    ema=ema,
    map_location=device,
)
model.eval()

# Optional: swap in EMA shadow (usually 0.001-0.003 lower val than live)
# ema.copy_to(model.perceiver)

# IMPORTANT: forward must use cache_enabled=False or you'll get stale-cache
# garbage output (see HANDOFF.md, the autocast-cache bug):
with torch.no_grad():
    with torch.amp.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
        s_out = model(frames, attn_mask)  # [B, T, 256, H', W']
```

## Notes on use

- Loading uses `strict=False` because the backbone weights live in
  `checkpoints/efficient_sam3_repvit_s.pt` (Stage 1, frozen) and are loaded by
  `build_student_model` at construction, then this ckpt overlays trainable
  perceiver weights + BN buffers.
- EMA shadow typically scores 0.0015–0.0023 better on val than the live weights;
  swap with `ema.copy_to(model.perceiver)` after load if you want the smoothed version.
