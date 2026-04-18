# --------------------------------------------------------
# Stage 2 Temporal Memory Distillation Configuration
# --------------------------------------------------------

import os
import yaml
from yacs.config import CfgNode as CN

MAX_FRAMES = 8  # FIXED — ONNX static shape requirement, never dynamic

_C = CN()
_C.BASE = ['']

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
_C.DATA.BATCH_SIZE = 2  # per-GPU, video memory-heavy
_C.DATA.DATA_PATH = ''
_C.DATA.DATASET = 'sav'
_C.DATA.SAV_TAR_DIR = '/mnt/hdd/datasets/SA-V/'
_C.DATA.SAV_TAR_RANGE = [0, 55]  # reserved for numbered splits; 60 hash-named tars globbed directly
_C.DATA.SAV_INDEX_CACHE = 'data/sav_index.pkl'
_C.DATA.MEAN = [123.675, 116.28, 103.53]
_C.DATA.STD = [58.395, 57.12, 57.375]
_C.DATA.IMG_SIZE = 1008  # SAM3 ViT-H RoPE freqs_cis baked at 1008; both models trained at this size
_C.DATA.MAX_FRAMES = MAX_FRAMES
_C.DATA.PIN_MEMORY = True
_C.DATA.NUM_WORKERS = 14
_C.DATA.PREFETCH_FACTOR = 4
_C.DATA.PERSISTENT_WORKERS = True
_C.DATA.DEBUG = False
_C.DATA.NUM_SAMPLES = -1

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.TYPE = 'efficient_sam3_temporal'
_C.MODEL.NAME = 'efficient_sam3_stage2'
_C.MODEL.BACKBONE = 'repvit_m0_9'
_C.MODEL.STUDENT_CKPT = ''           # Stage 1 checkpoint path
_C.MODEL.TEACHER_CKPT = ''           # SAM3 ViT-H tracker checkpoint path
_C.MODEL.PRETRAINED = ''
_C.MODEL.RESUME = ''

# Perceiver (replaces teacher dense memory bank)
_C.MODEL.PERCEIVER_DIM = 256
_C.MODEL.PERCEIVER_DEPTH = 4
_C.MODEL.PERCEIVER_LATENTS = 64
_C.MODEL.PERCEIVER_HEADS = 8
_C.MODEL.PERCEIVER_MLP_RATIO = 4.0
_C.MODEL.PERCEIVER_DROPOUT = 0.0

# -----------------------------------------------------------------------------
# Distillation settings
# -----------------------------------------------------------------------------
_C.DISTILL = CN()
_C.DISTILL.ENABLED = True
_C.DISTILL.MSE_WEIGHT = 1.0
_C.DISTILL.COSINE_WEIGHT = 1.0
_C.DISTILL.FEATURE_DIM = 256         # teacher pix_feat_with_mem channel dim

# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.EPOCHS = 100
_C.TRAIN.WARMUP_EPOCHS = 5
_C.TRAIN.WEIGHT_DECAY = 0.05
_C.TRAIN.BASE_LR = 1e-4
_C.TRAIN.WARMUP_LR = 1e-7
_C.TRAIN.MIN_LR = 1e-6
_C.TRAIN.CLIP_GRAD = 1.0
_C.TRAIN.AUTO_RESUME = True
_C.TRAIN.ACCUMULATION_STEPS = 1
_C.TRAIN.USE_CHECKPOINT = False
_C.TRAIN.AMP_DTYPE = 'bfloat16'      # bf16 on Ampere+
_C.TRAIN.FIND_UNUSED_PARAMETERS = False

_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = 'cosine'
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 30
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1

_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'adamw'
_C.TRAIN.OPTIMIZER.EPS = 1e-8
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9

# -----------------------------------------------------------------------------
# Eval
# -----------------------------------------------------------------------------
_C.EVAL = CN()
_C.EVAL.DAVIS_ROOT = ''
_C.EVAL.EVERY_N_EPOCHS = 5

# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
_C.AMP_ENABLE = True
_C.OUTPUT = ''
_C.TAG = 'default'
_C.SAVE_FREQ = 1
_C.PRINT_FREQ = 10
_C.SEED = 0
_C.EVAL_MODE = False
_C.THROUGHPUT_MODE = False
_C.LOCAL_RANK = 0


def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
            config.defrost()
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):
    if getattr(args, 'cfg', None):
        _update_config_from_file(config, args.cfg)

    config.defrost()
    if getattr(args, 'opts', None):
        config.merge_from_list(args.opts)

    if getattr(args, 'batch_size', None):
        config.DATA.BATCH_SIZE = args.batch_size
    if getattr(args, 'data_path', None):
        config.DATA.DATA_PATH = args.data_path
        config.DATA.SAV_TAR_DIR = args.data_path
    if getattr(args, 'student_ckpt', None):
        config.MODEL.STUDENT_CKPT = args.student_ckpt
    if getattr(args, 'teacher_ckpt', None):
        config.MODEL.TEACHER_CKPT = args.teacher_ckpt
    if getattr(args, 'resume', None):
        config.MODEL.RESUME = args.resume
    if getattr(args, 'accumulation_steps', None):
        config.TRAIN.ACCUMULATION_STEPS = args.accumulation_steps
    if getattr(args, 'use_checkpoint', False):
        config.TRAIN.USE_CHECKPOINT = True
    if getattr(args, 'disable_amp', False) or getattr(args, 'only_cpu', False):
        config.AMP_ENABLE = False
    if getattr(args, 'output', None):
        config.OUTPUT = args.output
    if getattr(args, 'tag', None):
        config.TAG = args.tag
    if getattr(args, 'eval', False):
        config.EVAL_MODE = True
    if getattr(args, 'throughput', False):
        config.THROUGHPUT_MODE = True

    if getattr(args, 'local_rank', None) is None and 'LOCAL_RANK' in os.environ:
        args.local_rank = int(os.environ['LOCAL_RANK'])
    config.LOCAL_RANK = args.local_rank if getattr(args, 'local_rank', None) is not None else 0

    # sanity: MAX_FRAMES is a compile-time constant, refuse to change it
    assert config.DATA.MAX_FRAMES == MAX_FRAMES, \
        f"MAX_FRAMES is fixed at {MAX_FRAMES} for ONNX static shape; got {config.DATA.MAX_FRAMES}"

    config.freeze()


def get_config(args=None):
    config = _C.clone()
    if args is not None:
        update_config(config, args)
    return config
