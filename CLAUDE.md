# CLAUDE.md

This file provides guidance to AI assistants when working with code in this repository.

我的问题中，回答尽量使用中文回答，这是一条全局上下文都试用的提示。

以第一性原理！从原始需求和问题本质出发，不从惯例或模板出发。 

1.不要假设我清楚自己想要什么。动机或目标不清晰时，停下来讨论。 

2.目标清晰但路径不是最短的，直接告诉我并建议更好的办法。 

3.遇到问题追根因，不打补丁。每个决策都要能回答"为什么"。 

4.输出说重点，砍掉一切不改变决策的信息。

## Project Overview

**MemISTD** is a research project for **Infrared Small Target Detection (IRSTD)** using memory-augmented deep learning. The core model is `MemISTDSmallTarget` (located in `model_2/`), a UNet-based architecture with a **Dual Memory System** for target/background separation.

### Research Innovation

- **Target Memory Module**: Learns prototypical small target features (Gaussian-like bright spots) via patch-based memory reading (`TargetBranch`)
- **Background Memory Module**: Learns typical background texture patterns (cloud edges, sea clutter) via patch-based memory reading (`BackgroundBranch`)
- **Multi-scale Memory**: Each memory branch operates at multiple patch sizes [3, 5, 7] for multi-scale context
- **Residual Reconstruction Loss**: Physically-constrained decomposition: `original_image ≈ rec_bg + rec_tgt`
- **ArithmeticFusion**: `spatial_reweight` strategy — `Global * (1 + target_weight) * (1 - bg_weight_gated)` for adaptive suppression

---

## Architecture Overview (`model_2/`)

### Main Model: `MemISTDSmallTarget` (`model_2/memistd_small_target.py`)

```
Input Image [B, 1, H, W]   (single-channel grayscale infrared)
    |
[UNetBackboneSmallTarget]  depth=3, base_channels=32
    -> encoder E0 [B, 32, H, W]
    -> encoder E1 [B, 64, H/2, W/2]
    -> encoder E2 [B, 128, H/4, W/4]
    -> decoder with skip connections
    -> TinyTargetAttention
    -> backbone_feat [B, 128, H, W]
    |
[FeatureSplitBranch]       in=128, split to 3x mem_channels=64
    -> global_feat     [B, 64, H, W]  (global projection)
    -> target_feat     [B, 64, H, W]  (target projection + ChannelAttention)
    -> background_feat [B, 64, H, W]  (bg projection + ChannelAttention)
    |
[DualMemorySystem]         multi-scale patch memory
    ├── bg_branches  [patch_sizes=3,5,7] x BackgroundBranch  -> bg_fusion  -> background_feat_memory  [B, 64, H, W]
    └── tg_branches  [patch_sizes=3,5,7] x TargetBranch      -> tg_fusion  -> target_feat_memory     [B, 64, H, W]
    |
[MemoryDecoder x2]         (only when decode_image=True, i.e. during training)
    -> target_recon_img     [B, 1, H, W]
    -> background_recon_img [B, 1, H, W]
    |
[ArithmeticFusion]         strategy='spatial_reweight'
    -> fused_feat [B, 64, H, W]
    |
[CoordAtt (neck)]          CoordinateAttention-based lightweight FPN
    -> P0 features [B, 64, H, W]
    |
[SingleScaleDetectionHead] SmallTargetHead (YOLOX-style, P0 only, stride=1)
    -> P0_reg [B, 4, H, W]
    -> P0_obj [B, 1, H, W]
    -> P0_cls [B, 1, H, W]
```

### Key Modules

| Module | File | Description |
|--------|------|-------------|
| `MemISTDSmallTarget` | `model_2/memistd_small_target.py` | Main model |
| `UNetBackboneSmallTarget` | `model_2/memistd_small_target.py` | 3-level UNet encoder-decoder |
| `FeatureSplitBranch` | `model_2/memistd_small_target.py` | Splits backbone output into global/target/bg streams |
| `DualMemorySystem` | `model_2/memistd_small_target.py` | Multi-scale memory (BackgroundBranch + TargetBranch) |
| `ArithmeticFusion` | `model_2/memistd_small_target.py` | Spatial-reweight fusion of global+bg+target features |
| `CoordAtt` | `model_2/memistd_small_target.py` | Coordinate Attention FPN (neck) |
| `TinyTargetAttention` | `model_2/tiny_target_modules.py` | Small target attention enhancement |
| `SmallTargetHead` | `model_2/tiny_target_modules.py` | YOLOX-style detection head |
| `FeatureRefinementModule` | `model_2/tiny_target_modules.py` | Feature refinement in FPN |
| `ResidualReconstructionLoss` | `model_2/losses.py` | Memory reconstruction loss |
| `YOLOLossOptimized` | `model_2/yolox_loss_optimized.py` | YOLO detection loss with SimOTA |
| `FocalLoss`, `NWDLoss` | `model_2/losses.py` | Small target specific losses |

### Memory Module Detail

```python
# _BaseMemoryBranch: patch-based memory read
# similarity_type = 'dot' (scaled dot product)
# memory = nn.Parameter(randn(num_memories, C * patch_size^2))

# BackgroundBranch: fold-based reconstruction
# out = fold(memory_read_out) / divisor

# TargetBranch: same fold-based reconstruction (same as BackgroundBranch)
# (center-pixel extraction was disabled)

# DualMemorySystem:
# bg_branches: [BackgroundBranch(64mem,ps=3), BackgroundBranch(64mem,ps=5), BackgroundBranch(64mem,ps=7)]
# tg_branches: [TargetBranch(16mem,ps=3),     TargetBranch(16mem,ps=5),     TargetBranch(16mem,ps=7)]
# bg_fusion / tg_fusion: MultiScaleFusion(type='attention')
```

### ArithmeticFusion (`spatial_reweight` strategy)

```python
target_weight     = sigmoid(target_proj(f_tg))        # [B, C, H, W]
background_weight = sigmoid(background_proj(f_bc))    # [B, C, H, W]

# adaptive gate: suppress bg_weight where target_weight is high
gate_factor = exp(-3.0 * target_weight)               # k=3.0
bg_weight_gated = background_weight * gate_factor

out = f_global * (1 + target_weight) * (1 - bg_weight_gated)
out = SpatialAttention(out)                           # kernel_size=7
```

### Loss Function Design (`model_2/losses.py`)

```
total_loss = yolo_loss + residual_recon_weight * residual_recon_loss

residual_recon_loss = (
    loss_global              # L1(rec_bg + rec_tgt, img)
  + alpha * loss_tgt_sparse  # L2((1-mask) * rec_tgt)      [default alpha=0.1]
  + beta  * loss_tgt_content # MSE(mask * rec_tgt, mask * img) [default beta=2.0]
  + gamma * loss_bg_inpaint  # Background inpainting in target region [default gamma=0.5]
)
```

Current config (`config/memistd_small_target_config_v2.yaml`):
```yaml
residual_recon_loss:
  alpha: 0.1   # sparse constraint (L2, prevents over-suppression)
  beta: 2.0    # target content constraint
  gamma: 0.5   # background inpainting
```

---

## Dataset: IRDST

### Dataset Structure

```
dataset/IRDST/IRDST_real/
├── images/
│   ├── train/   # Grayscale infrared images (.png)
│   └── test/
├── masks/
│   ├── train/   # Binary segmentation masks (.png)
│   └── test/
├── boxes/
│   ├── train/   # Bounding box annotations (.txt, one per image)
│   └── test/
└── center/
    ├── train/   # Target center coordinates
    └── test/
```

### Dataloader (`MemISTD_Dataloader.py`)

- Class: `IRDSTDataset`
- Collate function: `irdst_collate_fn`
- Input: **single-channel (grayscale)** infrared images → `[B, 1, H, W]`
- Normalization: IRDST: `(pixel/255 + 0.1246) / 1.0923` | ITSDT: `(pixel/255 - 0.4874) / 0.2457`
- Output per sample:
  - `images`: `[B, 1, H, W]` float tensor
  - `targets_dict["boxes"]`: list of `[N, 4]` tensors (x, y, w, h format)
  - `targets_dict["labels"]`: list of `[N]` label tensors
  - `targets_dict["target_mask"]`: `[B, 1, H, W]` binary mask

### Training / Eval Image Sizes (from config)

```yaml
data:
  input_shape: [480, 720]       # training crop size
  pad_to_size: [240, 240]       # training pad size
  eval_pad_to_size: [480, 720]  # evaluation pad size
  pad_only: true
  pad_divisor: 4
```

---

## Training

### Distributed Training (Primary) — `train_dist_2.py`

```bash
# 4-GPU distributed training (recommended)
torchrun --nproc_per_node=4 train_dist_2.py --config config/memistd_small_target_config_v2.yaml

# Single-GPU (debugging)
python train_dist_2.py --config config/memistd_small_target_config_v2.yaml

# Resume from checkpoint
torchrun --nproc_per_node=4 train_dist_2.py \
  --config config/memistd_small_target_config_v2.yaml \
  --resume checkpoint/<timestamp>
```

### Single-GPU Training — `MemISTD_Trainer_SmallTarget_2.py`

```bash
python MemISTD_Trainer_SmallTarget_2.py --config config/memistd_small_target_config_v2.yaml
```

### Key Training Config Parameters (`config/memistd_small_target_config_v2.yaml`)

```yaml
model:
  in_channels: 1
  num_classes: 1
  base_channels: 32             # -> backbone channels: [32, 64, 128]
  backbone_depth: 3
  target_memory_slots: 16
  background_memory_slots: 64
  ms_fusion_type: attention
  global_fusion_strategy: spatial_reweight
  bg_patch_sizes: [3, 5, 7]
  tg_patch_sizes: [3, 5, 7]

training:
  batch_size: 1
  accumulation_steps: 1
  epochs: 200
  lr: 0.001
  weight_decay: 0.01
  fp16: false
  save_dir: ./logs/small_target_v2
  save_period: 1
  save_every_epoch: true

distributed:
  enabled: true
  backend: nccl
  world_size: 4

yolox_loss:
  strides: [1]                  # single scale, stride=1 (full resolution)
  center_radius: 5.0
  use_focal: true
  focal_alpha: 0.25
  focal_gamma: 3.0
  adaptive_center_radius: true

eval:
  eval_flag: true
  eval_period: 10
  conf_thresh: 0.13
  iou_thresh: 0.2
  nms_thresh: 0.2
```

### Checkpoints

Checkpoints are saved to `checkpoint/<timestamp>/`:
- `last.pth`: Latest checkpoint (for resume)
- `best.pth`: Best checkpoint by mAP
- `config.yaml`: Config snapshot

---

## Evaluation

### Evaluation Script — `MemISTD_Evaluate_SmallTarget 2.py`

```bash
python "MemISTD_Evaluate_SmallTarget 2.py" \
  --weights checkpoint/<timestamp>/best.pth \
  --config config/memistd_small_target_config_v2.yaml

# With visualization
python "MemISTD_Evaluate_SmallTarget 2.py" \
  --weights checkpoint/<timestamp>/best.pth \
  --config config/memistd_small_target_config_v2.yaml \
  --save-vis --vis-dir ./outputs/
```

### Evaluation Metrics

- **Recall (R)**: Target detection rate
- **Precision (P)**: False alarm suppression
- **F1 Score**: Harmonic mean of R and P
- **AP@0.5 (mAP)**: Primary metric, area under PR curve at IoU=0.5
- **FPS**: Inference speed

Evaluation uses `utils/eval_utils.py::evaluate_detection()` which returns:
```python
{
    'recall': float,
    'precision': float,
    'f1': float,
    'ap50': float,        # Primary metric
    'fps': float,
    'total_gt': int,
    'tp': int, 'fp': int, 'fn': int,
}
```

### Inline Evaluation (during training)

Evaluation runs automatically every `eval_period` epochs via `evaluate_detection()` in `train_dist_2.py` and `MemISTD_Trainer_SmallTarget_2.py`, using `utils/eval_utils.py`.

---

## File Structure

```
MemISTD/
├── model_2/                          # Current active model (USE THIS)
│   ├── memistd_small_target.py       # Main network (MemISTDSmallTarget)
│   ├── tiny_target_modules.py        # TinyTargetAttention, SmallTargetHead
│   ├── losses.py                     # ResidualReconstructionLoss, FocalLoss, NWDLoss
│   ├── yolox_loss_optimized.py       # YOLO loss with SimOTA
│   └── __init__.py
├── config/
│   └── memistd_small_target_config_v2.yaml  # Active training config
├── utils/
│   ├── eval_utils.py                 # evaluate_detection(), print_evaluation_report()
│   ├── callbacks.py                  # LossHistory, TensorBoard writer
│   ├── utils_bbox.py                 # Bounding box decode, NMS
│   └── utils.py
├── dataset/IRDST/IRDST_real/         # Dataset (see structure above)
├── checkpoint/                       # Saved checkpoints (<timestamp>/last.pth, best.pth)
├── logs/                             # TensorBoard logs
├── docs/
│   ├── drafts/                       # AI-generated analysis docs (YYYY-MM-DD_<topic>.md)
│   └── plans/                        # Implementation plans
├── scripts/
│   └── cursor_temp/                  # Temporary test/verify scripts
├── 文档/                             # Human-curated Chinese technical docs
├── train_dist_2.py                   # Distributed training entry point (PRIMARY)
├── MemISTD_Trainer_SmallTarget_2.py  # Single-GPU training (PRIMARY)
├── MemISTD_Dataloader.py             # IRDSTDataset, irdst_collate_fn
├── "MemISTD_Evaluate_SmallTarget 2.py"  # Evaluation script (PRIMARY)
├── model_data/classes.txt            # Class names: ['target']
└── requirements.txt

# Legacy / do NOT use for new development:
model/       # older version
model_1/     # older version
TDCNet/      # Original TDCNet (temporal 3D CNN — NOT this project)
train_dist.py / train_dist_1.py            # older training scripts
MemISTD_Trainer_SmallTarget.py             # older trainer
MemISTD_Trainer_SmallTarget_1.py           # older trainer
"MemISTD_Evaluate_SmallTarget.py"          # older eval
"MemISTD_Evaluate_SmallTarget 1.py"        # older eval
```

---

## Development Notes

### IMPORTANT: Active Code Locations

- **Model**: `model_2/` — all model code lives here
- **Primary training**: `train_dist_2.py` (multi-GPU) or `MemISTD_Trainer_SmallTarget_2.py` (single-GPU)
- **Primary evaluation**: `MemISTD_Evaluate_SmallTarget 2.py`
- **Active config**: `config/memistd_small_target_config_v2.yaml`
- **DO NOT** reference `TDCNet/`, `model/`, `model_1/`, or `train_dist.py` / `train_dist_1.py`

### Model Forward Pass API

```python
# Training forward pass (with reconstruction)
outputs = model(images, decode_image=True)
# outputs keys:
#   'predictions'            : {'P0_reg': [B,4,H,W], 'P0_obj': [B,1,H,W], 'P0_cls': [B,1,H,W]}
#   'background_recon_img'   : [B, 1, H, W]
#   'target_recon_img'       : [B, 1, H, W]
#   'original_img'           : [B, 1, H, W]  (copy of input)
#   'target_feat_recon'      : [B, 64, H, W] (target memory feature)
#   'background_feat_recon'  : [B, 64, H, W] (bg memory feature)

# Inference forward pass (no reconstruction)
outputs = model(images, decode_image=False)

# Loss computation
# NOTE: must set outputs['target_mask'] = target_mask before calling compute_loss
outputs['target_mask'] = target_mask   # [B, 1, H, W] from dataloader
loss_dict = model.module.compute_loss(
    outputs=outputs,
    labels=yolo_targets,          # list of [N, 5] tensors (cx, cy, w, h, cls)
    yolox_loss=yolo_loss,
    residual_recon_loss=residual_recon_loss,
    residual_recon_weight=residual_recon_weight,
)
# loss_dict keys: total_loss, yolo_loss, loss_box, loss_obj, loss_cls,
#                 residual_recon_loss, loss_global, loss_tgt_sparse,
#                 loss_tgt_content, loss_bg_inpaint, num_fg
```

### Training Loss Interpretation (tqdm bar)

```
total=0.424   # Total loss (should decrease)
yolo=0.347    # YOLO detection loss
box=0.235     # Bounding box regression
obj=0.112     # Objectness
cls=0         # Classification (expected 0 for single-class)
res_recon=0.077  # Residual reconstruction loss
global=0.018     # Global reconstruction L1
tgt_sparse=0.001 # Target sparsity (L2, very small = good)
tgt_content=0.02 # Target content MSE (grows as model learns targets)
bg_inpaint=0.037 # Background inpainting
```

### TensorBoard Monitoring

```bash
tensorboard --logdir ./logs --port 6006
```

Metrics logged: `total_loss`, `yolo_loss`, `loss_box`, `loss_obj`, `loss_cls`,
`residual_recon_loss`, `loss_global`, `loss_tgt_sparse`, `loss_tgt_content`,
`loss_bg_inpaint`, `mAP`

### File Naming Convention

AI-generated documents must follow `.cursor/rules/file-organization.mdc`:
- Analysis docs → `docs/drafts/YYYY-MM-DD_<topic>.md`
- Plans → `docs/plans/YYYY-MM-DD_<feature>_plan.md`
- Temp scripts → `scripts/cursor_temp/test_<desc>.py` or `verify_<desc>.py` 