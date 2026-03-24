#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MemISTD Small Target Detection — Standalone Evaluation Script (Primary)

Usage:
    python "MemISTD_Evaluate_SmallTarget 2.py" \\
        --weights checkpoint/<timestamp>/best.pth \\
        --config  config/memistd_small_target_config_v2.yaml

    # With visualization output
    python "MemISTD_Evaluate_SmallTarget 2.py" \\
        --weights checkpoint/<timestamp>/best.pth \\
        --config  config/memistd_small_target_config_v2.yaml \\
        --save-vis --vis-dir ./outputs/vis

    # Override thresholds
    python "MemISTD_Evaluate_SmallTarget 2.py" \\
        --weights checkpoint/<timestamp>/best.pth \\
        --config  config/memistd_small_target_config_v2.yaml \\
        --conf-thresh 0.13 --iou-thresh 0.2 --nms-thresh 0.2

Metrics reported:
    Recall (R), Precision (P), F1, AP@0.5 (primary), FPS
"""

import os
import sys
import argparse
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
import yaml
from pathlib import Path

# ── project root ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_2.memistd_small_target import MemISTDSmallTarget
from MemISTD_Dataloader import IRDSTDataset, ITSDTDataset, irdst_collate_fn
from utils.eval_utils import evaluate_detection, print_evaluation_report


# =============================================================================
#  Config
# =============================================================================

def load_config(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


# =============================================================================
#  Model construction  (mirrors train_dist_2.py exactly)
# =============================================================================

def build_model(cfg: dict) -> MemISTDSmallTarget:
    """Instantiate MemISTDSmallTarget from config (inference-only build)."""
    mc = cfg.get('model', {})
    model = MemISTDSmallTarget(
        in_channels=mc.get('in_channels', 1),
        num_classes=mc.get('num_classes', 1),
        base_channels=mc.get('base_channels', 32),
        backbone_depth=mc.get('backbone_depth', 3),
        target_memory_slots=mc.get('target_memory_slots', 16),
        background_memory_slots=mc.get('background_memory_slots', 64),
        ms_fusion_type=mc.get('ms_fusion_type', 'attention'),
        global_fusion_strategy=mc.get('global_fusion_strategy', 'spatial_reweight'),
        bg_patch_sizes=mc.get('bg_patch_sizes', [3, 5, 7]),
        tg_patch_sizes=mc.get('tg_patch_sizes', [3, 5, 7]),
    )
    return model


def load_checkpoint(model: nn.Module, weights_path: str, device: torch.device) -> nn.Module:
    """
    Load checkpoint weights into model.
    Handles DDP-saved state dicts (strips 'module.' prefix automatically).
    """
    print(f'[INFO] Loading weights: {weights_path}')
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)

    # Resolve state-dict from various checkpoint formats
    state_dict = None
    if isinstance(ckpt, dict):
        for key in ('model_state_dict', 'model', 'state_dict'):
            if key in ckpt and isinstance(ckpt[key], dict):
                state_dict = ckpt[key]
                break
        if state_dict is None:
            state_dict = ckpt   # assume the whole file is the state-dict
    else:
        raise ValueError(f'Unexpected checkpoint type: {type(ckpt)}')

    # Strip DDP 'module.' prefix if present
    cleaned = {
        k[len('module.'):] if k.startswith('module.') else k: v
        for k, v in state_dict.items()
    }

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f'  [WARN] {len(missing)} missing keys: '
              f'{missing[:5]}{" ..." if len(missing) > 5 else ""}')
    if unexpected:
        print(f'  [WARN] {len(unexpected)} unexpected keys: '
              f'{unexpected[:5]}{" ..." if len(unexpected) > 5 else ""}')

    if isinstance(ckpt, dict):
        epoch   = ckpt.get('epoch',    'N/A')
        best_ap = ckpt.get('mAP',      ckpt.get('best_map', 'N/A'))
        if isinstance(best_ap, float):
            best_ap = f'{best_ap * 100:.2f}%'
        print(f'  Checkpoint info — epoch={epoch}  saved_mAP={best_ap}')

    return model


# =============================================================================
#  Dataset  (mirrors test_dataset construction in train_dist_2.py exactly)
# =============================================================================

def build_eval_dataset(cfg: dict, split: str = 'test'):
    """
    Build dataset for evaluation (no augmentation).
    Supports dataset_type: "irdst" (default) | "itsdt"
    """
    dc           = cfg.get('data', {})
    dataset_root = dc.get('dataset_root', 'dataset/IRDST/IRDST_real')
    input_shape  = dc.get('input_shape', [480, 720])
    eval_pad     = dc.get('eval_pad_to_size', dc.get('pad_to_size', input_shape))
    pad_divisor  = dc.get('pad_divisor', 4)
    dataset_type = dc.get('dataset_type', 'irdst').lower()

    common_kwargs = dict(
        image_size         = input_shape[0],
        type               = split,
        augment            = False,
        crop_size          = dc.get('crop_size', 480),
        base_size          = dc.get('base_size', 512),
        pad_only           = dc.get('pad_only', True),
        pad_to_size        = eval_pad,
        pad_divisor        = pad_divisor,
        flip_augmentation  = False,
        vflip_augmentation = False,
        brightness_jitter  = 0.0,
        contrast_jitter    = 0.0,
        gaussian_noise     = 0.0,
        scale_jitter       = 0.0,
    )

    if dataset_type == 'itsdt':
        dataset = ITSDTDataset(dataset_root=dataset_root, **common_kwargs)
    else:  # 默认 irdst
        dataset = IRDSTDataset(dataset_root=dataset_root, **common_kwargs)

    print(f'[INFO] Eval dataset ({split}, type={dataset_type}): {len(dataset)} images  '
          f'[root: {os.path.abspath(dataset_root)}]')

    if len(dataset) == 0:
        raise RuntimeError(
            f'[FATAL] Eval dataset is EMPTY!\n'
            f'  dataset_root = {dataset_root}\n'
            f'  abs path     = {os.path.abspath(dataset_root)}\n'
            f'  dataset_type = {dataset_type}\n'
            f'Please check that the dataset path is correct and accessible.'
        )

    return dataset


# =============================================================================
#  Evaluation  (calls evaluate_detection() exactly as train_dist_2.py does)
# =============================================================================

def run_evaluation(
    model:       nn.Module,
    dataset:     IRDSTDataset,
    device:      torch.device,
    conf_thresh: float,
    iou_thresh:  float,
    nms_thresh:  float,
    save_vis:    bool = False,
    vis_dir:     str  = './outputs/vis',
) -> dict:
    """
    Full evaluation loop delegated to evaluate_detection() in utils/eval_utils.py.
    Mirrors the evaluate_model() call inside train_dist_2.py exactly.
    """
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    with torch.no_grad():
        results = evaluate_detection(
            model=model,
            test_dataset=dataset,
            conf_thresh=conf_thresh,
            nms_threshold=nms_thresh,
            iou_threshold=iou_thresh,
            max_detections=300,
            device=str(device),
            warmup_iterations=3,
            local_rank=0,
            disable_pbar=False,
            save_visualizations=save_vis,
            output_dir=vis_dir if save_vis else None,
            show_score=True,
        )
    return results


# =============================================================================
#  CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description='MemISTD Small Target — Standalone Evaluator',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--weights', default='./checkpoint/20260312_154828/last.pth',
                   help='Path to checkpoint file (.pth)')
    p.add_argument('--config',
                   default='config/memistd_small_target_config_v2.yaml',
                   help='Path to YAML config file')
    p.add_argument('--split', default='test',
                   choices=['test', 'train'],
                   help='Dataset split to evaluate')
    p.add_argument('--device', default='',
                   help='Device: "cuda", "cuda:0", "cpu", or "" for auto-detect')

    # Threshold overrides (fall back to config values when not specified)
    p.add_argument('--conf-thresh', type=float, default=0.27,
                   help='Confidence threshold (overrides config eval.conf_thresh)')
    p.add_argument('--iou-thresh',  type=float, default=None,
                   help='IoU match threshold for TP/FP (overrides config eval.iou_thresh)')
    p.add_argument('--nms-thresh',  type=float, default=None,
                   help='NMS IoU threshold (overrides config eval.nms_thresh)')

    # Visualisation
    p.add_argument('--save-vis', action='store_true',
                   help='Save detection visualisation images (requires PIL)')
    p.add_argument('--vis-dir', default='./outputs/vis',
                   help='Output directory for visualisation images')
    return p.parse_args()


# =============================================================================
#  Entry point
# =============================================================================

def main():
    args = parse_args()

    # ── device ────────────────────────────────────────────────────────────────
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[INFO] Using device: {device}')

    # ── config ────────────────────────────────────────────────────────────────
    cfg      = load_config(args.config)
    eval_cfg = cfg.get('eval', {})

    conf_thresh = args.conf_thresh if args.conf_thresh is not None \
        else eval_cfg.get('conf_thresh', 0.27)
    iou_thresh  = args.iou_thresh  if args.iou_thresh  is not None \
        else eval_cfg.get('iou_thresh',  0.2)
    nms_thresh  = args.nms_thresh  if args.nms_thresh  is not None \
        else eval_cfg.get('nms_thresh',  0.2)

    print(f'[INFO] Thresholds — conf={conf_thresh}  iou={iou_thresh}  nms={nms_thresh}')

    # ── model ─────────────────────────────────────────────────────────────────
    model = build_model(cfg)
    model = load_checkpoint(model, args.weights, device)
    model = model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f'[INFO] Model parameters: {total_params / 1e6:.2f} M')

    # ── dataset ───────────────────────────────────────────────────────────────
    dataset = build_eval_dataset(cfg, split=args.split)

    # ── evaluate ──────────────────────────────────────────────────────────────
    print('\n[INFO] Starting evaluation ...')
    t0 = time.perf_counter()

    results = run_evaluation(
        model=model,
        dataset=dataset,
        device=device,
        conf_thresh=conf_thresh,
        iou_thresh=iou_thresh,
        nms_thresh=nms_thresh,
        save_vis=args.save_vis,
        vis_dir=args.vis_dir,
    )

    wall_time = time.perf_counter() - t0
    print(f'[INFO] Evaluation finished in {wall_time:.1f}s')

    # ── report ────────────────────────────────────────────────────────────────
    print()
    print_evaluation_report(results)

    # ── summary line (machine-readable) ───────────────────────────────────────
    print('\n[SUMMARY]')
    print(f"  Recall    : {results['recall']    * 100:.2f}%")
    print(f"  Precision : {results['precision'] * 100:.2f}%")
    print(f"  F1        : {results['f1']        * 100:.2f}%")
    print(f"  AP@0.5    : {results['ap50']      * 100:.2f}%")
    print(f"  FPS       : {results['fps']:.2f}")
    print(f"  Total GT  : {results['total_gt']}")
    print(f"  TP / FP / FN : {results['tp']} / {results['fp']} / {results['fn']}")

    if args.save_vis:
        print(f'\n[INFO] Visualisations saved to: {args.vis_dir}')


if __name__ == '__main__':
    main()
