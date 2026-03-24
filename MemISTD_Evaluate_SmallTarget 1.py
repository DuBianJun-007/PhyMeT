"""
MemISTD Small Target Detection Evaluation Script
=================================================

Evaluate model performance with metrics:
- Recall (R)
- Precision (P)
- F1 Score
- AP@0.5 (AP50)
- FPS (Frames Per Second)

No OpenCV dependency - pure NumPy/PyTorch/PIL implementation.

Usage:
    python MemISTD_Evaluate_SmallTarget.py --weights path/to/best.pth --config config/memistd_small_target_config.yaml
    python MemISTD_Evaluate_SmallTarget.py --save-vis --vis-dir ./outputs/
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import json
import yaml
from datetime import datetime
from typing import Dict

import torch

from model_1.memistd_small_target import MemISTDSmallTarget
from MemISTD_Dataloader import IRDSTDataset
from utils.eval_utils import evaluate_detection, print_evaluation_report


def save_results(metrics: Dict[str, float], output_path: str, model_path: str):
    """Save evaluation results to JSON file."""
    results = {
        "timestamp": datetime.now().isoformat(),
        "model_path": model_path,
        "metrics": {
            "recall": float(metrics["recall"]),
            "precision": float(metrics["precision"]),
            "f1": float(metrics["f1"]),
            "ap50": float(metrics["ap50"]),
            "fps": float(metrics["fps"]),
            "avg_inference_time_ms": float(metrics["avg_inference_time_ms"]),
        },
        "statistics": {
            "total_images": int(metrics.get("total_images", 0)),
            "total_gt": int(metrics["total_gt"]),
            "tp": int(metrics["tp"]),
            "fp": int(metrics["fp"]),
            "fn": int(metrics["fn"]),
        },
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="MemISTD Small Target Detection Evaluation")
    parser.add_argument(
        "--weights",
        type=str,
        default="./checkpoint/20260224_115906/last.pth",
        help="Path to model weights",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./checkpoint/20260224_115906/config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--conf-thres",
        type=float,
        default=0.13,
        help="Confidence threshold for detection",
    )
    parser.add_argument(
        "--nms-thres",
        type=float,
        default=0.1,
        help="NMS IoU threshold",
    )
    parser.add_argument(
        "--iou-thres",
        type=float,
        default=0.2,
        help="IoU threshold for evaluation",
    )
    parser.add_argument(
        "--max-detections",
        type=int,
        default=10,
        help="Maximum detections per image",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for evaluation results (JSON)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda or cpu)",
    )
    parser.add_argument(
        "--save-vis",
        action="store_true",
        # default=False,
        default=True,
        help="Save visualization results (detection boxes drawn on images)",
    )
    parser.add_argument(
        "--vis-dir",
        type=str,
        default=r"e:\workspace\MemISTD\dataset\IRDST\IRDST_real\images\outputs",
        help="Directory to save visualization results",
    )
    parser.add_argument(
        "--show-score",
        action="store_true",
        default=False,
        help="Show confidence scores on visualization (default: False)",
    )
    
    args = parser.parse_args()
    
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"
    
    # 自动查找最新的 checkpoint
    if args.weights is None:
        checkpoint_dir = "./checkpoint"
        if os.path.exists(checkpoint_dir):
            checkpoint_subdirs = [d for d in os.listdir(checkpoint_dir) 
                                  if os.path.isdir(os.path.join(checkpoint_dir, d))]
            if checkpoint_subdirs:
                checkpoint_subdirs.sort(reverse=True)
                for subdir in checkpoint_subdirs:
                    best_path = os.path.join(checkpoint_dir, subdir, "best.pth")
                    last_path = os.path.join(checkpoint_dir, subdir, "last.pth")
                    if os.path.exists(best_path):
                        args.weights = best_path
                        print(f"Auto-found best.pth in: {best_path}")
                        break
                    elif os.path.exists(last_path):
                        args.weights = last_path
                        print(f"Auto-found last.pth in: {last_path}")
                        break
        
        if args.weights is None:
            parser.error("No checkpoint found. Please specify --weights path.")
    
    weights_dir = os.path.dirname(args.weights)
    if args.config is None:
        config_path = os.path.join(weights_dir, "config.yaml")
        if os.path.exists(config_path):
            args.config = config_path
            print(f"Using config from weights directory: {config_path}")
        else:
            args.config = "config/memistd_small_target_config.yaml"
            print(f"Using default config: {args.config}")
    
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    print("=" * 70)
    print("Loading model...")
    print("=" * 70)
    
    # 加载模型
    model = MemISTDSmallTarget(
        in_channels=cfg["model"]["in_channels"],
        num_classes=cfg["model"]["num_classes"],
        base_channels=cfg["model"]["base_channels"],
        backbone_depth=cfg["model"]["backbone_depth"],
        target_memory_slots=cfg["model"]["target_memory_slots"],
        background_memory_slots=cfg["model"]["background_memory_slots"],
        use_attention=cfg["model"]["use_attention"],
        norm_type=cfg["model"].get("norm_type", "bn"),
        act_type=cfg["model"].get("act_type", "relu"),
        use_residual=cfg["model"].get("use_residual", True),
        save_heatmaps=cfg["model"].get("save_heatmaps", False),
        ms_fusion_type=cfg["model"].get("ms_fusion_type", "attention"),
        global_fusion_strategy=cfg["model"].get("global_fusion_strategy", "spatial_reweight"),
        bg_patch_sizes=cfg["model"].get("bg_patch_sizes", [3, 5]),
        tg_patch_sizes=cfg["model"].get("tg_patch_sizes", [3, 5]),
    )
    
    checkpoint = torch.load(args.weights, map_location="cpu")
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        epoch_info = checkpoint.get("epoch", "unknown")
        print(f"Checkpoint epoch: {epoch_info}")
    else:
        model.load_state_dict(checkpoint)
    
    model.to(args.device)
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded from: {args.weights}")
    print(f"Total parameters: {total_params:,}")
    
    # 获取数据配置
    input_shape = tuple(cfg["data"]["input_shape"])
    eval_pad_to_size = cfg["data"].get("eval_pad_to_size", cfg["data"].get("pad_to_size"))
    
    # 加载测试数据集
    test_dataset = IRDSTDataset(
        dataset_root=cfg["data"]["dataset_root"],
        image_size=input_shape[0],
        type="test",
        augment=False,
        crop_size=cfg["data"].get("crop_size", 480),
        base_size=cfg["data"].get("base_size", 512),
        pad_only=cfg["data"].get("pad_only", True),
        pad_to_size=eval_pad_to_size,
        pad_divisor=cfg["data"].get("pad_divisor", 32),
        flip_augmentation=False,
        vflip_augmentation=False,
        brightness_jitter=0.0,
        contrast_jitter=0.0,
        gaussian_noise=0.0,
        scale_jitter=0.0,
    )
    
    print(f"Test dataset: {len(test_dataset)} images")
    
    # 设置可视化目录
    vis_dir = args.vis_dir
    if vis_dir is None and args.save_vis:
        vis_dir = r"e:\workspace\MemISTD\dataset\IRDST\IRDST_real\images\outputs"
    
    print("\n" + "=" * 70)
    print("Starting evaluation...")
    print(f"Confidence threshold: {args.conf_thres}")
    print(f"NMS threshold: {args.nms_thres}")
    print(f"IoU threshold: {args.iou_thres}")
    if args.save_vis:
        print(f"Save visualizations: {vis_dir}")
    print("=" * 70)
    
    # 运行评估
    metrics = evaluate_detection(
        model=model,
        test_dataset=test_dataset,
        conf_threshold=args.conf_thres,
        nms_threshold=args.nms_thres,
        iou_threshold=args.iou_thres,
        max_detections=args.max_detections,
        device=args.device,
        warmup_iterations=3,
        local_rank=0,
        disable_pbar=False,
        save_visualizations=args.save_vis,
        output_dir=vis_dir,
        show_score=args.show_score,
    )
    
    print_evaluation_report(metrics, args.weights)
    
    if args.output is None:
        output_name = os.path.splitext(os.path.basename(args.weights))[0]
        args.output = os.path.join(weights_dir, f"{output_name}_eval_results.json")
    
    save_results(metrics, args.output, args.weights)
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
