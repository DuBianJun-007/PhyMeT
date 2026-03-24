"""
MemISTD Small Target Detection Evaluation Script
=================================================

Evaluate model performance with metrics:
- Recall (R)
- Precision (P)
- F1 Score
- AP@0.5 (AP50)
- FPS (Frames Per Second)

Usage:
    python MemISTD_Evaluate_SmallTarget.py --weights path/to/best.pth --config config/memistd_small_target_config.yaml
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import time
import json
import yaml
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2

from model.memistd_small_target import MemISTDSmallTarget
from MemISTD_Dataloader import IRDSTDataset


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Compute IoU between two boxes.
    
    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
    
    Returns:
        IoU value
    """
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = area1 + area2 - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


def compute_metrics(
    all_predictions: List[np.ndarray],
    all_targets: List[np.ndarray],
    iou_threshold: float = 0.5,
    conf_threshold: float = 0.05,
) -> Dict[str, float]:
    """
    Compute detection metrics.
    
    Args:
        all_predictions: List of prediction arrays, each [N, 6] (x1, y1, x2, y2, score, class)
        all_targets: List of target arrays, each [M, 4] (x, y, w, h)
        iou_threshold: IoU threshold for matching
        conf_threshold: Confidence threshold for filtering
    
    Returns:
        Dictionary of metrics
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_gt = 0
    
    all_scores = []
    all_tp_flags = []
    
    for preds, targets in zip(all_predictions, all_targets):
        if len(targets) > 0:
            gt_boxes = targets.copy()
            if gt_boxes.shape[1] == 4:
                x, y, w, h = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2], gt_boxes[:, 3]
                gt_boxes = np.stack([x, y, x + w, y + h], axis=1)
            total_gt += len(gt_boxes)
        else:
            gt_boxes = np.zeros((0, 4))
        
        if len(preds) > 0:
            valid_mask = preds[:, 4] >= conf_threshold
            preds = preds[valid_mask]
        
        if len(preds) == 0:
            total_fn += len(gt_boxes)
            continue
        
        pred_boxes = preds[:, :4]
        pred_scores = preds[:, 4]
        
        gt_matched = np.zeros(len(gt_boxes), dtype=bool)
        
        sorted_indices = np.argsort(-pred_scores)
        
        for idx in sorted_indices:
            pred_box = pred_boxes[idx]
            pred_score = pred_scores[idx]
            
            best_iou = 0.0
            best_gt_idx = -1
            
            for gt_idx, gt_box in enumerate(gt_boxes):
                if gt_matched[gt_idx]:
                    continue
                
                iou = compute_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            all_scores.append(pred_score)
            
            if best_iou >= iou_threshold:
                total_tp += 1
                gt_matched[best_gt_idx] = True
                all_tp_flags.append(1)
            else:
                total_fp += 1
                all_tp_flags.append(0)
        
        total_fn += np.sum(~gt_matched)
    
    recall = total_tp / max(total_gt, 1)
    precision = total_tp / max(total_tp + total_fp, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)
    
    ap = compute_ap(all_scores, all_tp_flags, total_gt)
    
    return {
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "ap50": ap,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "total_gt": total_gt,
    }


def compute_ap(
    scores: List[float],
    tp_flags: List[int],
    num_gt: int,
) -> float:
    """
    Compute Average Precision using 11-point interpolation.
    
    Args:
        scores: List of confidence scores
        tp_flags: List of TP flags (1 for TP, 0 for FP)
        num_gt: Total number of ground truth boxes
    
    Returns:
        AP value
    """
    if len(scores) == 0 or num_gt == 0:
        return 0.0
    
    scores = np.array(scores)
    tp_flags = np.array(tp_flags)
    
    sorted_indices = np.argsort(-scores)
    tp_flags = tp_flags[sorted_indices]
    
    tp_cumsum = np.cumsum(tp_flags)
    fp_cumsum = np.cumsum(1 - tp_flags)
    
    recall = tp_cumsum / num_gt
    precision = tp_cumsum / (tp_cumsum + fp_cumsum)
    
    recall_levels = np.linspace(0, 1, 11)
    ap = 0.0
    
    for level in recall_levels:
        mask = recall >= level
        if np.any(mask):
            ap += np.max(precision[mask])
    
    ap /= 11.0
    
    return ap


def evaluate_model(
    model: MemISTDSmallTarget,
    test_dataset: IRDSTDataset,
    conf_threshold: float = 0.05,
    nms_threshold: float = 0.5,
    iou_threshold: float = 0.5,
    max_detections: int = 30,
    device: str = "cuda",
    warmup_iterations: int = 0,
    save_visualizations: bool = False,
    output_dir: str = None,
) -> Dict[str, float]:
    """
    Evaluate model on test dataset.
    
    Args:
        model: Model to evaluate
        test_dataset: Test dataset
        conf_threshold: Confidence threshold
        nms_threshold: NMS threshold
        iou_threshold: IoU threshold for evaluation
        max_detections: Maximum detections per image
        device: Device to use
        warmup_iterations: Number of warmup iterations for FPS measurement
        save_visualizations: Whether to save visualization results
        output_dir: Directory to save visualization results
    
    Returns:
        Dictionary of evaluation results
    """
    model.eval()
    model.to(device)
    
    all_predictions = []
    all_targets = []
    
    inference_times = []
    
    if save_visualizations and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Visualizations will be saved to: {output_dir}")
    
    print("Warming up model...")
    warmup_images = torch.randn(1, 1, 480, 720).to(device)
    for _ in range(warmup_iterations):
        with torch.no_grad():
            _ = model.detect(warmup_images, conf_thres=conf_threshold, nms_thres=nms_threshold)
    
    if device == "cuda":
        torch.cuda.synchronize()
    
    print("Running evaluation...")
    total_detections = 0
    total_gt = 0
    
    pbar = tqdm(range(len(test_dataset)), desc="Evaluating")
    for idx in pbar:
        image_tensor, target = test_dataset[idx]
        
        image_batch = image_tensor.unsqueeze(0).to(device)
        
        if device == "cuda":
            torch.cuda.synchronize()
        
        start_time = time.perf_counter()
        
        with torch.no_grad():
            detections = model.detect(
                image_batch,
                conf_thres=conf_threshold,
                nms_thres=nms_threshold,
                max_detections=max_detections,
            )
        
        if device == "cuda":
            torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        inference_times.append(end_time - start_time)
        
        detections = detections.cpu().numpy()[0]
        valid_mask = detections[:, 4] > 0
        valid_dets = detections[valid_mask]
        all_predictions.append(valid_dets)
        
        boxes = target["boxes"]
        if torch.is_tensor(boxes):
            boxes = boxes.cpu().numpy()
        all_targets.append(boxes)
        
        num_dets = len(valid_dets)
        num_gt = len(boxes)
        total_detections += num_dets
        total_gt += num_gt
        
        pbar.set_postfix({
            "dets": num_dets,
            "total_dets": total_detections,
            "total_gt": total_gt
        })
        
        if save_visualizations and output_dir:
            img_array = image_tensor[0].cpu().numpy()
            # 还原预处理: x = x' * std + mean = x' * 1.0923 - 0.1246
            img_array = img_array * 1.0923 - 0.1246
            img_array = np.clip(img_array, 0, 1)
            img_array = (img_array * 255).astype(np.uint8)
            
            if len(img_array.shape) == 2:
                vis_img = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
            else:
                vis_img = img_array.copy()
            
            for gt_box in boxes:
                x, y, w, h = gt_box
                x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 1)
            
            for det in valid_dets:
                x1, y1, x2, y2, score, cls = det
                
                # 检查坐标是否有效
                if not all(np.isfinite([x1, y1, x2, y2])):
                    continue
                
                # 确保坐标顺序正确
                if x1 > x2:
                    x1, x2 = x2, x1
                if y1 > y2:
                    y1, y2 = y2, y1
                
                # 限制在图像范围内
                h, w = vis_img.shape[:2]
                x1 = max(0, min(int(x1), w - 1))
                y1 = max(0, min(int(y1), h - 1))
                x2 = max(0, min(int(x2), w - 1))
                y2 = max(0, min(int(y2), h - 1))
                
                # 确保宽高至少为 1
                if x2 <= x1:
                    x2 = x1 + 1
                if y2 <= y1:
                    y2 = y1 + 1
                
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 0, 255), 1)
                # cv2.putText(vis_img, f"{score:.2f}", (x1, y1 - 2),    # 打印分数
                #            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            
            file_name = target.get("file_name", f"{idx:08d}")
            if not file_name.endswith('.png'):
                file_name = f"{file_name}.png"
            output_path = os.path.join(output_dir, file_name)
            cv2.imwrite(output_path, vis_img)
    
    metrics = compute_metrics(
        all_predictions,
        all_targets,
        iou_threshold=iou_threshold,
        conf_threshold=conf_threshold,
    )
    
    avg_inference_time = np.mean(inference_times)
    fps = 1.0 / avg_inference_time
    
    metrics["fps"] = fps
    metrics["avg_inference_time_ms"] = avg_inference_time * 1000
    metrics["total_images"] = len(test_dataset)
    
    return metrics


def print_report(metrics: Dict[str, float], model_path: str, dataset_info: str = ""):
    """Print evaluation report."""
    print("\n" + "=" * 70)
    print("MemISTD Small Target Detection - Evaluation Report")
    print("=" * 70)
    print(f"Model: {model_path}")
    if dataset_info:
        print(f"Dataset: {dataset_info}")
    print(f"Total Images: {int(metrics.get('total_images', 0))}")
    
    print("\n" + "-" * 70)
    print("Performance Metrics")
    print("-" * 70)
    print(f"  Recall:       {metrics['recall'] * 100:.2f}%")
    print(f"  Precision:    {metrics['precision'] * 100:.2f}%")
    print(f"  F1 Score:     {metrics['f1'] * 100:.2f}%")
    print(f"  AP@0.5:       {metrics['ap50'] * 100:.2f}%")
    
    print("\n" + "-" * 70)
    print("Speed Metrics")
    print("-" * 70)
    print(f"  FPS:              {metrics['fps']:.2f} frames/sec")
    print(f"  Inference Time:   {metrics['avg_inference_time_ms']:.2f} ms/image")
    
    print("\n" + "-" * 70)
    print("Detection Statistics")
    print("-" * 70)
    print(f"  Total GT Boxes:    {int(metrics['total_gt'])}")
    print(f"  True Positives:    {int(metrics['tp'])}")
    print(f"  False Positives:   {int(metrics['fp'])}")
    print(f"  False Negatives:   {int(metrics['fn'])}")
    
    print("\n" + "=" * 70)


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
        default="./checkpoint/20260220_192628/last.pth",
        help="Path to model weights (e.g., checkpoint/20260213_143310/best.pth)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default='./checkpoint/20260220_192628/config.yaml',
        help="Path to config file (default: use config.yaml from weights directory)",
    )
    parser.add_argument(
        "--conf-thres",
        type=float,
        default=0.001,
        help="Confidence threshold for detection (default: use config value)",
    )
    parser.add_argument(
        "--nms-thres",
        type=float,
        default=0.5,
        help="NMS IoU threshold (default: use config value)",
    )
    parser.add_argument(
        "--iou-thres",
        type=float,
        default=0.5,
        help="IoU threshold for evaluation",
    )
    parser.add_argument(
        "--max-detections",
        type=int,
        default=300,
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
        default=True,
        help="Save visualization results (detection boxes drawn on images)",
    )
    parser.add_argument(
        "--vis-dir",
        type=str,
        default="./dataset/IRDST/IRDST_real/images/outputs/",
        help="Directory to save visualization results (default: dataset_root/images/outputs)",
    )
    
    args = parser.parse_args()
    
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"
    
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
    
    if args.conf_thres is None:
        args.conf_thres = cfg.get("eval", {}).get("conf_thresh", 0.05)
    if args.nms_thres is None:
        args.nms_thres = cfg.get("eval", {}).get("nms_thresh", 0.5)
    
    print("=" * 70)
    print("Loading model...")
    print("=" * 70)
    
    model = MemISTDSmallTarget(
        in_channels=cfg["model"]["in_channels"],
        num_classes=cfg["model"]["num_classes"],
        base_channels=cfg["model"]["base_channels"],
        backbone_depth=cfg["model"]["backbone_depth"],
        target_memory_slots=cfg["model"]["target_memory_slots"],
        background_memory_slots=cfg["model"]["background_memory_slots"],
        use_attention=cfg["model"]["use_attention"],
    )
    
    checkpoint = torch.load(args.weights, map_location=args.device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        epoch_info = checkpoint.get("epoch", "unknown")
        mAP_info = checkpoint.get("mAP", "unknown")
        print(f"Checkpoint epoch: {epoch_info}")
        if mAP_info is not None and mAP_info != "unknown":
            print(f"Checkpoint mAP: {mAP_info * 100:.2f}%")
    else:
        model.load_state_dict(checkpoint)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded successfully!")
    print(f"Total parameters: {total_params:,}")
    
    print("\n" + "=" * 70)
    print("Loading test dataset...")
    print("=" * 70)
    
    input_shape = cfg["data"]["input_shape"]
    eval_pad_to_size = cfg["data"].get("eval_pad_to_size", input_shape)
    
    print(f"Input shape: {input_shape}")
    print(f"Eval pad to size: {eval_pad_to_size}")
    
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
    
    print(f"Test dataset size: {len(test_dataset)} images")
    
    vis_dir = args.vis_dir
    if vis_dir is None and args.save_vis:
        dataset_root = cfg["data"]["dataset_root"]
        vis_dir = os.path.join(dataset_root, "images", "outputs")
    
    print("\n" + "=" * 70)
    print("Starting evaluation...")
    print(f"Confidence threshold: {args.conf_thres}")
    print(f"NMS threshold: {args.nms_thres}")
    print(f"IoU threshold: {args.iou_thres}")
    if args.save_vis:
        print(f"Save visualizations: {vis_dir}")
    print("=" * 70)
    
    metrics = evaluate_model(
        model=model,
        test_dataset=test_dataset,
        conf_threshold=args.conf_thres,
        nms_threshold=args.nms_thres,
        iou_threshold=args.iou_thres,
        max_detections=args.max_detections,
        device=args.device,
        save_visualizations=args.save_vis,
        output_dir=vis_dir,
    )
    
    print_report(
        metrics=metrics,
        model_path=args.weights,
        dataset_info=f"Test set ({len(test_dataset)} images)",
    )
    
    if args.output is None:
        output_name = os.path.splitext(os.path.basename(args.weights))[0]
        args.output = os.path.join(weights_dir, f"{output_name}_eval_results.json")
    
    save_results(metrics, args.output, args.weights)


if __name__ == "__main__":
    main()
