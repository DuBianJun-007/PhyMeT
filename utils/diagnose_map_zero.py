"""
Diagnostic script to debug why mAP = 0%
=========================================

This script checks:
1. Model output confidence distribution
2. Detection results before/after filtering
3. Ground truth box format
4. Preprocessing consistency
"""

import os
import torch
import numpy as np
import yaml
from tqdm import tqdm

from model.memistd_small_target import MemISTDSmallTarget
from MemISTD_Dataloader import IRDSTDataset


def diagnose_model(config_path="config/memistd_small_target_config.yaml", weights_path=None):
    print("=" * 70)
    print("MemISTD Diagnostic Tool - Why mAP = 0%?")
    print("=" * 70)
    
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    print("\n[1] Loading model...")
    model = MemISTDSmallTarget(
        in_channels=cfg["model"]["in_channels"],
        num_classes=cfg["model"]["num_classes"],
        base_channels=cfg["model"]["base_channels"],
        backbone_depth=cfg["model"]["backbone_depth"],
        target_memory_slots=cfg["model"]["target_memory_slots"],
        background_memory_slots=cfg["model"]["background_memory_slots"],
        use_attention=cfg["model"]["use_attention"],
    )
    
    if weights_path and os.path.exists(weights_path):
        print(f"Loading weights from: {weights_path}")
        checkpoint = torch.load(weights_path, map_location=device)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
    else:
        print("WARNING: No weights loaded, using random initialization!")
    
    model = model.to(device)
    model.eval()
    
    print("\n[2] Loading test dataset...")
    input_shape = cfg["data"]["input_shape"]
    test_dataset = IRDSTDataset(
        dataset_root=cfg["data"]["dataset_root"],
        image_size=input_shape[0],
        type="test",
        augment=False,
        crop_size=cfg["data"].get("crop_size", 480),
        base_size=cfg["data"].get("base_size", 512),
        pad_only=cfg["data"].get("pad_only", True),
        pad_to_size=cfg["data"].get("pad_to_size", input_shape),
        pad_divisor=cfg["data"].get("pad_divisor", 32),
        flip_augmentation=False,
        vflip_augmentation=False,
        brightness_jitter=0.0,
        contrast_jitter=0.0,
        gaussian_noise=0.0,
        scale_jitter=0.0,
    )
    print(f"Test dataset size: {len(test_dataset)}")
    
    print("\n[3] Analyzing model outputs...")
    
    all_obj_scores = []
    all_cls_scores = []
    all_final_scores = []
    all_gt_boxes = []
    all_pred_boxes_before_filter = []
    all_pred_boxes_after_filter = []
    
    conf_thresholds = [0.001, 0.01, 0.05, 0.1, 0.3, 0.5]
    
    num_samples = min(20, len(test_dataset))
    
    for idx in tqdm(range(num_samples), desc="Analyzing"):
        image_tensor, target = test_dataset[idx]
        image_batch = image_tensor.unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(image_batch)
        
        predictions = outputs["predictions"]
        
        for scale in ["P0", "P1", "P2"]:
            obj = predictions[f"{scale}_obj"]
            cls = predictions[f"{scale}_cls"]
            
            obj_scores = torch.sigmoid(obj).cpu().numpy().flatten()
            all_obj_scores.extend(obj_scores)
            
            if cls is not None:
                cls_scores = torch.sigmoid(cls).cpu().numpy().flatten()
                all_cls_scores.extend(cls_scores)
        
        boxes = target["boxes"]
        if len(boxes) > 0:
            all_gt_boxes.append(boxes.cpu().numpy() if torch.is_tensor(boxes) else boxes)
        
        with torch.no_grad():
            detections = model.detect(
                image_batch,
                conf_thres=0.001,
                nms_thres=0.5,
                max_detections=300,
            )
        
        det = detections[0].cpu().numpy()
        valid_mask = det[:, 4] > 0
        valid_dets = det[valid_mask]
        
        if len(valid_dets) > 0:
            all_final_scores.extend(valid_dets[:, 4].tolist())
            all_pred_boxes_before_filter.append(valid_dets)
        
        for thresh in [0.001, 0.01, 0.05]:
            filtered = valid_dets[valid_dets[:, 4] > thresh]
            if idx == 0:
                all_pred_boxes_after_filter.append((thresh, len(filtered)))
    
    print("\n" + "=" * 70)
    print("DIAGNOSTIC RESULTS")
    print("=" * 70)
    
    print("\n[4] Objectness Score Distribution:")
    obj_scores = np.array(all_obj_scores)
    print(f"  Min:    {obj_scores.min():.6f}")
    print(f"  Max:    {obj_scores.max():.6f}")
    print(f"  Mean:   {obj_scores.mean():.6f}")
    print(f"  Median: {np.median(obj_scores):.6f}")
    print(f"  Std:    {obj_scores.std():.6f}")
    
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    print(f"\n  Percentiles:")
    for p in percentiles:
        print(f"    {p}%: {np.percentile(obj_scores, p):.6f}")
    
    if all_cls_scores:
        print("\n[5] Classification Score Distribution:")
        cls_scores = np.array(all_cls_scores)
        print(f"  Min:    {cls_scores.min():.6f}")
        print(f"  Max:    {cls_scores.max():.6f}")
        print(f"  Mean:   {cls_scores.mean():.6f}")
    
    print("\n[6] Final Detection Score Distribution (after NMS):")
    if all_final_scores:
        final_scores = np.array(all_final_scores)
        print(f"  Total detections: {len(final_scores)}")
        print(f"  Min:    {final_scores.min():.6f}")
        print(f"  Max:    {final_scores.max():.6f}")
        print(f"  Mean:   {final_scores.mean():.6f}")
        
        print(f"\n  Detections above threshold:")
        for thresh in [0.001, 0.01, 0.05, 0.1, 0.3, 0.5]:
            count = np.sum(final_scores > thresh)
            print(f"    > {thresh}: {count} detections ({100*count/len(final_scores):.1f}%)")
    else:
        print("  WARNING: No detections found even with conf_thres=0.001!")
        print("  This indicates a serious problem with the model.")
    
    print("\n[7] Ground Truth Analysis:")
    if all_gt_boxes:
        total_gt = sum(len(boxes) for boxes in all_gt_boxes)
        print(f"  Total GT boxes in {num_samples} samples: {total_gt}")
        
        all_widths = []
        all_heights = []
        for boxes in all_gt_boxes:
            for box in boxes:
                all_widths.append(box[2])
                all_heights.append(box[3])
        
        print(f"  GT box width range: [{min(all_widths):.1f}, {max(all_widths):.1f}]")
        print(f"  GT box height range: [{min(all_heights):.1f}, {max(all_heights):.1f}]")
        print(f"  GT box avg size: {np.mean(all_widths):.1f} x {np.mean(all_heights):.1f}")
    
    print("\n[8] Potential Issues:")
    issues = []
    
    if obj_scores.max() < 0.05:
        issues.append("CRITICAL: Max objectness score is below threshold 0.05!")
        issues.append("  -> Model is not confident about any detection")
        issues.append("  -> Solution: Lower conf_thresh or train longer")
    
    if obj_scores.mean() < 0.01:
        issues.append("WARNING: Mean objectness is very low")
        issues.append("  -> Model might not have learned to detect targets")
    
    if not all_final_scores:
        issues.append("CRITICAL: No detections even with conf_thres=0.001!")
        issues.append("  -> Check if model weights are loaded correctly")
        issues.append("  -> Check if model architecture matches training")
    
    if len(all_final_scores) > 0 and max(all_final_scores) < 0.01:
        issues.append("CRITICAL: All detection scores are below 0.01")
        issues.append("  -> Model has not converged properly")
        issues.append("  -> Try: lower learning rate, longer training, or check loss function")
    
    if issues:
        for issue in issues:
            print(f"  {issue}")
    else:
        print("  No obvious issues found in score distribution.")
        print("  The problem might be in:")
        print("    - Box coordinate format mismatch")
        print("    - Preprocessing inconsistency")
        print("    - IoU calculation")
    
    print("\n[9] Recommendations:")
    print("  1. Try evaluating with conf_thresh=0.001 to see if there are any detections")
    print("  2. Check if training loss components are balanced (box_loss vs obj_loss)")
    print("  3. Verify that GT boxes are in correct format (x, y, w, h)")
    print("  4. Check if model weights are loaded correctly")
    print("  5. Visualize some predictions vs ground truth")
    
    print("\n" + "=" * 70)
    
    return {
        "obj_scores": obj_scores,
        "final_scores": np.array(all_final_scores) if all_final_scores else None,
        "num_gt": sum(len(boxes) for boxes in all_gt_boxes) if all_gt_boxes else 0,
    }


def visualize_predictions(config_path="config/memistd_small_target_config.yaml", weights_path=None, num_images=5):
    """Visualize predictions vs ground truth"""
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = MemISTDSmallTarget(
        in_channels=cfg["model"]["in_channels"],
        num_classes=cfg["model"]["num_classes"],
        base_channels=cfg["model"]["base_channels"],
        backbone_depth=cfg["model"]["backbone_depth"],
        target_memory_slots=cfg["model"]["target_memory_slots"],
        background_memory_slots=cfg["model"]["background_memory_slots"],
        use_attention=cfg["model"]["use_attention"],
    )
    
    if weights_path and os.path.exists(weights_path):
        checkpoint = torch.load(weights_path, map_location=device)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    input_shape = cfg["data"]["input_shape"]
    test_dataset = IRDSTDataset(
        dataset_root=cfg["data"]["dataset_root"],
        image_size=input_shape[0],
        type="test",
        augment=False,
        crop_size=cfg["data"].get("crop_size", 480),
        base_size=cfg["data"].get("base_size", 512),
        pad_only=cfg["data"].get("pad_only", True),
        pad_to_size=cfg["data"].get("pad_to_size", input_shape),
    )
    
    fig, axes = plt.subplots(2, num_images, figsize=(4*num_images, 8))
    
    for idx in range(min(num_images, len(test_dataset))):
        image_tensor, target = test_dataset[idx]
        image_batch = image_tensor.unsqueeze(0).to(device)
        
        with torch.no_grad():
            detections = model.detect(image_batch, conf_thres=0.001, nms_thres=0.5)
        
        det = detections[0].cpu().numpy()
        valid_mask = det[:, 4] > 0.001
        valid_dets = det[valid_mask]
        
        img = image_tensor[0].cpu().numpy()
        
        ax1 = axes[0, idx]
        ax2 = axes[1, idx]
        
        ax1.imshow(img, cmap='gray')
        ax1.set_title(f'Image {idx}\nGT: {len(target["boxes"])} boxes')
        
        for box in target["boxes"]:
            x, y, w, h = box.cpu().numpy() if torch.is_tensor(box) else box
            rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='g', facecolor='none')
            ax1.add_patch(rect)
        
        ax2.imshow(img, cmap='gray')
        ax2.set_title(f'Pred: {len(valid_dets)} boxes')
        
        for det_box in valid_dets[:20]:
            x1, y1, x2, y2, score, cls = det_box
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')
            ax2.add_patch(rect)
            ax2.text(x1, y1-2, f'{score:.3f}', fontsize=6, color='r')
        
        ax1.axis('off')
        ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig('diagnostic_visualization.png', dpi=150)
    print(f"\nVisualization saved to: diagnostic_visualization.png")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/memistd_small_target_config.yaml")
    parser.add_argument("--weights", type=str, default=None, help="Path to model weights")
    parser.add_argument("--visualize", action="store_true", help="Generate visualization")
    args = parser.parse_args()
    
    diagnose_model(args.config, args.weights)
    
    if args.visualize:
        visualize_predictions(args.config, args.weights)
