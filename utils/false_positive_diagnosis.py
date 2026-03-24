"""
误检问题诊断脚本
================

分析模型预测结果，诊断误检原因
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def analyze_predictions(predictions, ground_truths, conf_thres=0.25):
    """
    分析预测结果，统计误检情况
    
    Args:
        predictions: 模型预测结果列表
        ground_truths: 真实标签列表
        conf_thres: 置信度阈值
    """
    stats = {
        "total_predictions": 0,
        "total_gt": 0,
        "true_positives": 0,
        "false_positives": 0,
        "false_negatives": 0,
        "conf_distribution": [],
        "fp_by_conf": defaultdict(int),
    }
    
    for pred, gt in zip(predictions, ground_truths):
        pred_boxes = pred[pred[:, 4] > conf_thres]
        gt_boxes = gt
        
        stats["total_predictions"] += len(pred_boxes)
        stats["total_gt"] += len(gt_boxes)
        
        for p in pred_boxes:
            stats["conf_distribution"].append(p[4])
            conf_bin = int(p[4] * 10) / 10
            stats["fp_by_conf"][conf_bin] += 0
        
        if len(gt_boxes) == 0:
            stats["false_positives"] += len(pred_boxes)
            for p in pred_boxes:
                conf_bin = int(p[4] * 10) / 10
                stats["fp_by_conf"][conf_bin] += 1
        else:
            matched = [False] * len(gt_boxes)
            for p in pred_boxes:
                ious = compute_iou(p[:4], gt_boxes[:, :4])
                best_iou = ious.max() if len(ious) > 0 else 0
                best_idx = ious.argmax() if len(ious) > 0 else -1
                
                if best_iou > 0.5 and best_idx >= 0 and not matched[best_idx]:
                    stats["true_positives"] += 1
                    matched[best_idx] = True
                else:
                    stats["false_positives"] += 1
                    conf_bin = int(p[4] * 10) / 10
                    stats["fp_by_conf"][conf_bin] += 1
            
            stats["false_negatives"] += sum(not m for m in matched)
    
    if stats["total_predictions"] > 0:
        stats["precision"] = stats["true_positives"] / stats["total_predictions"]
    else:
        stats["precision"] = 0
    
    if stats["total_gt"] > 0:
        stats["recall"] = stats["true_positives"] / stats["total_gt"]
    else:
        stats["recall"] = 0
    
    return stats


def compute_iou(box, boxes):
    """计算IoU"""
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area1 = (box[2] - box[0]) * (box[3] - box[1])
    area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area1 + area2 - inter
    
    return inter / (union + 1e-6)


def plot_confidence_distribution(stats, save_path="conf_distribution.png"):
    """绘制置信度分布"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    ax = axes[0]
    ax.hist(stats["conf_distribution"], bins=20, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Count')
    ax.set_title('Prediction Confidence Distribution')
    ax.axvline(x=0.25, color='r', linestyle='--', label='conf_thres=0.25')
    ax.axvline(x=0.5, color='g', linestyle='--', label='conf_thres=0.5')
    ax.legend()
    
    ax = axes[1]
    conf_bins = sorted(stats["fp_by_conf"].keys())
    fp_counts = [stats["fp_by_conf"][b] for b in conf_bins]
    ax.bar(conf_bins, fp_counts, width=0.08, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Confidence Bin')
    ax.set_ylabel('False Positive Count')
    ax.set_title('False Positives by Confidence')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confidence distribution saved to: {save_path}")


def diagnose_model(model, dataloader, device, conf_thres=0.25, num_samples=100):
    """
    诊断模型误检问题
    
    Args:
        model: 模型
        dataloader: 数据加载器
        device: 设备
        conf_thres: 置信度阈值
        num_samples: 分析样本数
    """
    model.eval()
    
    all_predictions = []
    all_ground_truths = []
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_samples:
                break
            
            images = batch["image"].to(device)
            gt_boxes = batch.get("boxes", [])
            
            outputs = model(images)
            
            for j, output in enumerate(outputs):
                if isinstance(output, dict):
                    pred = output.get("detections", output.get("predictions", None))
                    if pred is None:
                        continue
                else:
                    pred = output
                
                if torch.is_tensor(pred):
                    pred = pred.cpu().numpy()
                
                all_predictions.append(pred)
                
                if j < len(gt_boxes):
                    gt = gt_boxes[j]
                    if torch.is_tensor(gt):
                        gt = gt.cpu().numpy()
                    all_ground_truths.append(gt)
                else:
                    all_ground_truths.append(np.array([]))
    
    stats = analyze_predictions(all_predictions, all_ground_truths, conf_thres)
    
    print("\n" + "=" * 60)
    print("误检问题诊断报告")
    print("=" * 60)
    print(f"\n样本数量: {len(all_predictions)}")
    print(f"置信度阈值: {conf_thres}")
    print(f"\n预测总数: {stats['total_predictions']}")
    print(f"真实目标总数: {stats['total_gt']}")
    print(f"\n真正例 (TP): {stats['true_positives']}")
    print(f"假正例 (FP/误检): {stats['false_positives']}")
    print(f"假负例 (FN/漏检): {stats['false_negatives']}")
    print(f"\n精确率 (Precision): {stats['precision']:.4f}")
    print(f"召回率 (Recall): {stats['recall']:.4f}")
    
    print("\n误检原因分析:")
    if stats['precision'] < 0.3:
        print("  ⚠️ 精确率过低，大量误检")
        print("     建议: 提高置信度阈值到 0.5 或继续训练")
    
    if stats['recall'] < 0.3:
        print("  ⚠️ 召回率过低，大量漏检")
        print("     建议: 降低置信度阈值或继续训练")
    
    if stats['false_positives'] > stats['true_positives'] * 2:
        print("  ⚠️ 误检数量远超正确检测")
        print("     建议: 检查记忆模块是否收敛，考虑增加正交性损失权重")
    
    print("\n置信度分布分析:")
    conf_dist = stats["conf_distribution"]
    if len(conf_dist) > 0:
        mean_conf = np.mean(conf_dist)
        std_conf = np.std(conf_dist)
        print(f"  平均置信度: {mean_conf:.4f}")
        print(f"  置信度标准差: {std_conf:.4f}")
        
        high_conf_count = sum(1 for c in conf_dist if c > 0.5)
        low_conf_count = sum(1 for c in conf_dist if c < 0.3)
        print(f"  高置信度预测 (>0.5): {high_conf_count} ({high_conf_count/len(conf_dist)*100:.1f}%)")
        print(f"  低置信度预测 (<0.3): {low_conf_count} ({low_conf_count/len(conf_dist)*100:.1f}%)")
    
    plot_confidence_distribution(stats)
    
    return stats


if __name__ == "__main__":
    print("误检问题诊断工具")
    print("=" * 60)
    print("\n使用方法:")
    print("  from utils.false_positive_diagnosis import diagnose_model")
    print("  stats = diagnose_model(model, val_loader, device, conf_thres=0.25)")
