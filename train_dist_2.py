"""
MemISTD Small Target Detection - Distributed Training Script
=============================================================

Multi-GPU distributed training using PyTorch DDP.

Usage:
    # 4 GPUs on single node
    torchrun --nproc_per_node=4 train_dist.py --config config/xxx.yaml
    
    # Multi-node training
    torchrun --nnodes=2 --nproc_per_node=4 --node_rank=0 \
        --master_addr="10.0.0.1" --master_port=29500 \
        train_dist.py --config config/xxx.yaml
"""

import os

# 设置可见的 GPU 卡号（在 import torch 之前设置）
# 只使用 0, 1, 3 三张卡
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,3"

import random
import numpy as np
import torch
from utils.eval_utils import evaluate_detection, print_evaluation_report
import pickle
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import argparse
import yaml
import logging
from datetime import datetime
import shutil
import json

from model_2.memistd_small_target import MemISTDSmallTarget
from model_2.yolox_loss_optimized import YOLOLossOptimized
from model_2.losses import ResidualReconstructionLoss
from MemISTD_Dataloader import IRDSTDataset, ITSDTDataset, NUDTMIRSDTDataset, irdst_collate_fn


# ============================================================================
# 显存优化：内联优化函数
# ============================================================================

class MemoryOptimizedTrainer:
    """显存优化的训练器"""
    
    def __init__(self, device=0):
        self.device = device
        self.peak_memory = 0
    
    def get_memory_stats(self):
        """获取显存统计"""
        if not torch.cuda.is_available():
            return {"allocated": 0, "reserved": 0, "peak": 0}
        
        allocated = torch.cuda.memory_allocated(self.device) / 1024 / 1024
        reserved = torch.cuda.memory_reserved(self.device) / 1024 / 1024
        self.peak_memory = max(self.peak_memory, allocated)
        
        return {
            "allocated": allocated,
            "reserved": reserved,
            "peak": self.peak_memory
        }
    
    def cleanup_memory(self):
        """清理显存"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self.device)


def setup_distributed():
    """Initialize distributed training environment."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        local_rank = int(os.environ["SLURM_LOCALID"])
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    if world_size > 1:
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
        )
        torch.cuda.set_device(local_rank)

    return local_rank, rank, world_size


def freeze_pretrained_modules(model):
    """冻结预训练模块的参数，只训练 FPN(neck) 和检测头

    冻结的模块：
    - backbone
    - feature_split
    - dual_memory
    - final_arithmetic
    - target_decoder
    - background_decoder

    训练的模块：
    - neck (FPN)
    - detection_head
    """
    frozen_count = 0
    trainable_count = 0

    # 只训练 FPN(neck) 和检测头
    # 支持 DDP 包装后的参数名称（带 module. 前缀）
    trainable_prefixes = ["neck.", "detection_head.",
                          "module.neck.", "module.detection_head."]

    for name, param in model.named_parameters():
        # 检查是否是需要训练的模块
        is_trainable = any(name.startswith(prefix)
                           for prefix in trainable_prefixes)

        if is_trainable:
            param.requires_grad = True
            trainable_count += 1
        else:
            param.requires_grad = False
            frozen_count += 1

    return frozen_count, trainable_count


def unfreeze_all(model):
    """解冻所有参数"""
    for param in model.parameters():
        param.requires_grad = True


def load_pretrained_weights(model, checkpoint_path, local_rank=0):
    """加载预训练模型的权重，排除检测头和已移除的模块

    加载的模块：
    - backbone
    - feature_split
    - dual_memory
    - final_arithmetic
    - target_decoder
    - background_decoder
    - neck (FPN，排除 p1_refine 和 p2_refine)

    不加载的模块：
    - detection_head
    - neck.p1_refine
    - neck.p2_refine
    """
    if is_main_process(local_rank):
        logger.info(f"Loading pretrained weights from: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    # 排除检测头和已移除的模块
    exclude_prefixes = [
        "detection_head.",
        "neck.p1_refine.",
        "neck.p2_refine.",
        "module.detection_head.",
        "module.neck.p1_refine.",
        "module.neck.p2_refine.",
    ]

    # 检查模型是否是 DDP 包装的（参数名带有 module. 前缀）
    model_state_dict = model.state_dict()
    has_module_prefix = any(k.startswith("module.")
                            for k in model_state_dict.keys())

    # 筛选需要加载的权重
    pretrained_state_dict = {}
    for k, v in state_dict.items():
        # 排除检测头和已移除的模块
        if any(k.startswith(prefix) for prefix in exclude_prefixes):
            continue

        # 根据模型是否是 DDP 包装，调整键名
        if has_module_prefix:
            # 模型是 DDP 包装，需要添加 module. 前缀
            new_key = f"module.{k}" if not k.startswith("module.") else k
        else:
            # 模型不是 DDP 包装，保持原键名
            new_key = k

        pretrained_state_dict[new_key] = v

    # 加载权重
    missing, unexpected = model.load_state_dict(
        pretrained_state_dict, strict=False)

    if is_main_process(local_rank):
        logger.info(
            f"Loaded {len(pretrained_state_dict)} pretrained parameters")
        logger.info(
            f"Excluded modules: detection_head, neck.p1_refine, neck.p2_refine")
        if missing:
            missing_list = list(missing)[:10]
            logger.info(f"Missing keys ({len(missing)}): {missing_list}...")
        if unexpected:
            unexpected_list = list(unexpected)[:10]
            logger.info(
                f"Unexpected keys ({len(unexpected)}): {unexpected_list}...")

    return len(pretrained_state_dict)


def cleanup_distributed():
    """Cleanup distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(local_rank):
    """Check if current process is main process"""
    return local_rank == 0


def reduce_dict(input_dict, average=True):
    """Reduce dictionary across all processes"""
    if not dist.is_initialized():
        return input_dict

    world_size = dist.get_world_size()
    if world_size < 2:
        return input_dict

    with torch.no_grad():
        names = []
        values = []
        for k, v in input_dict.items():
            names.append(k)
            values.append(v)

        values = torch.stack(values, dim=0)
        dist.all_reduce(values, op=dist.ReduceOp.SUM)

        if average:
            values /= world_size

        result = {k: v for k, v in zip(names, values)}
        return result


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class CheckpointManager:
    """Checkpoint Manager for saving and loading training states"""

    def __init__(self, root_dir="checkpoint", monitor="mAP", mode="max", local_rank=0, resume_dir=None):
        self.root_dir = root_dir
        self.monitor = monitor
        self.mode = mode
        self.local_rank = local_rank
        self.resume_dir = resume_dir

        if resume_dir:
            self.save_dir = resume_dir
            if is_main_process(local_rank):
                logger.info(f"Resume from checkpoint: {resume_dir}")
            checkpoint_path = os.path.join(resume_dir, "last.pth")
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(
                    checkpoint_path, map_location="cpu", weights_only=False)
                self.best_value = checkpoint.get("best_value", float(
                    "-inf") if mode == "max" else float("inf"))
                self.best_epoch = checkpoint.get("best_epoch", 0)
                self.start_epoch = checkpoint.get("epoch", 0)
                logger.info(
                    f"Loaded checkpoint from epoch {self.start_epoch}")
            else:
                logger.warning(
                    f"Checkpoint file not found: {checkpoint_path}, starting from scratch")
                self.best_value = float(
                    "-inf") if mode == "max" else float("inf")
                self.best_epoch = 0
                self.start_epoch = 0
        else:
            if is_main_process(local_rank):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.save_dir = os.path.join(root_dir, timestamp)
                os.makedirs(self.save_dir, exist_ok=True)

                if mode == "min":
                    self.best_value = float("inf")
                else:
                    self.best_value = float("-inf")

                self.best_epoch = 0
                logger.info(f"Checkpoint directory created: {self.save_dir}")
            else:
                self.save_dir = None
                self.best_value = float(
                    "-inf") if mode == "max" else float("inf")
                self.best_epoch = 0

            self.start_epoch = 0

    def save_config(self, config):
        if not is_main_process(self.local_rank):
            return
        config_path = os.path.join(self.save_dir, "config.yaml")
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
        logger.info(f"Config saved: {config_path}")

    def load_checkpoint(self, model, optimizer, lr_scheduler=None, scaler=None):
        if not self.resume_dir:
            return 0

        checkpoint_path = os.path.join(self.resume_dir, "last.pth")
        if not os.path.exists(checkpoint_path):
            if is_main_process(self.local_rank):
                logger.warning(f"Checkpoint file not found: {checkpoint_path}")
            return 0

        checkpoint = torch.load(
            checkpoint_path, map_location=f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu", weights_only=False)

        if hasattr(model, "module"):
            model.module.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint["model_state_dict"])

        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if lr_scheduler is not None and "lr_scheduler_state_dict" in checkpoint:
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])

        if scaler is not None and "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])

        start_epoch = checkpoint.get("epoch", 0)

        if is_main_process(self.local_rank):
            logger.info(
                f"Successfully loaded checkpoint from epoch {start_epoch}")
            if "loss" in checkpoint:
                logger.info(f"Previous loss: {checkpoint['loss']:.4f}")
            if "mAP" in checkpoint and checkpoint["mAP"] is not None:
                logger.info(f"Previous mAP: {checkpoint['mAP'] * 100:.2f}%")

        return start_epoch

    def save_checkpoint(self, epoch, model, optimizer, lr_scheduler=None, scaler=None, loss=None, mAP=None, is_last=False, save_every_epoch=True):
        if not is_main_process(self.local_rank):
            return False

        if hasattr(model, "module"):
            model_state_dict = model.module.state_dict()
        else:
            model_state_dict = model.state_dict()

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            "mAP": mAP,
            "best_value": self.best_value,
            "best_epoch": self.best_epoch,
        }

        if lr_scheduler is not None:
            checkpoint["lr_scheduler_state_dict"] = lr_scheduler.state_dict()

        if scaler is not None:
            checkpoint["scaler_state_dict"] = scaler.state_dict()

        last_path = os.path.join(self.save_dir, "last.pth")
        torch.save(checkpoint, last_path)

        if save_every_epoch:
            epoch_path = os.path.join(self.save_dir, f"epoch_{epoch:04d}.pth")
            torch.save(checkpoint, epoch_path)
            logger.info(f"Epoch checkpoint saved: {epoch_path}")

        is_best = False
        if mAP is not None:
            is_best = mAP > self.best_value
            if is_best:
                self.best_value = mAP
                self.best_epoch = epoch
                best_path = os.path.join(self.save_dir, "best.pth")
                torch.save(checkpoint, best_path)
                logger.info(
                    f"Best model saved! Epoch {epoch}, mAP={mAP * 100:.2f}%")

        if is_last:
            logger.info(f"Last epoch model saved: {last_path}")

        return is_best

    def save_training_log(self, log_data, append=False):
        if not is_main_process(self.local_rank):
            return
        log_path = os.path.join(self.save_dir, "training_log.json")

        if append and os.path.exists(log_path):
            with open(log_path, "r", encoding="utf-8") as f:
                existing_log = json.load(f)
            if "epochs" in existing_log and "epochs" in log_data:
                existing_log["epochs"].extend(log_data["epochs"])
                log_data = existing_log

        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)

    def get_save_dir(self):
        return self.save_dir

    def get_start_epoch(self):
        return getattr(self, 'start_epoch', 0)


class LossWeightScheduler:
    """Loss Weight Scheduler for staged training"""

    def __init__(
        self,
        warmup_epochs: int = 5,
        residual_recon_start_epoch: int = 1,
        residual_recon_weight: float = 1.0,
        residual_recon_rampup_epochs: int = 10,
    ):
        self.warmup_epochs = warmup_epochs
        self.residual_recon_start_epoch = residual_recon_start_epoch
        self.residual_recon_weight = residual_recon_weight
        self.residual_recon_rampup_epochs = residual_recon_rampup_epochs

    def get_weights(self, epoch: int) -> dict:
        if epoch < self.warmup_epochs:
            progress = min(1.0, max(0.0, (epoch - self.residual_recon_start_epoch + 1) /
                           self.residual_recon_rampup_epochs)) if epoch >= self.residual_recon_start_epoch else 0.0
            return {
                "residual_recon_weight": self.residual_recon_weight * progress,
                "stage": "warmup",
            }
        else:
            return {
                "residual_recon_weight": self.residual_recon_weight,
                "stage": "training",
            }


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate_model(
    model,
    test_dataset,
    class_names,
    input_shape,
    conf_thresh=0.05,
    nms_thresh=0.5,
    iou_thresh=0.5,
    cuda=True,
    max_detections=300,
    local_rank=0,
):
    """
    Evaluate model performance using model.detect().
    No OpenCV dependency - pure NumPy/PyTorch implementation.

    Returns:
        Dictionary with P, R, F1, AP50, FPS metrics
    """
    device = f"cuda:{local_rank}" if cuda else "cpu"

    # 显存优化：评估前清理显存，禁用梯度计算
    if cuda:
        torch.cuda.empty_cache()

    with torch.no_grad():
        metrics = evaluate_detection(
            model=model,
            test_dataset=test_dataset,
            conf_thresh=conf_thresh,
            nms_threshold=nms_thresh,
            iou_threshold=iou_thresh,
            max_detections=max_detections,
            device=device,
            warmup_iterations=3,
            local_rank=local_rank,
            disable_pbar=not is_main_process(local_rank),
        )

    if is_main_process(local_rank):
        print_evaluation_report(metrics)
        logger.info(
            f"Recall: {metrics['recall']*100:.2f}%, Precision: {metrics['precision']*100:.2f}%, F1: {metrics['f1']*100:.2f}%")
        logger.info(
            f"AP@0.5: {metrics['ap50']*100:.2f}%, FPS: {metrics['fps']:.2f}")

    return metrics


def evaluate_model_distributed(
    model,
    test_dataset,
    class_names,
    input_shape,
    conf_thresh=0.05,
    nms_thresh=0.5,
    iou_thresh=0.5,
    cuda=True,
    max_detections=300,
    local_rank=0,
    world_size=1,
    rank=0,
):
    """
    多卡并行评估：每张卡只推理测试集的一个子集，最后 all_gather 汇总统计量
    到 rank 0 计算最终指标。相比单卡评估，速度提升约 world_size 倍。

    Returns:
        rank 0 返回完整指标字典；其他 rank 返回 None。
    """
    from utils.eval_utils import compute_ap, compute_iou_single
    from MemISTD_Dataloader import irdst_collate_fn

    device = f"cuda:{local_rank}" if cuda else "cpu"

    # 切换到评估模式
    raw_model = model.module if hasattr(model, "module") else model
    raw_model.eval()

    if cuda:
        torch.cuda.empty_cache()

    # ---- 用 DistributedSampler 对测试集分片 --------------------------------
    test_sampler = DistributedSampler(
        test_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        sampler=test_sampler,
        num_workers=2,
        pin_memory=True,
        collate_fn=irdst_collate_fn,
        drop_last=False,
    )

    # ---- 本卡推理，收集本地统计量 ------------------------------------------
    local_scores   = []   # List[float]
    local_tp_flags = []   # List[int]  1=TP, 0=FP
    local_tp  = 0
    local_fp  = 0
    local_fn  = 0
    local_gt  = 0

    # FPS 计时 —— 统一用 wall-clock（ms），避免跨 if/else 的 UnboundLocalError
    import time as _time
    use_cuda_timing = cuda and torch.cuda.is_available()
    if use_cuda_timing:
        starter = torch.cuda.Event(enable_timing=True)
        ender   = torch.cuda.Event(enable_timing=True)
    total_timing_ms = 0.0   # 统一计时变量，始终有效
    n_images = 0

    # 所有卡都显示进度条，方便观察各卡实际推理进度
    pbar = tqdm(
        test_loader,
        desc=f"[Rank {rank}/{world_size}] Eval",
        disable=False,
        dynamic_ncols=True,
        position=rank,
        leave=(rank == 0),
    )

    with torch.no_grad():
        for images, targets_dict in pbar:
            images = images.to(device, non_blocking=True)
            bs = images.shape[0]  # always 1

            # -- timed forward --
            if use_cuda_timing:
                starter.record()
                det_batch = raw_model.detect(
                    images,
                    conf_thres=conf_thresh,
                    nms_thres=nms_thresh,
                    max_detections=max_detections,
                )
                ender.record()
                torch.cuda.synchronize()
                total_timing_ms += starter.elapsed_time(ender)
            else:
                t0 = _time.perf_counter()
                det_batch = raw_model.detect(
                    images,
                    conf_thres=conf_thresh,
                    nms_thres=nms_thresh,
                    max_detections=max_detections,
                )
                total_timing_ms += (_time.perf_counter() - t0) * 1000.0
            n_images += bs

            gt_boxes_batch = targets_dict.get("boxes", [])

            for i in range(bs):
                # -- parse detections --
                if det_batch is not None and i < len(det_batch):
                    arr = det_batch[i].cpu().numpy().astype("float64")
                    valid = arr[:, 4] >= max(conf_thresh, 1e-9)
                    det_i = arr[valid, :5]  # (M, 5) x1y1x2y2+score
                else:
                    det_i = np.zeros((0, 5), dtype=np.float64)

                # -- parse GT (xywh -> xyxy) --
                if i < len(gt_boxes_batch):
                    gt_raw = gt_boxes_batch[i]
                    if isinstance(gt_raw, torch.Tensor):
                        gt_raw = gt_raw.cpu().numpy()
                    gt_raw = np.asarray(gt_raw, dtype=np.float64)
                    if gt_raw.ndim == 1:
                        gt_raw = gt_raw.reshape(1, -1) if gt_raw.size > 0 else np.zeros((0, 4), dtype=np.float64)
                    if len(gt_raw) > 0 and gt_raw.shape[1] >= 4:
                        gt_boxes = np.stack([
                            gt_raw[:, 0],
                            gt_raw[:, 1],
                            gt_raw[:, 0] + gt_raw[:, 2],
                            gt_raw[:, 1] + gt_raw[:, 3],
                        ], axis=1)
                    else:
                        gt_boxes = np.zeros((0, 4), dtype=np.float64)
                else:
                    gt_boxes = np.zeros((0, 4), dtype=np.float64)

                num_gt = len(gt_boxes)
                local_gt += num_gt

                # -- greedy match --
                if len(det_i) == 0:
                    local_fn += num_gt
                    continue

                pred_boxes  = det_i[:, :4]
                pred_scores = det_i[:, 4]
                gt_matched  = np.zeros(num_gt, dtype=bool)
                order       = np.argsort(-pred_scores)

                for idx in order:
                    score = float(pred_scores[idx])
                    local_scores.append(score)
                    if num_gt > 0:
                        ious = compute_iou_single(pred_boxes[idx], gt_boxes)
                        ious[gt_matched] = -1.0
                        best_gt  = int(np.argmax(ious))
                        best_iou = float(ious[best_gt])
                    else:
                        best_gt  = -1
                        best_iou = 0.0

                    if best_iou >= iou_thresh:
                        gt_matched[best_gt] = True
                        local_tp += 1
                        local_tp_flags.append(1)
                    else:
                        local_fp += 1
                        local_tp_flags.append(0)

                local_fn += int(np.sum(~gt_matched))

    # ---- 同步等待所有卡完成推理 --------------------------------------------
    if dist.is_initialized():
        dist.barrier()

    # ---- all_gather 汇总各卡数据到所有进程 ---------------------------------
    local_stats = {
        "scores":   local_scores,
        "tp_flags": local_tp_flags,
        "tp":  local_tp,
        "fp":  local_fp,
        "fn":  local_fn,
        "gt":  local_gt,
        "n_images": n_images,
        "timing_ms": total_timing_ms,   # 统一计时变量，单位 ms
    }

    if dist.is_initialized() and world_size > 1:
        gathered = [None] * world_size
        dist.all_gather_object(gathered, local_stats)
    else:
        gathered = [local_stats]

    # ---- 只有 rank 0 计算最终指标 ------------------------------------------
    if not is_main_process(local_rank):
        # 恢复训练模式
        raw_model.train()
        return {"ap50": 0.0, "recall": 0.0, "precision": 0.0, "f1": 0.0,
                "fps": 0.0, "avg_inference_time_ms": 0.0, "total_images": 0,
                "total_gt": 0, "tp": 0, "fp": 0, "fn": 0}

    # -- 合并所有卡的结果 --
    all_scores   = []
    all_tp_flags = []
    total_tp = total_fp = total_fn = total_gt = total_images = 0
    # FPS：多卡并行推理时，实际延迟取各卡中最慢的那张（木桶效应）
    # timing_ms 记录的是本卡推理所有图片的累计耗时
    # avg_ms_per_card = timing_ms / n_images  -> 每张图在该卡上的平均延迟
    # 并行时墙钟延迟取所有卡 avg_ms_per_card 的最大值
    max_avg_ms_per_card = 0.0

    for stats in gathered:
        all_scores.extend(stats["scores"])
        all_tp_flags.extend(stats["tp_flags"])
        total_tp     += stats["tp"]
        total_fp     += stats["fp"]
        total_fn     += stats["fn"]
        total_gt     += stats["gt"]
        total_images += stats["n_images"]
        # 该卡每张图的平均推理延迟（木桶效应：取最慢卡）
        card_avg_ms = stats["timing_ms"] / max(stats["n_images"], 1)
        max_avg_ms_per_card = max(max_avg_ms_per_card, card_avg_ms)
        logger.info(
            f"  [Rank stats] n={stats['n_images']}, tp={stats['tp']}, "
            f"fp={stats['fp']}, fn={stats['fn']}, gt={stats['gt']}, "
            f"avg_ms={card_avg_ms:.1f}"
        )

    # -- 计算指标 --
    recall    = total_tp / max(total_gt, 1)
    precision = total_tp / max(total_tp + total_fp, 1)
    f1        = 2.0 * precision * recall / max(precision + recall, 1e-9)
    ap50      = compute_ap(all_scores, all_tp_flags, total_gt)

    # FPS = 1000 / 最慢卡的单张平均延迟（反映真实推理吞吐）
    avg_ms = max_avg_ms_per_card
    fps    = 1000.0 / max(avg_ms, 1e-9)

    metrics = {
        "recall":    recall,
        "precision": precision,
        "f1":        f1,
        "ap50":      ap50,
        "fps":       fps,
        "avg_inference_time_ms": avg_ms,
        "total_images": total_images,
        "total_gt":  total_gt,
        "tp":  total_tp,
        "fp":  total_fp,
        "fn":  total_fn,
    }

    print_evaluation_report(metrics)
    logger.info(
        f"[Distributed Eval x{world_size} GPUs] "
        f"Recall: {recall*100:.2f}%, Precision: {precision*100:.2f}%, "
        f"F1: {f1*100:.2f}%, AP@0.5: {ap50*100:.2f}%, FPS: {fps:.2f}"
    )

    # 恢复训练模式
    raw_model.train()
    return metrics


def fit_one_epoch(
    model,
    yolo_loss,
    residual_recon_loss,
    optimizer,
    epoch,
    epoch_step,
    train_loader,
    Epoch,
    cuda,
    fp16,
    scaler,
    loss_weight_scheduler=None,
    local_rank=0,
    accumulation_steps=1,
):
    """Training loop for one epoch"""
    total_loss = 0
    total_yolo_loss = 0
    total_loss_box = 0
    total_loss_obj = 0
    total_loss_cls = 0
    total_residual_recon_loss = 0
    total_loss_global = 0
    total_loss_tgt_sparse = 0
    total_loss_tgt_content = 0
    total_loss_bg_inpaint = 0

    if loss_weight_scheduler is not None:
        weights = loss_weight_scheduler.get_weights(epoch)
        residual_recon_weight = weights["residual_recon_weight"]
        stage = weights["stage"]
    else:
        residual_recon_weight = 1.0
        stage = "default"

    model.train()
    if is_main_process(local_rank):
        logger.info(
            f"Epoch {epoch + 1}/{Epoch} - Stage: {stage} | ResidualRecon Weight: {residual_recon_weight:.4f}")
        pbar = tqdm(total=epoch_step, desc=f"Epoch {epoch + 1}/{Epoch}",
                    postfix=dict, mininterval=0.3, dynamic_ncols=True)

    train_loader.sampler.set_epoch(epoch)

    for iteration, batch in enumerate(train_loader):
        if iteration >= epoch_step:
            break

        images, targets_dict = batch
        with torch.no_grad():
            if cuda:
                images = images.cuda(local_rank)

                boxes_list = targets_dict["boxes"]
                labels_list = targets_dict["labels"]

                yolo_targets = []
                for boxes, labels in zip(boxes_list, labels_list):
                    if len(boxes) > 0:
                        boxes = boxes.cuda(local_rank)
                        labels = labels.cuda(local_rank)
                        cx = boxes[:, 0] + boxes[:, 2] / 2
                        cy = boxes[:, 1] + boxes[:, 3] / 2
                        w = boxes[:, 2]
                        h = boxes[:, 3]
                        target = torch.stack(
                            [cx, cy, w, h, labels.float()], dim=1)
                        yolo_targets.append(target)
                    else:
                        yolo_targets.append(torch.zeros(0, 5).cuda(local_rank))

                target_mask = targets_dict["target_mask"]
                if cuda:
                    target_mask = target_mask.cuda(local_rank)

        if iteration % accumulation_steps == 0:
            optimizer.zero_grad()

        if fp16:
            with torch.amp.autocast('cuda'):
                outputs = model(images, decode_image=True)
                outputs["target_mask"] = target_mask

                loss_dict = model.module.compute_loss(
                    outputs=outputs,
                    labels=yolo_targets,
                    yolox_loss=yolo_loss,
                    residual_recon_loss=residual_recon_loss,
                    residual_recon_weight=residual_recon_weight,
                )
                loss = loss_dict["total_loss"]
                loss_yolo = loss_dict["yolo_loss"]
                loss_box = loss_dict["loss_box"]
                loss_obj = loss_dict["loss_obj"]
                loss_cls = loss_dict["loss_cls"]
                loss_residual_recon = loss_dict["residual_recon_loss"]
                loss_global = loss_dict["loss_global"]
                loss_tgt_sparse = loss_dict["loss_tgt_sparse"]
                loss_tgt_content = loss_dict["loss_tgt_content"]
                loss_bg_inpaint = loss_dict["loss_bg_inpaint"]

                if torch.isnan(loss) or torch.isinf(loss):
                    if is_main_process(local_rank):
                        logger.warning(
                            f"NaN/Inf detected in loss at iteration {iteration}!")
                        logger.warning(
                            f"  yolo_loss: {loss_yolo.item() if torch.is_tensor(loss_yolo) else loss_yolo}")
                        logger.warning(
                            f"  residual_recon_loss: {loss_residual_recon.item() if torch.is_tensor(loss_residual_recon) else loss_residual_recon}")
                    del outputs, loss_dict
                    continue

                del outputs, loss_dict
        else:
            outputs = model(images, decode_image=True)
            outputs["target_mask"] = target_mask

            loss_dict = model.module.compute_loss(
                outputs=outputs,
                labels=yolo_targets,
                yolox_loss=yolo_loss,
                residual_recon_loss=residual_recon_loss,
                residual_recon_weight=residual_recon_weight,
            )
            loss = loss_dict["total_loss"]
            loss_yolo = loss_dict["yolo_loss"]
            loss_box = loss_dict["loss_box"]
            loss_obj = loss_dict["loss_obj"]
            loss_cls = loss_dict["loss_cls"]
            loss_residual_recon = loss_dict["residual_recon_loss"]
            loss_global = loss_dict["loss_global"]
            loss_tgt_sparse = loss_dict["loss_tgt_sparse"]
            loss_tgt_content = loss_dict["loss_tgt_content"]
            loss_bg_inpaint = loss_dict["loss_bg_inpaint"]

            if torch.isnan(loss) or torch.isinf(loss):
                if is_main_process(local_rank):
                    logger.warning(
                        f"NaN/Inf detected in loss at iteration {iteration}!")
                    logger.warning(
                        f"  yolo_loss: {loss_yolo.item() if torch.is_tensor(loss_yolo) else loss_yolo}")
                    logger.warning(
                        f"  residual_recon_loss: {loss_residual_recon.item() if torch.is_tensor(loss_residual_recon) else loss_residual_recon}")
                del outputs, loss_dict
                continue

            del outputs, loss_dict

        loss = loss / accumulation_steps

        if fp16:  # 计算梯度
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (iteration + 1) % accumulation_steps == 0:  # 更新参数
            if fp16:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=10.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=10.0)
                optimizer.step()

        total_loss += loss.item() * accumulation_steps
        total_yolo_loss += loss_yolo.item()
        total_loss_box += loss_box.item() if isinstance(loss_box, torch.Tensor) else 0
        total_loss_obj += loss_obj.item() if isinstance(loss_obj, torch.Tensor) else 0
        total_loss_cls += loss_cls.item() if isinstance(loss_cls, torch.Tensor) else 0
        total_residual_recon_loss += loss_residual_recon.item() if isinstance(loss_residual_recon,
                                                                              torch.Tensor) else 0
        total_loss_global += loss_global.item() if isinstance(loss_global,
                                                              torch.Tensor) else 0
        total_loss_tgt_sparse += loss_tgt_sparse.item() if isinstance(loss_tgt_sparse,
                                                                      torch.Tensor) else 0
        total_loss_tgt_content += loss_tgt_content.item() if isinstance(loss_tgt_content,
                                                                        torch.Tensor) else 0
        total_loss_bg_inpaint += loss_bg_inpaint.item() if isinstance(loss_bg_inpaint,
                                                                      torch.Tensor) else 0

        if is_main_process(local_rank):
            pbar.set_postfix(
                **{
                    "lr": optimizer.param_groups[0]["lr"],
                    "acc": f"{iteration % accumulation_steps + 1}/{accumulation_steps}",
                    "total": total_loss / (iteration + 1),
                    "box": total_loss_box / (iteration + 1),
                    "obj": total_loss_obj / (iteration + 1),
                    "cls": total_loss_cls / (iteration + 1),
                    "yolo": total_yolo_loss / (iteration + 1),
                    "res_recon": total_residual_recon_loss / (iteration + 1),
                    "global": total_loss_global / (iteration + 1),
                    "tgt_sparse": total_loss_tgt_sparse / (iteration + 1),
                    "tgt_content": total_loss_tgt_content / (iteration + 1),
                    "bg_inpaint": total_loss_bg_inpaint / (iteration + 1),
                }
            )
            pbar.update(1)

        # 显存优化：每 50 个 iteration 清理一次显存碎片
        if cuda and (iteration + 1) % 50 == 0:
            torch.cuda.empty_cache()

    if is_main_process(local_rank):
        pbar.close()
        logger.info(f"Finish Epoch {epoch + 1}/{Epoch}")

    # 防止 epoch_step=0 导致 ZeroDivisionError（数据集为空或 DataLoader drop_last 截断后为空）
    if epoch_step == 0:
        raise RuntimeError(
            f"[FATAL] epoch_step=0 in fit_one_epoch (Epoch {epoch + 1}). "
            f"train_loader is empty — check dataset path, dataset size, batch_size, "
            f"and drop_last settings."
        )

    if is_main_process(local_rank):
        logger.info(f"Total Loss: {total_loss / epoch_step:.4f}")

    return {
        "total_loss": total_loss / epoch_step,
        "yolo_loss": total_yolo_loss / epoch_step,
        "loss_box": total_loss_box / epoch_step,
        "loss_obj": total_loss_obj / epoch_step,
        "loss_cls": total_loss_cls / epoch_step,
        "residual_recon_loss": total_residual_recon_loss / epoch_step,
        "loss_global": total_loss_global / epoch_step,
        "loss_tgt_sparse": total_loss_tgt_sparse / epoch_step,
        "loss_tgt_content": total_loss_tgt_content / epoch_step,
        "loss_bg_inpaint": total_loss_bg_inpaint / epoch_step,
    }


def train_distributed(config_path="config/memistd_small_target_config.yaml", resume_dir=None):
    """Main distributed training function"""
    local_rank, rank, world_size = setup_distributed()

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed_everything(cfg.get("seed", 42) + rank)

    cuda = torch.cuda.is_available()

    checkpoint_manager = CheckpointManager(
        root_dir="checkpoint",
        monitor="mAP",
        mode="max",
        local_rank=local_rank,
        resume_dir=resume_dir,
    )

    if not resume_dir:
        checkpoint_manager.save_config(cfg)

    if is_main_process(local_rank):
        logger.info("Creating model...")

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
        global_fusion_strategy=cfg["model"].get(
            "global_fusion_strategy", "spatial_reweight"),
        bg_patch_sizes=cfg["model"].get("bg_patch_sizes", [3, 5, 7]),
        tg_patch_sizes=cfg["model"].get("tg_patch_sizes", [3, 5, 7]),
    )

    if cuda:
        model = model.cuda(local_rank)
        model = DDP(model, device_ids=[
                    local_rank], output_device=local_rank, find_unused_parameters=True)

    # 创建优化器和学习率调度器（需要在 resume_dir 分支之前定义）
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"].get("weight_decay", 0.01),
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["training"]["epochs"], eta_min=cfg["training"]["lr"] * 0.01
    )

    # 创建 GradScaler（需要在 resume_dir 分支之前定义）
    use_fp16 = bool(cfg["training"].get("fp16", False))
    scaler = GradScaler('cuda') if use_fp16 and cuda else None

    # 恢复训练或从头训练
    if resume_dir:
        # 恢复训练：加载完整 checkpoint
        if is_main_process(local_rank):
            logger.info("Resuming training from checkpoint...")
        start_epoch = checkpoint_manager.load_checkpoint(
            model, optimizer, lr_scheduler, scaler)
    else:
        # 新训练：从头开始
        start_epoch = 0
        if is_main_process(local_rank):
            logger.info("Starting training from scratch...")

    if is_main_process(local_rank):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel()
                               for p in model.parameters() if p.requires_grad)
        logger.info(
            f"Model Parameters: {total_params / 1e6:.2f}M (trainable: {trainable_params / 1e6:.2f}M)")
        logger.info(f"World Size: {world_size} GPUs")

    yolox_cfg = cfg.get("yolox_loss", {})
    yolo_loss = YOLOLossOptimized(
        num_classes=cfg["model"]["num_classes"],
        fp16=cfg["training"]["fp16"],
        strides=yolox_cfg.get("strides", [1]),  # 单尺度检测，只有 P0
        center_radius=yolox_cfg.get("center_radius", 5.0),
        use_focal=yolox_cfg.get("use_focal", True),
        focal_alpha=yolox_cfg.get("focal_alpha", 0.25),
        focal_gamma=yolox_cfg.get("focal_gamma", 2.0),
        adaptive_center_radius=yolox_cfg.get("adaptive_center_radius", True),
    )

    residual_recon_cfg = cfg.get("residual_recon_loss", {})
    residual_recon_loss = ResidualReconstructionLoss(
        alpha=residual_recon_cfg.get("alpha", 1.0),
        beta=residual_recon_cfg.get("beta", 1.0),
        gamma=residual_recon_cfg.get("gamma", 0.5),
    )

    if is_main_process(local_rank):
        logger.info("Loading dataset...")

    input_shape = cfg["data"]["input_shape"]

    # ------------------------------------------------------------------
    # 数据集工厂：通过配置文件中 data.dataset_type 选择数据集类
    # 支持: "irdst"（默认）| "itsdt"
    # ------------------------------------------------------------------
    def build_dataset(split, augment, pad_to_size_key):
        dataset_type = cfg["data"].get("dataset_type", "irdst").lower()
        common_kwargs = dict(
            image_size       = input_shape[0],
            type             = split,
            augment          = augment,
            crop_size        = cfg["data"].get("crop_size", 480),
            base_size        = cfg["data"].get("base_size", 512),
            pad_only         = cfg["data"].get("pad_only", True),
            pad_to_size      = cfg["data"].get(pad_to_size_key, input_shape),
            pad_divisor      = cfg["data"].get("pad_divisor", 32),
            flip_augmentation  = cfg["data"].get("flip_augmentation", True) if augment else False,
            vflip_augmentation = cfg["data"].get("vflip_augmentation", False) if augment else False,
            brightness_jitter  = cfg["data"].get("brightness_jitter", 0.1) if augment else 0.0,
            contrast_jitter    = cfg["data"].get("contrast_jitter",   0.1) if augment else 0.0,
            gaussian_noise     = cfg["data"].get("gaussian_noise",    0.005) if augment else 0.0,
            scale_jitter       = cfg["data"].get("scale_jitter",      0.0),
        )
        if dataset_type == "itsdt":
            return ITSDTDataset(
                dataset_root = cfg["data"]["dataset_root"],
                **common_kwargs,
            )
        elif dataset_type == "nudt":
            return NUDTMIRSDTDataset(
                dataset_root = cfg["data"]["dataset_root"],
                **common_kwargs,
            )
        else:  # 默认 irdst
            return IRDSTDataset(
                dataset_root = cfg["data"]["dataset_root"],
                **common_kwargs,
            )

    train_dataset = build_dataset("train", augment=cfg["data"].get("data_augment", True),
                                  pad_to_size_key="pad_to_size")
    test_dataset  = build_dataset("test",  augment=False,
                                  pad_to_size_key="eval_pad_to_size")

    if is_main_process(local_rank):
        logger.info(
            f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
        logger.info(f"Dataset root: {cfg['data']['dataset_root']} (abs: {os.path.abspath(cfg['data']['dataset_root'])})")

    # 早期检查：若训练集为空则立刻报错，避免后续 ZeroDivisionError 掩盖真正原因
    if len(train_dataset) == 0:
        raise RuntimeError(
            f"[FATAL] Train dataset is EMPTY!\n"
            f"  dataset_root = {cfg['data']['dataset_root']}\n"
            f"  abs path     = {os.path.abspath(cfg['data']['dataset_root'])}\n"
            f"  dataset_type = {cfg['data'].get('dataset_type', 'irdst')}\n"
            f"Please check that the dataset path is correct and the directory is mounted/accessible."
        )

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
        sampler=train_sampler,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=True,
        drop_last=True,
        collate_fn=irdst_collate_fn,
    )

    save_dir = checkpoint_manager.get_save_dir()
    if is_main_process(local_rank):
        os.makedirs(save_dir, exist_ok=True)

    loss_scheduler_cfg = cfg.get("loss_scheduler", {})
    loss_weight_scheduler = LossWeightScheduler(
        warmup_epochs=loss_scheduler_cfg.get("warmup_epochs", 5),
        residual_recon_start_epoch=loss_scheduler_cfg.get(
            "residual_recon_start_epoch", 1),
        residual_recon_weight=loss_scheduler_cfg.get(
            "residual_recon_weight", 1.0),
        residual_recon_rampup_epochs=loss_scheduler_cfg.get(
            "residual_recon_rampup_epochs", 10),
    )

    training_log = {"config": cfg, "epochs": []}

    class_names_list = ["target"]
    eval_conf_thresh = cfg.get("eval", {}).get("conf_thresh", 0.27)
    eval_nms_thresh = cfg.get("eval", {}).get("nms_thresh", 0.5)
    eval_period = cfg.get("eval", {}).get("eval_period", 1)

    if is_main_process(local_rank):
        logger.info(f"Evaluation period: every {eval_period} epochs")
        logger.info("Start training...")

    # start_epoch 已在上方 resume/从头训练分支中正确赋值，此处不再重复 load_checkpoint
    accumulation_steps = cfg["training"].get("accumulation_steps", 1)
    if is_main_process(local_rank):
        logger.info(f"Gradient Accumulation Steps: {accumulation_steps}")
        logger.info(
            f"Effective Batch Size: {world_size * cfg['training']['batch_size'] * accumulation_steps}")

    # 显存优化：创建显存监控器
    memory_monitor = MemoryOptimizedTrainer(device=local_rank)
    #################################################################
    eval_results = evaluate_model_distributed(
        model=model,
        test_dataset=test_dataset,
        class_names=class_names_list,
        input_shape=input_shape,
        conf_thresh=eval_conf_thresh,
        nms_thresh=eval_nms_thresh,
        iou_thresh=cfg["eval"].get("iou_thresh", 0.5),
        cuda=cuda,
        local_rank=local_rank,
        world_size=world_size,
        rank=rank,
    )

    # 显存优化：评估后立即清理显存
    if cuda:
        torch.cuda.empty_cache()

    # 将 rank 0 计算的 ap50 广播给所有进程，确保各卡保存相同的 best.pth
    ap50_container = [eval_results["ap50"]]
    if dist.is_initialized():
        dist.broadcast_object_list(ap50_container, src=0)
    current_mAP = ap50_container[0]
    #################################################################
    for epoch in range(start_epoch, cfg["training"]["epochs"]):
        # 显存优化：每个 epoch 开始前清理显存碎片
        memory_monitor.cleanup_memory()

        loss_dict = fit_one_epoch(
            model=model,
            yolo_loss=yolo_loss,
            residual_recon_loss=residual_recon_loss,
            optimizer=optimizer,
            epoch=epoch,
            epoch_step=len(train_loader),
            train_loader=train_loader,
            Epoch=cfg["training"]["epochs"],
            cuda=cuda,
            fp16=cfg["training"]["fp16"],
            scaler=scaler,
            loss_weight_scheduler=loss_weight_scheduler,
            local_rank=local_rank,
            accumulation_steps=accumulation_steps,
        )

        current_mAP = 0.00
        if (epoch + 1) % eval_period == 0 or (epoch == cfg["training"]["epochs"] - 1):
            if is_main_process(local_rank):
                logger.info(f"\n{'=' * 60}")
                logger.info(
                    f"Epoch {epoch + 1}/{cfg['training']['epochs']} - Starting evaluation")
                logger.info(f"{'=' * 60}")

            eval_results = evaluate_model_distributed(
                model=model,
                test_dataset=test_dataset,
                class_names=class_names_list,
                input_shape=input_shape,
                conf_thresh=eval_conf_thresh,
                nms_thresh=eval_nms_thresh,
                iou_thresh=cfg["eval"].get("iou_thresh", 0.5),
                cuda=cuda,
                local_rank=local_rank,
                world_size=world_size,
                rank=rank,
            )

            # 显存优化：评估后立即清理显存
            if cuda:
                torch.cuda.empty_cache()

            # 将 rank 0 计算的 ap50 广播给所有进程，确保各卡保存相同的 best.pth
            ap50_container = [eval_results["ap50"]]
            if dist.is_initialized():
                dist.broadcast_object_list(ap50_container, src=0)
            current_mAP = ap50_container[0]

            if is_main_process(local_rank):
                logger.info(
                    f"Epoch {epoch + 1} evaluation complete: AP@0.5 = {current_mAP * 100:.2f}%")

        if is_main_process(local_rank):
            # 显存优化：记录显存统计
            mem_stats = memory_monitor.get_memory_stats()
            logger.info(
                f"Epoch {epoch + 1} Memory Stats: "
                f"Allocated={mem_stats['allocated']:.1f}MB, "
                f"Reserved={mem_stats['reserved']:.1f}MB, "
                f"Peak={mem_stats['peak']:.1f}MB"
            )

            epoch_info = {
                "epoch": epoch + 1,
                "loss": loss_dict["total_loss"],
                "yolo_loss": loss_dict["yolo_loss"],
                "loss_box": loss_dict["loss_box"],
                "loss_obj": loss_dict["loss_obj"],
                "loss_cls": loss_dict["loss_cls"],
                "residual_recon_loss": loss_dict["residual_recon_loss"],
                "loss_global": loss_dict["loss_global"],
                "loss_tgt_sparse": loss_dict["loss_tgt_sparse"],
                "loss_tgt_content": loss_dict["loss_tgt_content"],
                "loss_bg_inpaint": loss_dict["loss_bg_inpaint"],
                "lr": optimizer.param_groups[0]["lr"],
            }
            training_log["epochs"].append(epoch_info)

        is_last = (epoch == cfg["training"]["epochs"] - 1)
        save_every_epoch = cfg["training"].get("save_every_epoch", True)
        checkpoint_manager.save_checkpoint(
            epoch=epoch + 1,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            scaler=scaler,
            loss=loss_dict["total_loss"],
            mAP=current_mAP,
            is_last=is_last,
            save_every_epoch=save_every_epoch,
        )

        checkpoint_manager.save_training_log(
            training_log, append=(start_epoch > 0))
        lr_scheduler.step()

    if is_main_process(local_rank):
        logger.info("Training complete!")
        logger.info(f"All training files saved to: {save_dir}")

    cleanup_distributed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--config", type=str, default="./config/memistd_small_target_config_v2.yaml", help="Path to config file")
    # parser.add_argument("--resume", type=str, default="./checkpoint/20260312_154828", help="Resume from checkpoint directory (e.g., checkpoint/20260215_123456)")
    # parser.add_argument("--config", type=str, default="./config/memistd_itsdt_config.yaml", help="Path to config file")
    # parser.add_argument("--resume", type=str, default="./checkpoint/20260318_163321", help="Resume from checkpoint directory (e.g., checkpoint/20260215_123456)")
    parser.add_argument("--config", type=str, default="./config/memistd_nudt_config.yaml", help="Path to config file")
    parser.add_argument("--resume", type=str, default="./checkpoint/20260320_195725", help="Resume from checkpoint directory (e.g., checkpoint/20260215_123456)")
    # parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint directory (e.g., checkpoint/20260215_123456)")
    args = parser.parse_args()

    train_distributed(args.config, args.resume)
