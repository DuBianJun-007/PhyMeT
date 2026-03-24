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

from model_1.memistd_small_target import MemISTDSmallTarget
from model_1.yolox_loss_optimized import YOLOLossOptimized
from model_1.losses import ResidualReconstructionLoss
from MemISTD_Dataloader import IRDSTDataset, irdst_collate_fn


def setup_distributed():
    """Initialize distributed training"""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        local_rank = int(os.environ["SLURM_LOCALID"])
    else:
        rank = -1
        world_size = -1
        local_rank = 0
    
    if rank != -1:
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
        )
        torch.cuda.set_device(local_rank)
    
    return local_rank, rank, world_size


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
                    checkpoint = torch.load(checkpoint_path, map_location="cpu")
                    self.best_value = checkpoint.get("best_value", float("-inf") if mode == "max" else float("inf"))
                    self.best_epoch = checkpoint.get("best_epoch", 0)
                    self.start_epoch = checkpoint.get("epoch", 0)
                    logger.info(f"Loaded checkpoint from epoch {self.start_epoch}")
                else:
                    logger.warning(f"Checkpoint file not found: {checkpoint_path}, starting from scratch")
                    self.best_value = float("-inf") if mode == "max" else float("inf")
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
                self.best_value = float("-inf") if mode == "max" else float("inf")
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
        
        checkpoint = torch.load(checkpoint_path, map_location=f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu")
        
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
            logger.info(f"Successfully loaded checkpoint from epoch {start_epoch}")
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
                logger.info(f"Best model saved! Epoch {epoch}, mAP={mAP * 100:.2f}%")
        
        if is_last:
            logger.info(f"Last epoch model saved: {last_path}")
        
        return is_best
    
    def save_training_log(self, log_data):
        if not is_main_process(self.local_rank):
            return
        log_path = os.path.join(self.save_dir, "training_log.json")
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
            progress = min(1.0, max(0.0, (epoch - self.residual_recon_start_epoch + 1) / self.residual_recon_rampup_epochs)) if epoch >= self.residual_recon_start_epoch else 0.0
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
    
    metrics = evaluate_detection(
        model=model,
        test_dataset=test_dataset,
        conf_threshold=conf_thresh,
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
        logger.info(f"Recall: {metrics['recall']*100:.2f}%, Precision: {metrics['precision']*100:.2f}%, F1: {metrics['f1']*100:.2f}%")
        logger.info(f"AP@0.5: {metrics['ap50']*100:.2f}%, FPS: {metrics['fps']:.2f}")
    
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
    total_loss_bg_smooth = 0
    total_loss_tgt_content = 0
    
    if loss_weight_scheduler is not None:
        weights = loss_weight_scheduler.get_weights(epoch)
        residual_recon_weight = weights["residual_recon_weight"]
        stage = weights["stage"]
    else:
        residual_recon_weight = 1.0
        stage = "default"
    
    model.train()
    if is_main_process(local_rank):
        logger.info(f"Epoch {epoch + 1}/{Epoch} - Stage: {stage} | ResidualRecon Weight: {residual_recon_weight:.4f}")
        pbar = tqdm(total=epoch_step, desc=f"Epoch {epoch + 1}/{Epoch}", postfix=dict, mininterval=0.3, dynamic_ncols=True)
    
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
                        target = torch.stack([cx, cy, w, h, labels.float()], dim=1)
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
                loss_bg_smooth = loss_dict["loss_bg_smooth"]
                loss_tgt_content = loss_dict["loss_tgt_content"]
                
                if torch.isnan(loss) or torch.isinf(loss):
                    if is_main_process(local_rank):
                        logger.warning(f"NaN/Inf detected in loss at iteration {iteration}!")
                        logger.warning(f"  yolo_loss: {loss_yolo.item() if torch.is_tensor(loss_yolo) else loss_yolo}")
                        logger.warning(f"  residual_recon_loss: {loss_residual_recon.item() if torch.is_tensor(loss_residual_recon) else loss_residual_recon}")
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
            loss_bg_smooth = loss_dict["loss_bg_smooth"]
            loss_tgt_content = loss_dict["loss_tgt_content"]
            
            if torch.isnan(loss) or torch.isinf(loss):
                if is_main_process(local_rank):
                    logger.warning(f"NaN/Inf detected in loss at iteration {iteration}!")
                    logger.warning(f"  yolo_loss: {loss_yolo.item() if torch.is_tensor(loss_yolo) else loss_yolo}")
                    logger.warning(f"  residual_recon_loss: {loss_residual_recon.item() if torch.is_tensor(loss_residual_recon) else loss_residual_recon}")
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimizer.step()
        
        total_loss += loss.item() * accumulation_steps
        total_yolo_loss += loss_yolo.item()
        total_loss_box += loss_box.item() if isinstance(loss_box, torch.Tensor) else 0
        total_loss_obj += loss_obj.item() if isinstance(loss_obj, torch.Tensor) else 0
        total_loss_cls += loss_cls.item() if isinstance(loss_cls, torch.Tensor) else 0
        total_residual_recon_loss += loss_residual_recon.item() if isinstance(loss_residual_recon, torch.Tensor) else 0
        total_loss_global += loss_global.item() if isinstance(loss_global, torch.Tensor) else 0
        total_loss_tgt_sparse += loss_tgt_sparse.item() if isinstance(loss_tgt_sparse, torch.Tensor) else 0
        total_loss_bg_smooth += loss_bg_smooth.item() if isinstance(loss_bg_smooth, torch.Tensor) else 0
        total_loss_tgt_content += loss_tgt_content.item() if isinstance(loss_tgt_content, torch.Tensor) else 0
        
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
                    "bg_smooth": total_loss_bg_smooth / (iteration + 1),
                    "tgt_content": total_loss_tgt_content / (iteration + 1),
                }
            )
            pbar.update(1)
    
    if is_main_process(local_rank):
        pbar.close()
        logger.info(f"Finish Epoch {epoch + 1}/{Epoch}")
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
        "loss_bg_smooth": total_loss_bg_smooth / epoch_step,
        "loss_tgt_content": total_loss_tgt_content / epoch_step,
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
        global_fusion_strategy=cfg["model"].get("global_fusion_strategy", "spatial_reweight"),
        bg_patch_sizes=cfg["model"].get("bg_patch_sizes", [3, 5]),
        tg_patch_sizes=cfg["model"].get("tg_patch_sizes", [3, 5]),
    )
    
    if cuda:
        model = model.cuda(local_rank)
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    
    if is_main_process(local_rank):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model Parameters: {total_params / 1e6:.2f}M (trainable: {trainable_params / 1e6:.2f}M)")
        logger.info(f"World Size: {world_size} GPUs")
    
    yolox_cfg = cfg.get("yolox_loss", {})
    yolo_loss = YOLOLossOptimized(
        num_classes=cfg["model"]["num_classes"],
        fp16=cfg["training"]["fp16"],
        strides=yolox_cfg.get("strides", [1, 2, 4]),
        center_radius=yolox_cfg.get("center_radius", 5.0),
        use_focal=yolox_cfg.get("use_focal", True),
        focal_alpha=yolox_cfg.get("focal_alpha", 0.25),
        focal_gamma=yolox_cfg.get("focal_gamma", 2.0),
        adaptive_center_radius=yolox_cfg.get("adaptive_center_radius", True),
    )
    
    residual_recon_cfg = cfg.get("residual_recon_loss", {})
    residual_recon_loss = ResidualReconstructionLoss(
        alpha=residual_recon_cfg.get("alpha", 1.0),
        beta=residual_recon_cfg.get("beta", 0.1),
        gamma=residual_recon_cfg.get("gamma", 1.0),
    )
    
    optimizer = optim.Adam(model.parameters(), lr=cfg["training"]["lr"])
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["training"]["epochs"], eta_min=cfg["training"]["lr"] * 0.01
    )
    
    if is_main_process(local_rank):
        logger.info("Loading dataset...")
    
    input_shape = cfg["data"]["input_shape"]
    
    train_dataset = IRDSTDataset(
        dataset_root=cfg["data"]["dataset_root"],
        image_size=input_shape[0],
        type="train",
        augment=cfg["data"].get("data_augment", True),
        crop_size=cfg["data"].get("crop_size", 480),
        base_size=cfg["data"].get("base_size", 512),
        pad_only=cfg["data"].get("pad_only", True),
        pad_to_size=cfg["data"].get("pad_to_size", input_shape),
        pad_divisor=cfg["data"].get("pad_divisor", 32),
        flip_augmentation=cfg["data"].get("flip_augmentation", True),
        vflip_augmentation=cfg["data"].get("vflip_augmentation", False),
        brightness_jitter=cfg["data"].get("brightness_jitter", 0.1),
        contrast_jitter=cfg["data"].get("contrast_jitter", 0.1),
        gaussian_noise=cfg["data"].get("gaussian_noise", 0.005),
        scale_jitter=cfg["data"].get("scale_jitter", 0.0),
    )
    
    test_dataset = IRDSTDataset(
        dataset_root=cfg["data"]["dataset_root"],
        image_size=input_shape[0],
        type="test",
        augment=False,
        crop_size=cfg["data"].get("crop_size", 480),
        base_size=cfg["data"].get("base_size", 512),
        pad_only=cfg["data"].get("pad_only", True),
        pad_to_size=cfg["data"].get("eval_pad_to_size", input_shape),
        pad_divisor=cfg["data"].get("pad_divisor", 32),
        flip_augmentation=False,
        vflip_augmentation=False,
        brightness_jitter=0.0,
        contrast_jitter=0.0,
        gaussian_noise=0.0,
        scale_jitter=0.0,
    )
    
    if is_main_process(local_rank):
        logger.info(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    
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
    
    use_fp16 = bool(cfg["training"].get("fp16", False))
    scaler = GradScaler('cuda') if use_fp16 and cuda else None
    
    loss_scheduler_cfg = cfg.get("loss_scheduler", {})
    loss_weight_scheduler = LossWeightScheduler(
        warmup_epochs=loss_scheduler_cfg.get("warmup_epochs", 5),
        residual_recon_start_epoch=loss_scheduler_cfg.get("residual_recon_start_epoch", 1),
        residual_recon_weight=loss_scheduler_cfg.get("residual_recon_weight", 1.0),
        residual_recon_rampup_epochs=loss_scheduler_cfg.get("residual_recon_rampup_epochs", 10),
    )
    
    training_log = {"config": cfg, "epochs": []}
    
    class_names_list = ["target"]
    eval_conf_thresh = cfg.get("eval", {}).get("conf_thresh", 0.05)
    eval_nms_thresh = cfg.get("eval", {}).get("nms_thresh", 0.5)
    eval_period = cfg.get("eval", {}).get("eval_period", 1)
    
    if is_main_process(local_rank):
        logger.info(f"Evaluation period: every {eval_period} epochs")
        logger.info("Start training...")
    
    # 加载模型和优化器（不加载 lr_scheduler 状态）
    start_epoch = checkpoint_manager.load_checkpoint(model, optimizer, lr_scheduler=None, scaler=scaler)

    # 重新初始化学习率调度器
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["training"]["epochs"], eta_min=cfg["training"]["lr"] * 0.01
    )

    # 直接设置 last_epoch 属性（不调用 step()）
    if start_epoch > 0:
        lr_scheduler.last_epoch = start_epoch
        # 手动更新学习率
        lr = lr_scheduler.get_last_lr()[0]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        if is_main_process(local_rank):
            logger.info(f"Learning rate scheduler set to epoch {start_epoch}")
            logger.info(f"Current learning rate: {optimizer.param_groups[0]['lr']}")
    
    accumulation_steps = cfg["training"].get("accumulation_steps", 1)
    if is_main_process(local_rank):
        logger.info(f"Gradient Accumulation Steps: {accumulation_steps}")
        logger.info(f"Effective Batch Size: {world_size * cfg['training']['batch_size'] * accumulation_steps}")
    
    for epoch in range(start_epoch, cfg["training"]["epochs"]):
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
                logger.info(f"Epoch {epoch + 1}/{cfg['training']['epochs']} - Starting evaluation")
                logger.info(f"{'=' * 60}")
            
            eval_results = evaluate_model(
                model=model,
                test_dataset=test_dataset,
                class_names=class_names_list,
                input_shape=input_shape,
                conf_thresh=eval_conf_thresh,
                nms_thresh=eval_nms_thresh,
                iou_thresh=cfg["eval"].get("iou_thresh", 0.5),
                cuda=cuda,
                local_rank=local_rank,
            )
            
            current_mAP = eval_results["ap50"]
            
            if is_main_process(local_rank):
                logger.info(f"Epoch {epoch + 1} evaluation complete: AP@0.5 = {current_mAP * 100:.2f}%")
        
        if is_main_process(local_rank):
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
                "loss_bg_smooth": loss_dict["loss_bg_smooth"],
                "loss_tgt_content": loss_dict["loss_tgt_content"],
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
        
        checkpoint_manager.save_training_log(training_log)
        lr_scheduler.step()
    
    if is_main_process(local_rank):
        logger.info("Training complete!")
        logger.info(f"All training files saved to: {save_dir}")
    
    cleanup_distributed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/memistd_small_target_config 1.yaml", help="Path to config file")
    parser.add_argument("--resume", type=str, default="./checkpoint/20260225_013926", help="Resume from checkpoint directory (e.g., checkpoint/20260215_123456)")
    # parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint directory (e.g., checkpoint/20260215_123456)")
    args = parser.parse_args()
    
    train_distributed(args.config, args.resume)