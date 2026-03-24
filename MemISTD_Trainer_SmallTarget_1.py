"""
MemISTD Small Target Detection Training Script
===============================================

Training script optimized for infrared small target detection.

Key Features:
1. Multi-scale detection at P0/P1/P2 (full resolution, 1/2, 1/4)
2. Focal Loss + NWD Loss for small target optimization
3. Enhanced SimOTA with larger center_radius
4. Memory-augmented target/background separation
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
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
from utils.eval_utils import evaluate_detection, print_evaluation_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class CheckpointManager:
    """Checkpoint Manager for saving and loading training states"""
    
    def __init__(self, root_dir="checkpoint", monitor="mAP", mode="max", resume_dir=None):
        self.root_dir = root_dir
        self.monitor = monitor
        self.mode = mode
        self.resume_dir = resume_dir
        
        if resume_dir:
            self.save_dir = resume_dir
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
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.save_dir = os.path.join(root_dir, timestamp)
            os.makedirs(self.save_dir, exist_ok=True)
            
            if mode == "min":
                self.best_value = float("inf")
            else:
                self.best_value = float("-inf")
            
            self.best_epoch = 0
            self.start_epoch = 0
            logger.info(f"Checkpoint directory created: {self.save_dir}")
    
    def save_config(self, config):
        config_path = os.path.join(self.save_dir, "config.yaml")
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
        logger.info(f"Config saved: {config_path}")
    
    def load_checkpoint(self, model, optimizer, lr_scheduler=None, scaler=None):
        if not self.resume_dir:
            return 0
        
        checkpoint_path = os.path.join(self.resume_dir, "last.pth")
        if not os.path.exists(checkpoint_path):
            logger.warning(f"Checkpoint file not found: {checkpoint_path}")
            return 0
        
        checkpoint = torch.load(checkpoint_path, map_location="cuda" if torch.cuda.is_available() else "cpu")
        
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if lr_scheduler is not None and "lr_scheduler_state_dict" in checkpoint:
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        
        if scaler is not None and "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
        start_epoch = checkpoint.get("epoch", 0)
        
        logger.info(f"Successfully loaded checkpoint from epoch {start_epoch}")
        if "loss" in checkpoint:
            logger.info(f"Previous loss: {checkpoint['loss']:.4f}")
        if "mAP" in checkpoint and checkpoint["mAP"] is not None:
            logger.info(f"Previous mAP: {checkpoint['mAP'] * 100:.2f}%")
        
        return start_epoch
    
    def save_checkpoint(self, epoch, model, optimizer, lr_scheduler=None, scaler=None, loss=None, mAP=None, is_last=False, save_every_epoch=True):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
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
        
        if self.monitor == "mAP" and mAP is not None:
            current_value = mAP
            metric_name = "mAP"
        elif self.monitor == "loss" and loss is not None:
            current_value = loss
            metric_name = "loss"
        elif mAP is not None:
            current_value = mAP
            metric_name = "mAP"
        elif loss is not None:
            current_value = loss
            metric_name = "loss"
        else:
            current_value = None
            metric_name = None
        
        is_best = False
        
        if current_value is not None:
            if metric_name == "mAP":
                is_best = current_value > self.best_value
            else:
                is_best = current_value < self.best_value
            
            if is_best:
                self.best_value = current_value
                self.best_epoch = epoch
                best_path = os.path.join(self.save_dir, "best.pth")
                torch.save(checkpoint, best_path)
                
                if metric_name == "mAP":
                    logger.info(f"Best model saved! Epoch {epoch}, {metric_name}={current_value * 100:.2f}%")
                else:
                    logger.info(f"Best model saved! Epoch {epoch}, {metric_name}={current_value:.6f}")
        
        if is_last:
            logger.info(f"Last epoch model saved: {last_path}")
        
        return is_best
    
    def save_training_log(self, log_data):
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
):
    """
    Evaluate model performance using model.detect().
    No OpenCV dependency - pure NumPy/PyTorch implementation.
    
    Returns:
        Dictionary with P, R, F1, AP50, FPS metrics
    """
    device = "cuda" if cuda else "cpu"
    
    metrics = evaluate_detection(
        model=model,
        test_dataset=test_dataset,
        conf_threshold=conf_thresh,
        nms_threshold=nms_thresh,
        iou_threshold=iou_thresh,
        max_detections=max_detections,
        device=device,
        warmup_iterations=3,
        local_rank=0,
        disable_pbar=False,
    )
    
    print_evaluation_report(metrics)
    logger.info(f"Recall: {metrics['recall']*100:.2f}%, Precision: {metrics['precision']*100:.2f}%, F1: {metrics['f1']*100:.2f}%")
    logger.info(f"AP@0.5: {metrics['ap50']*100:.2f}%, FPS: {metrics['fps']:.2f}")
    
    return metrics


def decode_multiscale_outputs(
    outputs,
    num_classes,
    conf_thres=0.05,
    nms_thres=0.5,
    strides=[1, 2, 4],
):
    """Decode multi-scale outputs"""
    from torchvision.ops import batched_nms
    
    batch_size = outputs[0].shape[0]
    all_predictions = []
    
    for i, (output, stride) in enumerate(zip(outputs, strides)):
        B, C, H, W = output.shape
        
        yv, xv = torch.meshgrid([torch.arange(H), torch.arange(W)], indexing="ij")
        grid = torch.stack((xv, yv), 2).view(1, H, W, 2).type_as(output)
        grid = grid + 0.5
        
        output = output.permute(0, 2, 3, 1).contiguous()
        output[..., :2] = (output[..., :2] + grid) * stride
        output[..., 2:4] = torch.exp(output[..., 2:4]) * stride
        output[..., 2:4] = torch.clamp(output[..., 2:4], min=2.0)
        
        output = output.view(batch_size, -1, 5 + num_classes)
        all_predictions.append(output)
    
    predictions = torch.cat(all_predictions, 1)
    
    batch_detections = []
    for img_i, pred in enumerate(predictions):
        pred = pred[pred[:, 4] > conf_thres]
        
        if pred.shape[0] == 0:
            batch_detections.append(None)
            continue
        
        class_conf, class_pred = torch.max(pred[:, 5:], 1, keepdim=True)
        detections = torch.cat(
            (pred[:, :4], pred[:, 4:5], class_conf, class_pred.float()),
            1,
        )
        
        keep = batched_nms(
            detections[:, :4],
            detections[:, 4] * detections[:, 5],
            detections[:, 6],
            nms_thres,
        )
        
        if keep.shape[0] > 0:
            det = detections[keep]
            x1 = det[:, 0] - det[:, 2] / 2
            y1 = det[:, 1] - det[:, 3] / 2
            x2 = det[:, 0] + det[:, 2] / 2
            y2 = det[:, 1] + det[:, 3] / 2
            boxes = torch.stack([x1, y1, x2, y2], dim=1)
            scores = (det[:, 4] * det[:, 5]).unsqueeze(1)
            batch_detections.append(torch.cat([boxes, scores], dim=1))
        else:
            batch_detections.append(None)
    
    return batch_detections


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
    if local_rank == 0:
        logger.info(
            f"Epoch {epoch + 1}/{Epoch} - Stage: {stage} | ResidualRecon Weight: {residual_recon_weight:.4f}"
        )
        pbar = tqdm(
            total=epoch_step,
            desc=f"Epoch {epoch + 1}/{Epoch}",
            postfix=dict,
            mininterval=0.3,
            dynamic_ncols=True,
        )
    
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
                
                loss_dict = model.compute_loss(
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
                    logger.warning(f"NaN/Inf detected in loss at iteration {iteration}!")
                    logger.warning(f"  yolo_loss: {loss_yolo.item() if torch.is_tensor(loss_yolo) else loss_yolo}")
                    logger.warning(f"  residual_recon_loss: {loss_residual_recon.item() if torch.is_tensor(loss_residual_recon) else loss_residual_recon}")
                    del outputs, loss_dict
                    continue
                
                del outputs, loss_dict
        else:
            outputs = model(images, decode_image=True)
            outputs["target_mask"] = target_mask
            
            loss_dict = model.compute_loss(
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
                logger.warning(f"NaN/Inf detected in loss at iteration {iteration}!")
                logger.warning(f"  yolo_loss: {loss_yolo.item() if torch.is_tensor(loss_yolo) else loss_yolo}")
                logger.warning(f"  residual_recon_loss: {loss_residual_recon.item() if torch.is_tensor(loss_residual_recon) else loss_residual_recon}")
                del outputs, loss_dict
                continue
            
            del outputs, loss_dict
        
        loss = loss / accumulation_steps
        
        if fp16:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        if (iteration + 1) % accumulation_steps == 0:
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
        
        if local_rank == 0:
            pbar.set_postfix(
                **{
                    "lr": optimizer.param_groups[0]["lr"],
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
    
    if local_rank == 0:
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


def train_small_target(config_path="config/memistd_small_target_config.yaml", resume_dir=None):
    """Main training function"""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    seed_everything(cfg.get("seed", 42))
    
    cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda else "cpu")
    local_rank = 0
    
    checkpoint_manager = CheckpointManager(
        root_dir="checkpoint",
        monitor="mAP",
        mode="max",
        resume_dir=resume_dir,
    )
    
    if not resume_dir:
        checkpoint_manager.save_config(cfg)
    
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
        # save_heatmaps=cfg["model"].get("save_heatmaps", False),
        ms_fusion_type=cfg["model"].get("ms_fusion_type", "attention"),
        global_fusion_strategy=cfg["model"].get("global_fusion_strategy", "spatial_reweight"),
        bg_patch_sizes=cfg["model"].get("bg_patch_sizes", [3, 5]),
        tg_patch_sizes=cfg["model"].get("tg_patch_sizes", [3, 5]),
    )
    
    if cuda:
        model = model.cuda()
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model Parameters: {total_params / 1e6:.2f}M (trainable: {trainable_params / 1e6:.2f}M)")
    
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
    
    logger.info(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=True,
        drop_last=True,
        collate_fn=irdst_collate_fn,
    )
    
    save_dir = checkpoint_manager.get_save_dir()
    os.makedirs(save_dir, exist_ok=True)
    
    scaler = GradScaler('cuda') if cfg["training"]["fp16"] and cuda else None
    
    accumulation_steps = cfg["training"].get("accumulation_steps", 1)
    logger.info(f"Gradient Accumulation Steps: {accumulation_steps}")
    logger.info(f"Effective Batch Size: {cfg['training']['batch_size'] * accumulation_steps}")
    
    loss_scheduler_cfg = cfg.get("loss_scheduler", {})
    loss_weight_scheduler = LossWeightScheduler(
        warmup_epochs=loss_scheduler_cfg.get("warmup_epochs", 5),
        residual_recon_start_epoch=loss_scheduler_cfg.get("residual_recon_start_epoch", 1),
        residual_recon_weight=loss_scheduler_cfg.get("residual_recon_weight", 1.0),
        residual_recon_rampup_epochs=loss_scheduler_cfg.get("residual_recon_rampup_epochs", 10),
    )
    logger.info(f"\nLoss Weight Scheduler:\n{loss_weight_scheduler}\n")
    
    training_log = {
        "config": cfg,
        "epochs": [],
    }
    
    class_names_list = ["target"]
    eval_conf_thresh = cfg.get("eval", {}).get("conf_thresh", 0.05)
    eval_nms_thresh = cfg.get("eval", {}).get("nms_thresh", 0.5)
    eval_period = cfg.get("eval", {}).get("eval_period", 1)
    strides = yolox_cfg.get("strides", [1, 2, 4])
    
    logger.info(f"Evaluation period: every {eval_period} epochs")
    
    start_epoch = checkpoint_manager.load_checkpoint(model, optimizer, lr_scheduler, scaler)
    
    logger.info("Start training...")
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
        
        current_mAP = None
        if (epoch + 1) % eval_period == 0 or (epoch == cfg["training"]["epochs"] - 1):
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
            )
            
            current_mAP = eval_results["ap50"]
            logger.info(f"Epoch {epoch + 1} evaluation complete: AP@0.5 = {current_mAP * 100:.2f}%")
        
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
            "mAP": current_mAP if current_mAP is not None else None,
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
    
    logger.info("Training complete!")
    logger.info(f"All training files saved to: {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config/memistd_small_target_config 1.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="./checkpoint/20260224_115906",
        # default=None,
        help="Resume from checkpoint directory (e.g., checkpoint/20260215_123456)",
    )
    args = parser.parse_args()
    
    train_small_target(args.config, args.resume)
