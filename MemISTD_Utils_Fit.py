"""
MemISTD Training Utility Functions
"""

import os
import torch
import numpy as np
from tqdm import tqdm
from typing import Optional, Any, Dict, List, Tuple


def get_lr(optimizer):
    """Get current learning rate from optimizer"""
    for param_group in optimizer.param_groups:
        return param_group["lr"]
    return 0.0


def compute_detection_metrics(
    outputs: torch.Tensor,
    targets: Dict,
    num_classes: int,
    input_shape: Tuple[int, int],
    image_shape: Tuple[int, int],
    conf_thres: float = 0.25,
    nms_iou: float = 0.45,
    letterbox_image: bool = True,
) -> Dict[str, float]:
    """
    计算检测指标: Precision, Recall, F1, AP50

    Args:
        outputs: 模型原始输出 [batch, anchors, 85]
        targets: 标签字典，包含 boxes 等
        num_classes: 类别数
        input_shape: 模型输入尺寸 (h, w)
        image_shape: 原始图像尺寸 (h, w)
        conf_thres: 置信度阈值
        nms_iou: NMS IoU阈值

    Returns:
        metrics: 包含 P, R, F1, AP50 的字典
    """
    try:
        from utils.utils_bbox import decode_outputs, non_max_suppression
    except ImportError:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "ap50": 0.0, "mAP": 0.0}

    device = outputs.device

    outputs = decode_outputs(outputs, input_shape)
    outputs = non_max_suppression(
        outputs,
        num_classes,
        input_shape,
        image_shape,
        letterbox_image,
        conf_thres=conf_thres,
        nms_thres=nms_iou,
    )

    all_pred_boxes = []
    all_gt_boxes = []

    if outputs[0] is not None:
        for pred in outputs[0]:
            x1, y1, x2, y2 = pred[:4]
            conf = pred[4] * pred[5]
            all_pred_boxes.append(
                [x1.item(), y1.item(), x2.item(), y2.item(), conf.item()]
            )

    gt_boxes = targets.get("boxes", [])
    if isinstance(gt_boxes, list):
        for gt_box in gt_boxes:
            if isinstance(gt_box, torch.Tensor):
                all_gt_boxes.append(gt_box.cpu().numpy())
            elif isinstance(gt_box, np.ndarray):
                all_gt_boxes.append(gt_box)

    if len(all_pred_boxes) == 0 and len(all_gt_boxes) == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "ap50": 1.0, "mAP": 1.0}
    elif len(all_gt_boxes) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "ap50": 0.0, "mAP": 0.0}
    elif len(all_pred_boxes) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "ap50": 0.0, "mAP": 0.0}

    all_gt_boxes = np.array(all_gt_boxes) if all_gt_boxes else np.zeros((0, 4))
    all_pred_boxes = np.array(all_pred_boxes)

    if len(all_gt_boxes) == 0 or len(all_pred_boxes) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "ap50": 0.0, "mAP": 0.0}

    gt_xyxy = np.zeros_like(all_gt_boxes)
    gt_xyxy[:, 0] = all_gt_boxes[:, 0] - all_gt_boxes[:, 2] / 2
    gt_xyxy[:, 1] = all_gt_boxes[:, 1] - all_gt_boxes[:, 3] / 2
    gt_xyxy[:, 2] = all_gt_boxes[:, 0] + all_gt_boxes[:, 2] / 2
    gt_xyxy[:, 3] = all_gt_boxes[:, 1] + all_gt_boxes[:, 3] / 2

    tp = 0
    fp = 0
    matched_gt = set()

    for pred in sorted(all_pred_boxes, key=lambda x: x[4], reverse=True):
        px1, py1, px2, py2, pconf = pred
        pred_w = px2 - px1
        pred_h = py2 - py1

        best_iou = 0
        best_gt_idx = -1

        for gt_idx, gt_box in enumerate(gt_xyxy):
            if gt_idx in matched_gt:
                continue

            gx1, gy1, gx2, gy2 = gt_box
            inter_x1 = max(px1, gx1)
            inter_y1 = max(py1, gy1)
            inter_x2 = min(px2, gx2)
            inter_y2 = min(py2, gy2)

            inter_w = max(0, inter_x2 - inter_x1)
            inter_h = max(0, inter_y2 - inter_y1)
            inter_area = inter_w * inter_h

            pred_area = pred_w * pred_h
            gt_area = (gx2 - gx1) * (gy2 - gy1)
            union_area = pred_area + gt_area - inter_area + 1e-7

            iou = inter_area / union_area if union_area > 0 else 0
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= 0.5:
            tp += 1
            matched_gt.add(best_gt_idx)
        else:
            fp += 1

    fn = len(all_gt_boxes) - len(matched_gt)

    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = 2 * precision * recall / (precision + recall + 1e-7)

    ap50 = precision

    mAP = ap50

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "ap50": ap50,
        "mAP": mAP,
    }


def memistd_fit_one_epoch(
    model_train,
    model,
    ema,
    detection_loss,
    memory_recon_loss,
    orthogonality_loss,
    loss_history,
    eval_callback,
    optimizer,
    epoch,
    epoch_step,
    epoch_step_val,
    train_loader,
    val_loader,
    Epoch,
    Cuda,
    fp16,
    scaler,
    save_period,
    save_dir,
    local_rank=0,
    config=None,
):
    """MemISTD training function for one epoch"""

    loss = 0
    val_loss = 0

    def _to_device(x, device):
        if torch.is_tensor(x):
            return x.to(device)
        if isinstance(x, list):
            return [_to_device(v, device) for v in x]
        if isinstance(x, tuple):
            return tuple(_to_device(v, device) for v in x)
        if isinstance(x, dict):
            return {k: _to_device(v, device) for k, v in x.items()}
        return x

    pbar = None
    if local_rank == 0:
        print("Start Train")
        pbar = tqdm(
            total=epoch_step, desc=f"Epoch {epoch + 1}/{Epoch}", mininterval=0.3
        )

    model_train.train()

    for iteration, batch in enumerate(train_loader):
        if iteration >= epoch_step:
            break

        images, targets = batch[0], batch[1]

        with torch.no_grad():
            if Cuda:
                device = torch.device(f"cuda:{local_rank}")
                images = images.to(device)
                if isinstance(targets, dict):
                    targets = _to_device(targets, device)

        optimizer.zero_grad()

        if not fp16:
            outputs = model_train(images)

            # 将当前epoch传递给config
            if config is None:
                config = {}
            config["current_epoch"] = epoch

            loss_dict = compute_loss(
                detection_loss,
                memory_recon_loss,
                orthogonality_loss,
                outputs,
                targets,
                images,
                config,
            )
            loss_value = loss_dict["total_loss"]
            loss_value.backward()
            torch.nn.utils.clip_grad_norm_(model_train.parameters(), max_norm=10.0)
            optimizer.step()
        else:
            from torch.cuda.amp import autocast

            with autocast():
                outputs = model_train(images)

                # 将当前epoch传递给config
                if config is None:
                    config = {}
                config["current_epoch"] = epoch

                loss_dict = compute_loss(
                    detection_loss,
                    memory_recon_loss,
                    orthogonality_loss,
                    outputs,
                    targets,
                    images,
                    config,
                )
                loss_value = loss_dict["total_loss"]
            scaler.scale(loss_value).backward()
            torch.nn.utils.clip_grad_norm_(model_train.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()

        if ema:
            ema.update(model_train)

        loss += loss_value.item()

        if local_rank == 0 and pbar is not None:
            lr_value = get_lr(optimizer)
            pbar.set_postfix({"loss": loss / (iteration + 1), "lr": lr_value})
            pbar.update(1)

    if local_rank == 0 and pbar is not None:
        pbar.close()
        print("Finish Train")
        print("Start Validation")
        pbar = tqdm(
            total=epoch_step_val, desc=f"Epoch {epoch + 1}/{Epoch}", mininterval=0.3
        )

    if ema:
        model_train_eval = ema.ema
    else:
        model_train_eval = model_train.eval()

    all_metrics = []
    num_classes = 1
    input_shape = (512, 512)
    image_shape = (512, 512)
    conf_thres = 0.25
    nms_iou = 0.45

    if config is not None:
        eval_cfg = config.get("Eval", {})
        if isinstance(eval_cfg, dict):
            conf_thres = eval_cfg.get("eval_conf_thresh", 0.25)
            nms_iou = eval_cfg.get("eval_nms_thresh", 0.45)
        else:
            conf_thres = config.get("eval_conf_thresh", 0.25)
            nms_iou = config.get("eval_nms_thresh", 0.45)

        data_cfg = config.get("Data", {})
        if isinstance(data_cfg, dict):
            base_size = data_cfg.get("base_size", 512)
            input_shape_cfg = data_cfg.get("input_shape", [512, 512])
            if isinstance(input_shape_cfg, (list, tuple)):
                input_shape = (int(input_shape_cfg[0]), int(input_shape_cfg[1]))
            else:
                input_shape = (int(input_shape_cfg), int(input_shape_cfg))
            image_shape = (base_size, base_size)
        else:
            input_shape = config.get("input_shape", [512, 512])
            if isinstance(input_shape, list) and len(input_shape) >= 2:
                input_shape = (input_shape[0], input_shape[1])
            base_size = config.get("base_size", 512)
            image_shape = (base_size, base_size)

    for iteration, batch in enumerate(val_loader):
        if iteration >= epoch_step_val:
            break
        images, targets = batch[0], batch[1]

        with torch.no_grad():
            if Cuda:
                device = torch.device(f"cuda:{local_rank}")
                images = images.to(device)
                if isinstance(targets, dict):
                    targets = _to_device(targets, device)

            outputs = model_train_eval(images)

            if config is None:
                config = {}
            config["current_epoch"] = epoch

            loss_dict = compute_loss(
                detection_loss,
                memory_recon_loss,
                orthogonality_loss,
                outputs,
                targets,
                images,
                config,
            )
            loss_value = loss_dict["total_loss"]

            if local_rank == 0:
                if isinstance(outputs, dict):
                    output_tensor = outputs.get(
                        "detection_output", outputs.get("output", None)
                    )
                    if output_tensor is None:
                        for k, v in outputs.items():
                            if (
                                isinstance(v, torch.Tensor)
                                and v.dim() == 3
                                and v.shape[1] > 4
                            ):
                                output_tensor = v
                                break
                else:
                    output_tensor = outputs

                if output_tensor is not None:
                    metrics = compute_detection_metrics(
                        output_tensor,
                        targets,
                        num_classes,
                        input_shape,
                        image_shape,
                        conf_thres=conf_thres,
                        nms_iou=nms_iou,
                        letterbox_image=True,
                    )
                    all_metrics.append(metrics)

        val_loss += loss_value.item()
        if local_rank == 0 and pbar is not None:
            pbar.set_postfix({"val_loss": val_loss / (iteration + 1)})
            pbar.update(1)

    if local_rank == 0 and pbar is not None:
        pbar.close()
        print("Finish Validation")

        avg_loss = loss / epoch_step
        avg_val_loss = val_loss / epoch_step_val

        loss_history.append_loss(epoch + 1, avg_loss, avg_val_loss)
        if eval_callback:
            eval_callback.on_epoch_end(epoch + 1, model_train_eval)

        print(f"Epoch:{epoch + 1}/{Epoch}")
        print(f"Total Loss: {avg_loss:.3f} || Val Loss: {avg_val_loss:.3f}")

        if all_metrics:
            avg_precision = np.mean([m["precision"] for m in all_metrics])
            avg_recall = np.mean([m["recall"] for m in all_metrics])
            avg_f1 = np.mean([m["f1"] for m in all_metrics])
            avg_ap50 = np.mean([m["ap50"] for m in all_metrics])
            avg_map = np.mean([m["mAP"] for m in all_metrics])
            print(
                f"P: {avg_precision:.4f} || R: {avg_recall:.4f} || F1: {avg_f1:.4f} || AP50: {avg_ap50:.4f} || mAP: {avg_map:.4f}"
            )

        if ema:
            save_state_dict = ema.ema.state_dict()
        else:
            save_state_dict = model.state_dict()

        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(
                save_state_dict,
                os.path.join(
                    save_dir,
                    f"ep{epoch + 1:03d}-loss{avg_loss:.3f}-val_loss{avg_val_loss:.3f}.pth",
                ),
            )

        if len(loss_history.val_loss) <= 1 or avg_val_loss <= min(
            loss_history.val_loss
        ):
            print("Save best model to best_epoch_weights.pth")
            torch.save(
                save_state_dict, os.path.join(save_dir, "best_epoch_weights.pth")
            )

        torch.save(save_state_dict, os.path.join(save_dir, "last_epoch_weights.pth"))


def compute_loss(
    detection_loss,
    memory_recon_loss,
    orthogonality_loss,
    outputs,
    targets,
    images,
    config=None,
):
    """
    计算MemISTD总损失

    损失函数组成：
    L_total = L_det + α·L_recon_t + β·L_recon_b + γ·L_orth

    其中：
    - L_det: YOLO检测损失（box + obj + cls）
    - L_recon_t: 目标记忆重构损失（仅在目标区域计算）
    - L_recon_b: 背景记忆重构损失（仅在背景区域计算）
    - L_orth: 记忆正交损失（M_t ⊥ M_b）

    预热策略（Warm-up Strategy）：
    - 前 warmup_recon_epochs 个Epoch：冻结 L_recon，让骨干网络先学习基本特征
    - 前 warmup_orth_epochs 个Epoch：冻结 L_orth，让记忆模块先积累特征
    - 之后阶段：所有损失正常加权

    Args:
        detection_loss: 检测损失模块
        memory_recon_loss: 记忆重构损失模块
        orthogonality_loss: 正交损失模块
        outputs: 模型输出字典
        targets: 标签字典
        images: 输入图像
        config: 配置字典

    Returns:
        losses: 损失字典
            - loss_box: 边界框回归损失
            - loss_obj: 置信度损失
            - loss_cls: 分类损失
            - loss_recon_t: 目标重构损失
            - loss_recon_b: 背景重构损失
            - loss_recon: 总重构损失（加权）
            - loss_orth: 正交损失
            - total_loss: 总损失
    """
    losses = {}
    device = images.device

    if isinstance(targets, dict) and "boxes" in targets:
        det_loss = detection_loss(outputs, targets)
        if isinstance(det_loss, dict):
            losses.update(det_loss)
        else:
            losses["detection_loss"] = det_loss
    elif isinstance(targets, (list, tuple)):
        det_loss = detection_loss(outputs, targets)
        if isinstance(det_loss, dict):
            losses.update(det_loss)
        else:
            losses["detection_loss"] = det_loss

    if "loss_box" not in losses:
        losses["loss_box"] = torch.tensor(0.0, device=device, requires_grad=True)
    if "loss_obj" not in losses:
        losses["loss_obj"] = torch.tensor(0.0, device=device, requires_grad=True)
    if "loss_cls" not in losses:
        losses["loss_cls"] = torch.tensor(0.0, device=device, requires_grad=True)

    alpha = (
        memory_recon_loss.recon_t_weight
        if hasattr(memory_recon_loss, "recon_t_weight")
        else 1.0
    )
    beta = (
        memory_recon_loss.recon_b_weight
        if hasattr(memory_recon_loss, "recon_b_weight")
        else 0.1
    )

    if hasattr(memory_recon_loss, "forward"):
        recon_loss = memory_recon_loss(outputs, targets)
        if isinstance(recon_loss, dict):
            losses.update(recon_loss)
        else:
            losses["loss_recon"] = recon_loss

    if "loss_recon_t" not in losses:
        losses["loss_recon_t"] = torch.tensor(0.0, device=device, requires_grad=True)
    if "loss_recon_b" not in losses:
        losses["loss_recon_b"] = torch.tensor(0.0, device=device, requires_grad=True)
    if "loss_recon" not in losses:
        losses["loss_recon"] = losses["loss_recon_t"] + losses["loss_recon_b"]

    gamma = orthogonality_loss.weight if hasattr(orthogonality_loss, "weight") else 0.01

    if hasattr(orthogonality_loss, "forward"):
        orth_loss = orthogonality_loss(outputs)
        if isinstance(orth_loss, dict):
            losses.update(orth_loss)
        else:
            losses["loss_orth"] = orth_loss

    if "loss_orth" not in losses:
        losses["loss_orth"] = torch.tensor(0.0, device=device, requires_grad=True)

    loss_cfg = config.get("Loss", {}) if isinstance(config, dict) else {}
    warmup_recon_epochs = (
        loss_cfg.get("warmup_recon_epochs", None)
        if isinstance(loss_cfg, dict)
        else None
    )
    if warmup_recon_epochs is None:
        warmup_recon_epochs = config.get("warmup_recon_epochs", 5) if config else 5
    warmup_orth_epochs = (
        loss_cfg.get("warmup_orth_epochs", None) if isinstance(loss_cfg, dict) else None
    )
    if warmup_orth_epochs is None:
        warmup_orth_epochs = config.get("warmup_orth_epochs", 10) if config else 10

    current_epoch = config.get("current_epoch", 0) if config else 0

    recon_weight_factor = 0.0 if current_epoch < warmup_recon_epochs else 1.0
    orth_weight_factor = 0.0 if current_epoch < warmup_orth_epochs else 1.0

    if "loss_det" in losses:
        det_term = losses["loss_det"]
    else:
        box_weight = (
            loss_cfg.get("box_weight", None) if isinstance(loss_cfg, dict) else None
        )
        obj_weight = (
            loss_cfg.get("obj_weight", None) if isinstance(loss_cfg, dict) else None
        )
        cls_weight = (
            loss_cfg.get("cls_weight", None) if isinstance(loss_cfg, dict) else None
        )
        if box_weight is None:
            box_weight = config.get("box_weight", 2.5) if config else 2.5
        if obj_weight is None:
            obj_weight = config.get("obj_weight", 1.0) if config else 1.0
        if cls_weight is None:
            cls_weight = config.get("cls_weight", 1.0) if config else 1.0
        det_term = (
            box_weight * losses["loss_box"]
            + obj_weight * losses["loss_obj"]
            + cls_weight * losses["loss_cls"]
        )

    total_loss = (
        det_term
        + alpha * recon_weight_factor * losses["loss_recon_t"]
        + beta * recon_weight_factor * losses["loss_recon_b"]
        + gamma * orth_weight_factor * losses["loss_orth"]
    )

    losses["total_loss"] = total_loss

    return losses
