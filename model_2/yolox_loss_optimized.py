"""
Optimized YOLOX Loss for Small Target Detection
================================================

Key improvements:
1. Increased center_radius (2.5 -> 5.0) for better small target assignment
2. Adaptive center_radius based on target size
3. Focal Loss integration for class imbalance
4. Support for P0/P1/P2 multi-scale detection
"""

import math
from copy import deepcopy
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from .losses import FocalLoss, NWDLoss, CIoULoss


class IOUloss(nn.Module):
    def __init__(self, reduction="none", loss_type="iou"):
        super(IOUloss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, pred, target):
        assert pred.shape[0] == target.shape[0]

        pred = pred.reshape(-1, 4)
        target = target.reshape(-1, 4)
        tl = torch.max(
            (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
        )
        br = torch.min(
            (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
        )

        area_p = torch.prod(pred[:, 2:], 1)
        area_g = torch.prod(target[:, 2:], 1)

        en = (tl < br).type(tl.type()).prod(dim=1)
        area_i = torch.prod(br - tl, 1) * en
        area_u = area_p + area_g - area_i
        iou = (area_i) / (area_u + 1e-16)

        if self.loss_type == "iou":
            loss = 1 - iou**2
        elif self.loss_type == "giou":
            c_tl = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            c_br = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )
            area_c = torch.prod(c_br - c_tl, 1)
            giou = iou - (area_c - area_u) / area_c.clamp(1e-16)
            loss = 1 - giou.clamp(min=-1.0, max=1.0)
        elif self.loss_type == "ciou":
            b1_cxy = pred[:, :2]
            b2_cxy = target[:, :2]
            center_distance = torch.sum(torch.pow((b1_cxy - b2_cxy), 2), axis=-1)
            enclose_mins = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            enclose_maxes = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )
            enclose_wh = torch.max(enclose_maxes - enclose_mins, torch.zeros_like(br))
            enclose_diagonal = torch.sum(torch.pow(enclose_wh, 2), axis=-1)
            ciou = iou - 1.0 * (center_distance) / torch.clamp(
                enclose_diagonal, min=1e-6
            )
            v = (4 / (torch.pi**2)) * torch.pow(
                (
                    torch.atan(pred[:, 2] / torch.clamp(pred[:, 3], min=1e-6))
                    - torch.atan(target[:, 2] / torch.clamp(target[:, 3], min=1e-6))
                ),
                2,
            )
            alpha = v / torch.clamp((1.0 - iou + v), min=1e-6)
            ciou = ciou - alpha * v
            loss = 1 - ciou.clamp(min=-1.0, max=1.0)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss


class YOLOLossOptimized(nn.Module):
    """
    Optimized YOLO Loss for Small Target Detection
    
    Key improvements:
    1. Larger center_radius (5.0 instead of 2.5)
    2. Adaptive center_radius based on target size
    3. Focal Loss for objectness and classification
    4. Support for P0/P1/P2 scales (strides: 1, 2, 4)
    
    Args:
        num_classes: Number of detection classes
        fp16: Whether to use FP16
        strides: Strides for each scale (default: [1, 2, 4] for P0/P1/P2)
        center_radius: Center radius for SimOTA (default: 5.0)
        use_focal: Whether to use Focal Loss
        focal_alpha: Alpha for Focal Loss
        focal_gamma: Gamma for Focal Loss
        adaptive_center_radius: Whether to use adaptive center radius
    """
    
    def __init__(
        self,
        num_classes,
        fp16,
        strides=[1, 2, 4],
        center_radius=5.0,
        use_focal=True,
        focal_alpha=0.25,
        focal_gamma=2.0,
        adaptive_center_radius=True,
        debug=False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.strides = strides
        self.center_radius = center_radius
        self.use_focal = use_focal
        self.adaptive_center_radius = adaptive_center_radius
        self.debug = debug
        
        if use_focal:
            self.bcewithlog_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, reduction="none")
        else:
            self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")
        
        self.iou_loss = IOUloss(reduction="none")
        self.grids = [torch.zeros(1)] * len(strides)
        self.fp16 = fp16

    def forward(self, inputs, labels=None, return_components: bool = False):
        """
        Args:
            inputs: Supports two formats
                1. Multi-scale list: [[B, C+5, H1, W1], [B, C+5, H2, W2], [B, C+5, H3, W3]]
                2. Single-scale tensor: [B, C+5, H, W]
            labels: Ground truth labels
            return_components: Whether to return individual loss components
        
        Returns:
            Total loss or dictionary of loss components
        """
        if isinstance(inputs, (list, tuple)):
            return self._forward_multiscale(inputs, labels, return_components)
        else:
            return self._forward_singlescale(inputs, labels, return_components)

    def _forward_multiscale(self, inputs, labels, return_components):
        """Process multi-scale list input"""
        outputs = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []

        for k, (stride, output) in enumerate(zip(self.strides, inputs)):
            output, grid = self.get_output_and_grid(output, k, stride)
            x_shifts.append(grid[:, :, 0])
            y_shifts.append(grid[:, :, 1])
            expanded_strides.append(torch.ones_like(grid[:, :, 0]) * stride)
            outputs.append(output)

        total_loss, loss_box, loss_obj, loss_cls, num_fg = self.get_losses(
            x_shifts, y_shifts, expanded_strides, labels, torch.cat(outputs, 1)
        )
        
        if return_components:
            return {
                "total_loss": total_loss,
                "loss_box": loss_box,
                "loss_obj": loss_obj,
                "loss_cls": loss_cls,
                "num_fg": total_loss.new_tensor(float(num_fg)),
            }
        return total_loss

    def _forward_singlescale(self, inputs, labels, return_components):
        """Process single-scale tensor input"""
        stride = self.strides[0]
        output, grid = self.get_output_and_grid(inputs, 0, stride)
        
        x_shifts = [grid[:, :, 0]]
        y_shifts = [grid[:, :, 1]]
        expanded_strides = [torch.ones_like(grid[:, :, 0]) * stride]
        
        total_loss, loss_box, loss_obj, loss_cls, num_fg = self.get_losses(
            x_shifts, y_shifts, expanded_strides, labels, output
        )
        
        if return_components:
            return {
                "total_loss": total_loss,
                "loss_box": loss_box,
                "loss_obj": loss_obj,
                "loss_cls": loss_cls,
                "num_fg": total_loss.new_tensor(float(num_fg)),
            }
        return total_loss
    def get_output_and_grid(self, output, k, stride):
        grid = self.grids[k]
        hsize, wsize = output.shape[-2:]
        
        if grid.shape[2:4] != output.shape[2:4]:
            yv, xv = torch.meshgrid(
                [torch.arange(hsize), torch.arange(wsize)], indexing="ij"
            )
            grid = torch.stack((xv, yv), 2).view(1, hsize, wsize, 2).type(output.type())
            self.grids[k] = grid
        
        grid = grid.view(1, -1, 2)
        grid = grid + 0.5
        
        output = output.flatten(start_dim=2).permute(0, 2, 1)
        
        output[..., :2] = (output[..., :2] + grid.type_as(output)) * stride
        output[..., 2:4] = torch.exp(output[..., 2:4]) * stride
        
        return output, grid

    def get_losses(self, x_shifts, y_shifts, expanded_strides, labels, outputs):
        bbox_preds = outputs[:, :, :4]
        obj_preds = outputs[:, :, 4:5]
        cls_preds = outputs[:, :, 5:]

        total_num_anchors = outputs.shape[1]
        x_shifts = torch.cat(x_shifts, 1).type_as(outputs)
        y_shifts = torch.cat(y_shifts, 1).type_as(outputs)
        expanded_strides = torch.cat(expanded_strides, 1).type_as(outputs)

        cls_targets = []
        reg_targets = []
        obj_targets = []
        fg_masks = []

        num_fg = 0.0
        for batch_idx in range(outputs.shape[0]):
            num_gt = len(labels[batch_idx])
            if num_gt == 0:
                cls_target = outputs.new_zeros((0, self.num_classes))
                reg_target = outputs.new_zeros((0, 4))
                obj_target = outputs.new_zeros((total_num_anchors, 1))
                fg_mask = outputs.new_zeros(total_num_anchors).bool()
            else:
                gt_bboxes_per_image = labels[batch_idx][..., :4].type_as(outputs)
                gt_classes = labels[batch_idx][..., 4].type_as(outputs)
                bboxes_preds_per_image = bbox_preds[batch_idx]
                cls_preds_per_image = cls_preds[batch_idx]
                obj_preds_per_image = obj_preds[batch_idx]

                (
                    gt_matched_classes,
                    fg_mask,
                    pred_ious_this_matching,
                    matched_gt_inds,
                    num_fg_img,
                ) = self.get_assignments(
                    num_gt,
                    total_num_anchors,
                    gt_bboxes_per_image,
                    gt_classes,
                    bboxes_preds_per_image,
                    cls_preds_per_image,
                    obj_preds_per_image,
                    expanded_strides,
                    x_shifts,
                    y_shifts,
                )
                torch.cuda.empty_cache()
                num_fg += num_fg_img
                if num_fg_img == 0 or matched_gt_inds.numel() == 0:
                    cls_target = outputs.new_zeros((0, self.num_classes))
                    reg_target = outputs.new_zeros((0, 4))
                    obj_target = fg_mask.unsqueeze(-1)
                else:
                    cls_target = F.one_hot(
                        gt_matched_classes.to(torch.int64), self.num_classes
                    ).float() * pred_ious_this_matching.unsqueeze(-1)
                    obj_target = fg_mask.unsqueeze(-1)
                    reg_target = gt_bboxes_per_image[matched_gt_inds.long()]
            cls_targets.append(cls_target)
            reg_targets.append(reg_target)
            obj_targets.append(obj_target.type(cls_target.type()))
            fg_masks.append(fg_mask)

        cls_targets = torch.cat(cls_targets, 0)
        reg_targets = torch.cat(reg_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        fg_masks = torch.cat(fg_masks, 0)

        num_fg = max(num_fg, 1)
        loss_iou = (self.iou_loss(bbox_preds.reshape(-1, 4)[fg_masks], reg_targets)).sum() / num_fg
        loss_obj = (self.bcewithlog_loss(obj_preds.reshape(-1, 1), obj_targets)).sum() / num_fg
        
        # 对于单类别检测，不计算分类损失
        # if self.num_classes == 1:
            # loss_cls = torch.tensor(0.0, device=cls_preds.device)
        # else:
        loss_cls = (self.bcewithlog_loss(cls_preds.reshape(-1, self.num_classes)[fg_masks], cls_targets)).sum() / num_fg
        
        reg_weight = 5.0
        loss_box = reg_weight * loss_iou
        loss = loss_box + loss_obj + loss_cls

        return loss, loss_box, loss_obj, loss_cls, num_fg

    @torch.no_grad()
    def get_assignments(
        self,
        num_gt,
        total_num_anchors,
        gt_bboxes_per_image,
        gt_classes,
        bboxes_preds_per_image,
        cls_preds_per_image,
        obj_preds_per_image,
        expanded_strides,
        x_shifts,
        y_shifts,
    ):
        """
        YOLOX SimOTA 标签分配核心函数
        
        功能：为每个GT框分配合适的anchor点，确定哪些预测是正样本
        
        参数说明：
        ----------
        num_gt : int
            当前图像中GT框的数量
        total_num_anchors : int
            特征图上所有anchor点的总数 (H*W)
        gt_bboxes_per_image : Tensor [num_gt, 4]
            GT框坐标，格式为 (cx, cy, w, h) 中心点坐标+宽高
        gt_classes : Tensor [num_gt]
            每个GT框的类别索引
        bboxes_preds_per_image : Tensor [total_num_anchors, 4]
            网络预测的边界框，格式为 (cx, cy, w, h)
        cls_preds_per_image : Tensor [total_num_anchors, num_classes]
            网络预测的分类分数（sigmoid之前）
        obj_preds_per_image : Tensor [total_num_anchors, 1]
            网络预测的目标置信度（sigmoid之前）
        expanded_strides : Tensor [total_num_anchors]
            每个anchor点对应的下采样步长
        x_shifts : Tensor [total_num_anchors]
            anchor点在特征图上的x坐标偏移
        y_shifts : Tensor [total_num_anchors]
            anchor点在特征图上的y坐标偏移
        
        返回值：
        -------
        gt_matched_classes : Tensor [num_fg]
            每个正样本对应的GT类别
        fg_mask : Tensor [total_num_anchors]
            布尔张量，标记哪些anchor是正样本
        pred_ious_this_matching : Tensor [num_fg]
            每个正样本与其匹配GT的IoU值
        matched_gt_inds : Tensor [num_fg]
            每个正样本匹配的GT索引
        num_fg : int
            正样本总数
        """
        
        # ========== 第零步：验证GT框尺寸有效性 ==========
        # 过滤掉宽或高 <= 0 的无效GT框，避免后续计算出错
        gt_w = gt_bboxes_per_image[:, 2]
        gt_h = gt_bboxes_per_image[:, 3]
        invalid_gt_mask = (gt_w <= 0) | (gt_h <= 0)
        
        if invalid_gt_mask.any():
            if self.debug:
                print(f"[DEBUG] 发现 {(invalid_gt_mask.sum().item())} 个无效GT框（宽高<=0）")
            valid_gt_mask = ~invalid_gt_mask
            gt_bboxes_per_image = gt_bboxes_per_image[valid_gt_mask]
            gt_classes = gt_classes[valid_gt_mask]
            num_gt = gt_bboxes_per_image.shape[0]
            
            # 如果所有GT都无效，返回空结果
            if num_gt == 0:
                fg_mask = gt_classes.new_zeros(total_num_anchors).bool()
                return (
                    gt_classes.new_tensor([]),
                    fg_mask,
                    gt_classes.new_tensor([]),
                    gt_classes.new_tensor([], dtype=torch.long),
                    0,
                )
        
        # ========== 第一步：初步筛选候选anchor ==========
        # 调用 get_in_boxes_info 筛选出可能在GT框内或中心区域内的anchor
        # fg_mask: [total_num_anchors] 布尔张量，标记候选anchor
        # is_in_boxes_and_center: [num_gt, num_in_boxes_anchor] 标记每个GT与候选anchor的位置关系
        fg_mask, is_in_boxes_and_center = self.get_in_boxes_info(
            gt_bboxes_per_image,
            expanded_strides,
            x_shifts,
            y_shifts,
            total_num_anchors,
            num_gt,
        )
        if self.debug:
            print(f"[DEBUG] 候选anchor数量: {fg_mask.sum().item()} / {total_num_anchors}")

        # ========== 第二步：提取候选anchor的预测值 ==========
        # 只保留候选anchor的预测，减少后续计算量
        bboxes_preds_per_image = bboxes_preds_per_image[fg_mask]  # [num_in_boxes_anchor, 4]
        cls_preds_ = cls_preds_per_image[fg_mask]  # [num_in_boxes_anchor, num_classes]
        obj_preds_ = obj_preds_per_image[fg_mask]  # [num_in_boxes_anchor, 1]
        num_in_boxes_anchor = bboxes_preds_per_image.shape[0]
        if self.debug:
            print(f"[DEBUG] 筛选后anchor数量: {num_in_boxes_anchor}")

        # 边界情况：如果没有候选anchor，返回空结果
        if num_in_boxes_anchor == 0:
            return (
                gt_classes.new_tensor([]),
                fg_mask,
                gt_classes.new_tensor([]),
                gt_classes.new_tensor([], dtype=torch.long),
                0,
            )

        # ========== 第三步：计算成对IoU ==========
        # pair_wise_ious: [num_gt, num_in_boxes_anchor]
        # 每个GT框与每个候选anchor预测框之间的IoU
        
        # 验证预测边界框的有效性（宽高必须 > 0）
        # 使用 nan_to_num 避免 isnan 检查触发 CUDA 断言
        pred_w = bboxes_preds_per_image[:, 2]
        pred_h = bboxes_preds_per_image[:, 3]
        # 将 NaN 替换为 -1（会被 <= 0 检测到）
        pred_w_safe = torch.nan_to_num(pred_w, nan=-1.0)
        pred_h_safe = torch.nan_to_num(pred_h, nan=-1.0)
        invalid_pred_mask = (pred_w_safe <= 0) | (pred_h_safe <= 0)
        
        if invalid_pred_mask.any():
            if self.debug:
                print(f"[DEBUG] 发现 {(invalid_pred_mask.sum().item())} 个无效预测框（宽高<=0或NaN）")
            valid_pred_mask = ~invalid_pred_mask
            bboxes_preds_per_image = bboxes_preds_per_image[valid_pred_mask]
            cls_preds_ = cls_preds_[valid_pred_mask]
            obj_preds_ = obj_preds_[valid_pred_mask]
            is_in_boxes_and_center = is_in_boxes_and_center[:, valid_pred_mask]
            
            fg_mask_indices = fg_mask.nonzero(as_tuple=True)[0]
            valid_fg_mask_indices = fg_mask_indices[valid_pred_mask]
            fg_mask = fg_mask.clone()
            fg_mask[:] = False
            fg_mask[valid_fg_mask_indices] = True
            
            num_in_boxes_anchor = bboxes_preds_per_image.shape[0]
            
            if num_in_boxes_anchor == 0:
                return (
                    gt_classes.new_tensor([]),
                    fg_mask,
                    gt_classes.new_tensor([]),
                    gt_classes.new_tensor([], dtype=torch.long),
                    0,
                )
        
        pair_wise_ious = self.bboxes_iou(
            gt_bboxes_per_image, bboxes_preds_per_image, False
        )
        # IoU loss = -log(IoU)，IoU越大loss越小
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)
        if self.debug:
            print(f"[DEBUG] IoU范围: [{pair_wise_ious.min().item():.4f}, {pair_wise_ious.max().item():.4f}]")

        # ========== 第四步：计算成对分类损失 ==========
        # SimOTA使用分类损失和IoU损失的加权和作为匹配代价
        if self.fp16:
            with torch.amp.autocast("cuda", enabled=False):
                # 将预测值扩展为 [num_gt, num_in_boxes_anchor, num_classes]
                # 并应用sigmoid激活，同时考虑目标置信度
                cls_preds_ = (
                    cls_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
                    * obj_preds_.unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
                )
                # 构造GT的one-hot编码 [num_gt, num_in_boxes_anchor, num_classes]
                gt_cls_per_image = (
                    F.one_hot(gt_classes.to(torch.int64), self.num_classes)
                    .float()
                    .unsqueeze(1)
                    .repeat(1, num_in_boxes_anchor, 1)
                )
                # 计算BCE损失，使用sqrt()是一种技巧，可以改善梯度
                # 注意：sqrt需要非负输入，sigmoid输出在[0,1]，所以安全
                cls_preds_sqrt = torch.sqrt(torch.clamp(cls_preds_, min=1e-7))
                pair_wise_cls_loss = F.binary_cross_entropy(
                    cls_preds_sqrt, gt_cls_per_image, reduction="none"
                ).sum(-1)  # [num_gt, num_in_boxes_anchor]
        else:
            cls_preds_ = (
                cls_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
                * obj_preds_.unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
            )
            gt_cls_per_image = (
                F.one_hot(gt_classes.to(torch.int64), self.num_classes)
                .float()
                .unsqueeze(1)
                .repeat(1, num_in_boxes_anchor, 1)
            )
            # 注意：sqrt需要非负输入，sigmoid输出在[0,1]，所以安全
            cls_preds_sqrt = torch.sqrt(torch.clamp(cls_preds_, min=1e-7))
            pair_wise_cls_loss = F.binary_cross_entropy(
                cls_preds_sqrt, gt_cls_per_image, reduction="none"
            ).sum(-1)
            del cls_preds_

        # NaN/Inf 防护：确保中间结果有效
        pair_wise_cls_loss = torch.nan_to_num(pair_wise_cls_loss, nan=1e4, posinf=1e4, neginf=1e4)
        pair_wise_ious_loss = torch.nan_to_num(pair_wise_ious_loss, nan=1e4, posinf=1e4, neginf=1e4)

        # ========== 第五步：计算总匹配代价 ==========
        # cost = 分类损失 + 3.0 * IoU损失 + 位置惩罚
        # 位置惩罚：如果anchor不在GT框内且不在中心区域，给予大惩罚(100000)
        cost = (
            pair_wise_cls_loss
            + 3.0 * pair_wise_ious_loss
            + 100000.0 * (~is_in_boxes_and_center).float()
        )
        
        # 再次确保 cost 有效
        cost = torch.nan_to_num(cost, nan=1e8, posinf=1e8, neginf=-1e8)
        
        if self.debug:
            print(f"[DEBUG] cost矩阵形状: {cost.shape}, 范围: [{cost.min().item():.4f}, {cost.max().item():.4f}]")

        # ========== 第六步：动态k匹配 ==========
        # 使用SimOTA的动态k策略进行最终匹配
        num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds = (
            self.dynamic_k_matching(cost, pair_wise_ious, gt_classes, num_gt, fg_mask)
        )
        if self.debug:
            print(f"[DEBUG] 正样本数量: {num_fg}, 匹配的GT索引: {matched_gt_inds}")
        
        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss
        return (
            gt_matched_classes,
            fg_mask,
            pred_ious_this_matching,
            matched_gt_inds,
            num_fg,
        )

    def bboxes_iou(self, bboxes_a, bboxes_b, xyxy=True):
        if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
            raise IndexError

        if xyxy:
            tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
            br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
            area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
            area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
        else:
            tl = torch.max(
                (bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
                (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2),
            )
            br = torch.min(
                (bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
                (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2),
            )

            area_a = torch.prod(bboxes_a[:, 2:], 1)
            area_b = torch.prod(bboxes_b[:, 2:], 1)
        en = (tl < br).type(tl.type()).prod(dim=2)
        area_i = torch.prod(br - tl, 2) * en
        
        area_a = torch.clamp(area_a, min=1e-7)
        area_b = torch.clamp(area_b, min=1e-7)
        
        iou = area_i / (area_a[:, None] + area_b - area_i)
        iou = torch.clamp(iou, min=0.0, max=1.0)
        iou = torch.nan_to_num(iou, nan=0.0, posinf=1.0, neginf=0.0)
        return iou

    def get_in_boxes_info(
        self,
        gt_bboxes_per_image,
        expanded_strides,
        x_shifts,
        y_shifts,
        total_num_anchors,
        num_gt,
        center_radius=None,
    ):
        if center_radius is None:
            center_radius = self.center_radius
        
        expanded_strides_per_image = expanded_strides[0]
        x_centers_per_image = (
            ((x_shifts[0] + 0.5) * expanded_strides_per_image)
            .unsqueeze(0)
            .repeat(num_gt, 1)
        )
        y_centers_per_image = (
            ((y_shifts[0] + 0.5) * expanded_strides_per_image)
            .unsqueeze(0)
            .repeat(num_gt, 1)
        )
        
        if self.debug:
            print(f"[DEBUG] 训练时anchor中心示例 (前5个):")
            print(f"  x_centers: {x_centers_per_image[0, :5].tolist()}")
            print(f"  y_centers: {y_centers_per_image[0, :5].tolist()}")
            print(f"  注意: anchor中心 = (grid_index + 0.5) * stride")

        gt_bboxes_per_image_l = (
            (gt_bboxes_per_image[:, 0] - 0.5 * gt_bboxes_per_image[:, 2])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_r = (
            (gt_bboxes_per_image[:, 0] + 0.5 * gt_bboxes_per_image[:, 2])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_t = (
            (gt_bboxes_per_image[:, 1] - 0.5 * gt_bboxes_per_image[:, 3])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_b = (
            (gt_bboxes_per_image[:, 1] + 0.5 * gt_bboxes_per_image[:, 3])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )

        b_l = x_centers_per_image - gt_bboxes_per_image_l
        b_r = gt_bboxes_per_image_r - x_centers_per_image
        b_t = y_centers_per_image - gt_bboxes_per_image_t
        b_b = gt_bboxes_per_image_b - y_centers_per_image
        bbox_deltas = torch.stack([b_l, b_t, b_r, b_b], 2)

        is_in_boxes = bbox_deltas.min(dim=-1).values > 0.0
        is_in_boxes_all = is_in_boxes.sum(dim=0) > 0

        if self.adaptive_center_radius:
            gt_w = gt_bboxes_per_image[:, 2]
            gt_h = gt_bboxes_per_image[:, 3]
            adaptive_radius = torch.max(
                center_radius * torch.ones_like(gt_w),
                torch.min(gt_w, gt_h) / 2
            )
            adaptive_radius = adaptive_radius.unsqueeze(1).repeat(1, total_num_anchors)
            radius_for_centers = adaptive_radius * expanded_strides_per_image.unsqueeze(0)
        else:
            radius_for_centers = center_radius * expanded_strides_per_image.unsqueeze(0)

        gt_bboxes_per_image_l = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) - radius_for_centers
        gt_bboxes_per_image_r = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) + radius_for_centers
        gt_bboxes_per_image_t = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) - radius_for_centers
        gt_bboxes_per_image_b = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) + radius_for_centers

        c_l = x_centers_per_image - gt_bboxes_per_image_l
        c_r = gt_bboxes_per_image_r - x_centers_per_image
        c_t = y_centers_per_image - gt_bboxes_per_image_t
        c_b = gt_bboxes_per_image_b - y_centers_per_image
        center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)

        is_in_centers = center_deltas.min(dim=-1).values > 0.0
        is_in_centers_all = is_in_centers.sum(dim=0) > 0

        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all
        is_in_boxes_and_center = (
            is_in_boxes[:, is_in_boxes_anchor] & is_in_centers[:, is_in_boxes_anchor]
        )
        return is_in_boxes_anchor, is_in_boxes_and_center

    def dynamic_k_matching(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask):
        """
        SimOTA 动态k匹配算法
        
        功能：根据代价矩阵和IoU，为每个GT框动态分配k个正样本anchor
        
        SimOTA的核心思想：
        - 每个GT框分配的正样本数量k是动态计算的
        - k值等于该GT与top-k个anchor的IoU之和（向下取整，至少为1）
        - 这使得大目标（IoU普遍高）分配更多正样本，小目标分配较少
        
        参数说明：
        ----------
        cost : Tensor [num_gt, num_in_boxes_anchor]
            代价矩阵，值越小表示该anchor越适合分配给该GT
        pair_wise_ious : Tensor [num_gt, num_in_boxes_anchor]
            每个GT与每个候选anchor的IoU矩阵
        gt_classes : Tensor [num_gt]
            每个GT框的类别索引
        num_gt : int
            GT框数量
        fg_mask : Tensor [total_num_anchors]
            候选anchor的布尔掩码（会被原地修改）
        
        返回值：
        -------
        num_fg : int
            最终的正样本总数
        gt_matched_classes : Tensor [num_fg]
            每个正样本对应的GT类别
        pred_ious_this_matching : Tensor [num_fg]
            每个正样本与其匹配GT的IoU值
        matched_gt_inds : Tensor [num_fg]
            每个正样本匹配的GT索引（0到num_gt-1）
        """
        # ========== 第零步：NaN/Inf 防护 ==========
        # 直接使用 nan_to_num，避免条件检查触发 CUDA 断言错误
        cost = torch.nan_to_num(cost, nan=1e8, posinf=1e8, neginf=-1e8)
        pair_wise_ious = torch.nan_to_num(pair_wise_ious, nan=0.0, posinf=1.0, neginf=0.0)
        
        # ========== 第一步：获取候选anchor数量 ==========
        num_in_boxes_anchor = pair_wise_ious.size(1)  # 候选anchor总数
        # DEBUG: print(f"[DEBUG dynamic_k] num_gt={num_gt}, num_in_boxes_anchor={num_in_boxes_anchor}")
        
        # 边界情况：没有候选anchor
        if num_in_boxes_anchor == 0:
            return 0, gt_classes.new_tensor([]), gt_classes.new_tensor([]), gt_classes.new_tensor([], dtype=torch.long)

        # ========== 第二步：初始化匹配矩阵 ==========
        # matching_matrix: [num_gt, num_in_boxes_anchor]
        # 值为1表示该anchor分配给该GT，值为0表示未分配
        matching_matrix = torch.zeros_like(cost)
        if self.debug:
            print(f"[DEBUG dynamic_k] matching_matrix shape: {matching_matrix.shape}")

        # ========== 第三步：计算动态k值 ==========
        # n_candidate_k: 取top-k个IoU最高的anchor来计算k值
        # 限制最大为10，最小为1
        n_candidate_k = min(10, num_in_boxes_anchor)
        n_candidate_k = max(1, n_candidate_k)
        
        # 确保 pair_wise_ious 的维度正确
        if pair_wise_ious.size(1) < n_candidate_k:
            n_candidate_k = max(1, pair_wise_ious.size(1))
        
        # topk_ious: [num_gt, n_candidate_k] 每个GT的top-k个IoU值
        topk_ious, _ = torch.topk(pair_wise_ious, n_candidate_k, dim=1)
        if self.debug:
            print(f"[DEBUG dynamic_k] topk_ious shape: {topk_ious.shape}")
            print(f"[DEBUG dynamic_k] topk_ious per GT: {topk_ious}")
        
        # dynamic_ks: [num_gt] 每个GT的动态k值
        # k = sum(top-k IoU)，即top-k个IoU之和取整
        # 这意味着：IoU越高，k越大，分配的正样本越多
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)
        if self.debug:
            print(f"[DEBUG dynamic_k] dynamic_ks: {dynamic_ks}")

        # ========== 第四步：为每个GT选择代价最小的k个anchor ==========
        for gt_idx in range(num_gt):
            # 获取当前GT的k值，确保不超过可用anchor数量
            k_value = min(dynamic_ks[gt_idx].item(), cost[gt_idx].size(0))
            k_value = max(1, min(k_value, num_in_boxes_anchor))  # 双重保护：确保k_value不超过候选anchor数量
            if self.debug:
                print(f"[DEBUG dynamic_k] GT {gt_idx}: k_value={k_value}")
            
            # 选择代价最小的k个anchor（largest=False表示选最小的）
            # pos_idx: 被选中的anchor在候选anchor中的索引
            _, pos_idx = torch.topk(
                cost[gt_idx], k=k_value, largest=False
            )
            if self.debug:
                print(f"[DEBUG dynamic_k] GT {gt_idx}: pos_idx={pos_idx}")
            
            # 在匹配矩阵中标记这些anchor分配给当前GT
            matching_matrix[gt_idx][pos_idx] = 1.0
        
        # 清理中间变量，释放显存
        del topk_ious, dynamic_ks, pos_idx

        # ========== 第五步：处理一个anchor匹配多个GT的冲突 ==========
        # anchor_matching_gt: [num_in_boxes_anchor] 每个anchor被多少个GT选中
        anchor_matching_gt = matching_matrix.sum(0)
        if self.debug:
            print(f"[DEBUG dynamic_k] anchor_matching_gt range: [{anchor_matching_gt.min().item()}, {anchor_matching_gt.max().item()}]")
        
        # 如果有anchor被多个GT选中（值>1），需要解决冲突
        if (anchor_matching_gt > 1).sum() > 0:
            if self.debug:
                print(f"[DEBUG dynamic_k] 发现 {(anchor_matching_gt > 1).sum().item()} 个冲突anchor")
            
            # 对于冲突的anchor，选择代价最小的GT
            # cost[:, anchor_matching_gt > 1]: 只取冲突anchor的代价
            # cost_argmin: 代价最小的GT索引
            _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
            
            # 先清零所有冲突anchor的匹配
            matching_matrix[:, anchor_matching_gt > 1] *= 0.0
            # 然后只保留代价最小的GT匹配
            matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0
            if self.debug:
                print(f"[DEBUG dynamic_k] 冲突解决完成")

        # ========== 第六步：提取最终匹配结果 ==========
        # fg_mask_inboxes: [num_in_boxes_anchor] 最终被选中的anchor
        fg_mask_inboxes = matching_matrix.sum(0) > 0.0
        num_fg = fg_mask_inboxes.sum().item()  # 正样本总数
        if self.debug:
            print(f"[DEBUG dynamic_k] 最终正样本数: {num_fg}")

        # 更新原始fg_mask（原地修改）
        # 将候选anchor中的正样本标记回原始的total_num_anchors维度
        fg_mask[fg_mask.clone()] = fg_mask_inboxes

        # matched_gt_inds: [num_fg] 每个正样本匹配的GT索引
        # argmax(0)找到每个正样本被哪个GT匹配
        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0).long()
        
        # gt_matched_classes: [num_fg] 每个正样本对应的GT类别
        gt_matched_classes = gt_classes[matched_gt_inds]

        # pred_ious_this_matching: [num_fg] 每个正样本与其GT的IoU
        # matching_matrix * pair_wise_ious 提取匹配位置的IoU
        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[
            fg_mask_inboxes
        ]
        # DEBUG: print(f"[DEBUG dynamic_k] pred_ious range: [{pred_ious_this_matching.min().item():.4f}, {pred_ious_this_matching.max().item():.4f}]")
        
        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds


def is_parallel(model):
    return type(model) in (
        nn.parallel.DataParallel,
        nn.parallel.DistributedDataParallel,
    )


def de_parallel(model):
    return model.module if is_parallel(model) else model


def copy_attr(a, b, include=(), exclude=()):
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith("_") or k in exclude:
            continue
        else:
            setattr(a, k, v)


class ModelEMA:
    """Exponential Moving Average for model parameters"""

    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        self.ema = deepcopy(de_parallel(model)).eval()
        self.updates = updates
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = de_parallel(model).state_dict()
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1 - d) * msd[k].detach()

    def update_attr(self, model, include=(), exclude=("process_group", "reducer")):
        copy_attr(self.ema, model, include, exclude)


def weights_init(net, init_type="normal", init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, "weight") and classname.find("Conv") != -1:
            if init_type == "normal":
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "xavier":
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == "kaiming":
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(
                    "initialization method [%s] is not implemented" % init_type
                )
        elif classname.find("BatchNorm2d") != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    print("initialize network with %s type" % init_type)
    net.apply(init_func)


if __name__ == "__main__":
    print("=" * 60)
    print("Testing YOLOLossOptimized")
    print("=" * 60)
    
    loss_fn = YOLOLossOptimized(
        num_classes=1,
        fp16=False,
        strides=[1, 2, 4],
        center_radius=5.0,
        use_focal=True,
        adaptive_center_radius=True,
    )
    
    B, C, H, W = 2, 6, 60, 80
    inputs = torch.randn(B, C, H, W)
    
    labels = [
        torch.tensor([[40.0, 30.0, 5.0, 5.0, 0.0]]),
        torch.tensor([[20.0, 15.0, 3.0, 4.0, 0.0], [50.0, 40.0, 6.0, 6.0, 0.0]]),
    ]
    
    loss_dict = loss_fn([inputs], labels, return_components=True)
    print(f"Total Loss: {loss_dict['total_loss'].item():.4f}")
    print(f"Box Loss: {loss_dict['loss_box'].item():.4f}")
    print(f"Obj Loss: {loss_dict['loss_obj'].item():.4f}")
    print(f"Cls Loss: {loss_dict['loss_cls'].item():.4f}")
    print(f"Num FG: {loss_dict['num_fg'].item():.1f}")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
