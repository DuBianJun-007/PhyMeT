"""
小目标检测损失函数
===================

专门针对红外小目标检测优化的损失函数：
1. FocalLoss - 处理极端类别不平衡
2. NWDLoss - 归一化Wasserstein距离，适合小目标
3. CIoULoss - 完整IoU损失
4. CombinedDetectionLoss - 组合检测损失
5. ResidualReconstructionLoss - 残差重构损失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple, List, Dict
import math


class FocalLoss(nn.Module):
    """
    Focal Loss - 用于处理极端类别不平衡
    
    公式: FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    通过降低易分类样本的权重，使模型更关注难分类样本。
    
    参数:
        alpha: 正/负样本权重因子 (默认: 0.25)
        gamma: 难/易样本聚焦参数 (默认: 2.0)
        reduction: 归约方式 ('none', 'mean', 'sum')
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        前向传播
        
        参数:
            pred: 预测logits [N] 或 [N, C]
            target: 目标标签 [N] 或 [N, C] (0 或 1)
        
        返回:
            Focal损失值
        """
        pred_sigmoid = torch.sigmoid(pred)
        
        # 计算pt = p(正确类别)
        pt = torch.where(target == 1, pred_sigmoid, 1 - pred_sigmoid)
        
        # 计算alpha权重
        alpha_factor = torch.where(target == 1, self.alpha, 1 - self.alpha)
        
        # 计算focal权重: (1-pt)^gamma
        focal_weight = alpha_factor * (1 - pt).pow(self.gamma)
        
        # 二值交叉熵
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        
        # 最终损失
        loss = focal_weight * bce
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class NWDLoss(nn.Module):
    """
    归一化Wasserstein距离损失 - 专为小目标检测设计
    
    参考: "A Normalized Gaussian Wasserstein Distance for Tiny Object Detection"
    
    将边界框视为2D高斯分布，计算它们的Wasserstein距离。
    对于小目标，这种方法比IoU更合适，因为小目标的IoU对位置偏移非常敏感。
    
    参数:
        constant: 归一化常数 (默认: 12.5，适合小目标)
        reduction: 归约方式
    """
    
    def __init__(
        self,
        constant: float = 12.5,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.constant = constant
        self.reduction = reduction
        
    def forward(
        self,
        pred: Tensor,
        target: Tensor,
    ) -> Tensor:
        """
        计算预测框和目标框之间的归一化Wasserstein距离
        
        参数:
            pred: 预测框 [N, 4]，格式为 (cx, cy, w, h)
            target: 目标框 [N, 4]，格式为 (cx, cy, w, h)
        
        返回:
            NWD损失值
        """
        if pred.numel() == 0 or target.numel() == 0:
            return pred.sum() * 0.0
        
        pred_cx, pred_cy, pred_w, pred_h = pred.unbind(-1)
        target_cx, target_cy, target_w, target_h = target.unbind(-1)
        
        # 中心距离平方
        center_dist_sq = (pred_cx - target_cx).pow(2) + (pred_cy - target_cy).pow(2)
        
        # 方差之和 (将边界框视为高斯分布)
        pred_var = pred_w.pow(2) / 12 + pred_h.pow(2) / 12
        target_var = target_w.pow(2) / 12 + target_h.pow(2) / 12
        var_sum = pred_var + target_var
        
        # Wasserstein距离平方
        wasserstein_dist_sq = center_dist_sq + var_sum
        
        # Wasserstein距离
        wasserstein_dist = wasserstein_dist_sq.sqrt()
        
        # 归一化Wasserstein距离
        nwd = torch.exp(-wasserstein_dist / self.constant)
        
        # 损失 = 1 - NWD
        loss = 1 - nwd
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class CIoULoss(nn.Module):
    """
    Complete IoU Loss - 完整IoU损失
    
    CIoU = IoU - (d² / c²) - αv
    
    其中:
    - d²: 框中心之间的距离平方
    - c²: 包围框对角线的平方
    - α: 权重参数
    - v: 宽高比一致性
    
    相比GIoU，CIoU额外考虑了宽高比的一致性。
    
    参数:
        reduction: 归约方式
        eps: 数值稳定性常数
    """
    
    def __init__(
        self,
        reduction: str = "mean",
        eps: float = 1e-7,
    ) -> None:
        super().__init__()
        self.reduction = reduction
        self.eps = eps
        
    def forward(
        self,
        pred: Tensor,
        target: Tensor,
    ) -> Tensor:
        """
        计算CIoU损失
        
        参数:
            pred: 预测框 [N, 4]，格式为 (cx, cy, w, h)
            target: 目标框 [N, 4]，格式为 (cx, cy, w, h)
        
        返回:
            CIoU损失值
        """
        if pred.numel() == 0:
            return pred.sum() * 0.0
        
        # 将(cx, cy, w, h)转换为(x1, y1, x2, y2)
        pred_x1 = pred[..., 0] - pred[..., 2] / 2
        pred_y1 = pred[..., 1] - pred[..., 3] / 2
        pred_x2 = pred[..., 0] + pred[..., 2] / 2
        pred_y2 = pred[..., 1] + pred[..., 3] / 2
        
        target_x1 = target[..., 0] - target[..., 2] / 2
        target_y1 = target[..., 1] - target[..., 3] / 2
        target_x2 = target[..., 0] + target[..., 2] / 2
        target_y2 = target[..., 1] + target[..., 3] / 2
        
        # 计算交集
        inter_x1 = torch.max(pred_x1, target_x1)
        inter_y1 = torch.max(pred_y1, target_y1)
        inter_x2 = torch.min(pred_x2, target_x2)
        inter_y2 = torch.min(pred_y2, target_y2)
        
        inter_w = (inter_x2 - inter_x1).clamp(min=0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0)
        inter_area = inter_w * inter_h
        
        # 计算并集
        pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
        target_area = (target_x2 - target_x1) * (target_y2 - target_y1)
        union_area = pred_area + target_area - inter_area
        
        # IoU
        iou = inter_area / (union_area + self.eps)
        
        # 中心距离平方
        center_dist_sq = (pred[..., 0] - target[..., 0]).pow(2) + \
                        (pred[..., 1] - target[..., 1]).pow(2)
        
        # 包围框对角线平方
        enclose_x1 = torch.min(pred_x1, target_x1)
        enclose_y1 = torch.min(pred_y1, target_y1)
        enclose_x2 = torch.max(pred_x2, target_x2)
        enclose_y2 = torch.max(pred_y2, target_y2)
        
        enclose_diag_sq = (enclose_x2 - enclose_x1).pow(2) + \
                         (enclose_y2 - enclose_y1).pow(2)
        
        # 宽高比一致性
        v = (4 / (math.pi ** 2)) * \
            (torch.atan(target[..., 2] / (target[..., 3] + self.eps)) - \
             torch.atan(pred[..., 2] / (pred[..., 3] + self.eps))).pow(2)
        
        # 计算alpha权重
        with torch.no_grad():
            alpha = v / (1 - iou + v + self.eps)
        
        # CIoU
        ciou = iou - center_dist_sq / (enclose_diag_sq + self.eps) - alpha * v
        
        # 损失 = 1 - CIoU
        loss = 1 - ciou
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class CombinedDetectionLoss(nn.Module):
    """
    组合检测损失 - 用于小目标检测
    
    组合以下损失:
    1. 边界框回归损失 (NWD + CIoU)
    2. 目标性损失 (Focal)
    3. 分类损失 (Focal)
    
    参数:
        num_classes: 检测类别数
        nwd_weight: NWD损失权重
        ciou_weight: CIoU损失权重
        obj_weight: 目标性损失权重
        cls_weight: 分类损失权重
        focal_alpha: Focal损失alpha参数
        focal_gamma: Focal损失gamma参数
        use_nwd: 是否使用NWD损失
    """
    
    def __init__(
        self,
        num_classes: int = 1,
        nwd_weight: float = 1.0,
        ciou_weight: float = 2.0,
        obj_weight: float = 1.0,
        cls_weight: float = 1.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        use_nwd: bool = True,
    ) -> None:
        super().__init__()
        
        self.num_classes = num_classes
        self.nwd_weight = nwd_weight
        self.ciou_weight = ciou_weight
        self.obj_weight = obj_weight
        self.cls_weight = cls_weight
        self.use_nwd = use_nwd
        
        self.nwd_loss = NWDLoss(reduction="none") if use_nwd else None
        self.ciou_loss = CIoULoss(reduction="none")
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, reduction="none")
        
    def forward(
        self,
        pred_boxes: Tensor,
        pred_obj: Tensor,
        pred_cls: Tensor,
        target_boxes: Tensor,
        target_obj: Tensor,
        target_cls: Tensor,
        fg_mask: Optional[Tensor] = None,
    ) -> dict:
        """
        计算组合检测损失
        
        参数:
            pred_boxes: 预测框 [N, 4]，格式为 (cx, cy, w, h)
            pred_obj: 预测目标性 [N, 1]
            pred_cls: 预测分类 [N, num_classes]
            target_boxes: 目标框 [M, 4]
            target_obj: 目标目标性 [N, 1]
            target_cls: 目标分类 [N, num_classes] (one-hot)
            fg_mask: 前景掩码 [N]
        
        返回:
            损失值字典
        """
        losses = {}
        
        # 边界框回归损失
        if fg_mask is not None and fg_mask.sum() > 0:
            fg_pred_boxes = pred_boxes[fg_mask]
            fg_target_boxes = target_boxes
            
            if fg_pred_boxes.shape[0] > 0 and fg_target_boxes.shape[0] > 0:
                ciou_loss = self.ciou_loss(fg_pred_boxes, fg_target_boxes)
                losses["loss_ciou"] = self.ciou_weight * ciou_loss.mean()
                
                if self.use_nwd and self.nwd_loss is not None:
                    nwd_loss = self.nwd_loss(fg_pred_boxes, fg_target_boxes)
                    losses["loss_nwd"] = self.nwd_weight * nwd_loss.mean()
                else:
                    losses["loss_nwd"] = torch.tensor(0.0, device=pred_boxes.device)
            else:
                losses["loss_ciou"] = torch.tensor(0.0, device=pred_boxes.device)
                losses["loss_nwd"] = torch.tensor(0.0, device=pred_boxes.device)
        else:
            losses["loss_ciou"] = torch.tensor(0.0, device=pred_boxes.device)
            losses["loss_nwd"] = torch.tensor(0.0, device=pred_boxes.device)
        
        # 目标性损失
        obj_loss = self.focal_loss(pred_obj.squeeze(-1), target_obj.squeeze(-1))
        losses["loss_obj"] = self.obj_weight * obj_loss.mean()
        
        # 分类损失
        if self.num_classes > 0 and pred_cls is not None:
            if fg_mask is not None and fg_mask.sum() > 0:
                fg_pred_cls = pred_cls[fg_mask]
                fg_target_cls = target_cls[fg_mask] if target_cls.dim() > 1 else target_cls
                
                if fg_pred_cls.shape[0] > 0:
                    cls_loss = self.focal_loss(fg_pred_cls, fg_target_cls)
                    losses["loss_cls"] = self.cls_weight * cls_loss.mean()
                else:
                    losses["loss_cls"] = torch.tensor(0.0, device=pred_boxes.device)
            else:
                losses["loss_cls"] = torch.tensor(0.0, device=pred_boxes.device)
        else:
            losses["loss_cls"] = torch.tensor(0.0, device=pred_boxes.device)
        
        # 总损失
        total_loss = losses["loss_ciou"] + losses.get("loss_nwd", 0) + \
                    losses["loss_obj"] + losses["loss_cls"]
        losses["loss_total"] = total_loss
        
        return losses


class QualityFocalLoss(nn.Module):
    """
    Quality Focal Loss (QFL) - 质量焦点损失
    
    将分类分数与定位质量(IoU)结合。
    用于GFL (Generalized Focal Loss)。
    
    参数:
        beta: 聚焦参数
        use_sigmoid: 是否使用sigmoid
    """
    
    def __init__(
        self,
        beta: float = 2.0,
        use_sigmoid: bool = True,
    ) -> None:
        super().__init__()
        self.beta = beta
        self.use_sigmoid = use_sigmoid
        
    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        quality: Tensor,
    ) -> Tensor:
        """
        前向传播
        
        参数:
            pred: 预测分类分数 [N, num_classes]
            target: 目标标签 [N] (类别索引)
            quality: 定位质量(IoU) [N]
        
        返回:
            Quality focal损失
        """
        pred_sigmoid = torch.sigmoid(pred)
        
        scale = pred_sigmoid - quality.unsqueeze(-1)
        
        target_one_hot = F.one_hot(target, pred.shape[-1]).float()
        
        focal_weight = scale.abs().pow(self.beta) * target_one_hot
        
        bce = F.binary_cross_entropy_with_logits(pred, quality.unsqueeze(-1).expand_as(pred), reduction="none")
        
        loss = focal_weight * bce
        
        return loss.sum() / (target_one_hot.sum() + 1e-6)


class ResidualReconstructionLoss(nn.Module):
    """
    残差重构损失（简化版）
    
    基于物理约束: 原始图像 = 背景 + 目标
    
    包含以下损失项:
    1. 全局重构损失: 背景图 + 目标图 ≈ 原始图像
    2. 目标稀疏性损失: 目标图在背景区域应为0
    3. 目标内容损失: 目标图在目标区域应匹配原始图像
    4. 背景补全损失: 背景图在目标区域应是周围背景的平滑延伸
    
    参数:
        alpha: 目标稀疏性损失权重
        beta: 目标内容损失权重
        gamma: 背景补全损失权重
        inpaint_kernel_size: 背景补全损失的核大小（应大于目标尺寸）
    """
    
    def __init__(
        self,
        alpha: float = 0.1,
        beta: float = 2.0,
        gamma: float = 0.5,
        inpaint_kernel_size: int = 15,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.inpaint_kernel_size = inpaint_kernel_size
    
    def forward(
        self,
        img: Tensor,
        rec_bg: Tensor,
        rec_tgt: Tensor,
        mask: Tensor,
    ) -> Dict[str, Tensor]:
        """
        计算残差重构损失
        
        参数:
            img: 原始图像 [B, 1, H, W]
            rec_bg: 背景分支解码图像 [B, 1, H, W]
            rec_tgt: 目标分支解码图像 [B, 1, H, W]
            mask: GT掩码 [B, 1, H, W] (0 或 1)
        
        返回:
            包含各项损失值的字典
        """
        loss_global = F.l1_loss(rec_bg + rec_tgt, img)
        
        # 改进的稀疏性损失：使用 L2 而不是 L1，梯度更平缓
        bg_region = 1 - mask
        loss_tgt_sparse = torch.mean((bg_region * rec_tgt) ** 2)
        
        mask_sum = mask.sum()
        if mask_sum > 0:
            # loss_tgt_content = F.mse_loss(mask * rec_tgt, mask * img)
            diff = (mask * (rec_tgt - img)).pow(2)
            loss_tgt_content = diff.sum() / (mask_sum + 1e-6)
        else:
            loss_tgt_content = torch.tensor(0.0, device=img.device, requires_grad=True)
        
        loss_bg_inpaint = self.background_inpainting_loss(img, rec_bg, mask)
        
        total_loss = (
            loss_global
            + self.alpha * loss_tgt_sparse
            + self.beta * loss_tgt_content
            + self.gamma * loss_bg_inpaint
        )
        
        return {
            'loss_residual_recon': total_loss,
            'loss_global': loss_global,
            'loss_tgt_sparse': loss_tgt_sparse,
            'loss_tgt_content': loss_tgt_content,
            'loss_bg_inpaint': loss_bg_inpaint,
        }
    
    def background_inpainting_loss(self, img: Tensor, rec_bg: Tensor, mask: Tensor) -> Tensor:
        """
        背景补全损失：目标区域的背景应是周围背景的平滑延伸
        
        思路：目标区域的背景值应接近周围背景的局部均值
        
        参数:
            img: 原始图像 [B, 1, H, W]
            rec_bg: 背景重构图像 [B, 1, H, W]
            mask: 目标掩码 [B, 1, H, W]
        
        返回:
            背景补全损失值
        """
        kernel_size = self.inpaint_kernel_size
        padding = kernel_size // 2
        
        bg_mask = 1 - mask
        
        bg_sum = F.avg_pool2d(img * bg_mask, kernel_size, stride=1, padding=padding)
        bg_count = F.avg_pool2d(bg_mask, kernel_size, stride=1, padding=padding) + 1e-6
        local_bg_mean = bg_sum / bg_count
        
        mask_sum = mask.sum()
        if mask_sum > 0:
            # 只对目标区域（mask=1）求均值，避免全图平均导致信号稀释
            # F.l1_loss 默认分母是全图像素数，目标区域极少时梯度接近零
            diff = (mask * (rec_bg - local_bg_mean)).abs()
            return diff.sum() / (mask_sum + 1e-6)
        return torch.tensor(0.0, device=img.device, requires_grad=True)


if __name__ == "__main__":
    print("=" * 60)
    print("测试损失函数")
    print("=" * 60)
    
    # 测试数据
    pred_boxes = torch.randn(10, 4)
    pred_boxes[:, 2:4] = pred_boxes[:, 2:4].abs() + 0.5
    target_boxes = torch.randn(10, 4)
    target_boxes[:, 2:4] = target_boxes[:, 2:4].abs() + 0.5
    
    # 测试NWDLoss
    nwd_loss = NWDLoss()
    loss = nwd_loss(pred_boxes, target_boxes)
    print(f"NWDLoss: {loss.item():.4f}")
    
    # 测试CIoULoss
    ciou_loss = CIoULoss()
    loss = ciou_loss(pred_boxes, target_boxes)
    print(f"CIoULoss: {loss.item():.4f}")
    
    # 测试FocalLoss
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
    pred = torch.randn(100)
    target = (torch.rand(100) > 0.9).float()
    loss = focal_loss(pred, target)
    print(f"FocalLoss: {loss.item():.4f}")
    
    print("\n" + "=" * 60)
    print("所有测试通过!")
    print("=" * 60)
