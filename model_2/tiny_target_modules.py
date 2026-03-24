"""
Tiny Target Detection Modules
==============================

Specialized modules for infrared small target detection:
1. TinyTargetAttention - Attention module optimized for small targets
2. LocalContrastModule - Local contrast enhancement
3. SpatialAttentionEnhanced - Enhanced spatial attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional


class TinyTargetAttention(nn.Module):
    """
    Tiny Target Attention Module
    
    Optimized for detecting small targets (< 7x7 pixels):
    1. Uses small kernels (1x1, 3x3) to preserve local details
    2. Enhances local contrast for dim targets
    3. Applies spatial attention to highlight potential target regions
    
    Args:
        channels: Number of input channels
        reduction: Channel reduction ratio for attention
        kernel_size: Kernel size for local feature extraction
    """
    
    def __init__(
        self,
        channels: int,
        reduction: int = 4,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        
        self.channels = channels
        self.reduction = reduction
        
        mid_channels = max(channels // reduction, 16)
        
        self.local_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        
        self.point_conv = nn.Sequential(
            nn.Conv2d(channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
            nn.Sigmoid(),
        )
        
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        
        local_feat = self.local_conv(x)
        
        point_feat = self.point_conv(local_feat)
        
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_out, max_out], dim=1)
        spatial_weight = self.spatial_attention(spatial_input)
        
        enhanced = point_feat * spatial_weight
        
        out = x + self.gamma * enhanced
        
        return out


class LocalContrastModule(nn.Module):
    """
    Local Contrast Enhancement Module
    
    Enhances local contrast to highlight dim small targets against background.
    Uses difference of Gaussian-like filtering to boost local contrast.
    
    Args:
        channels: Number of input channels
        kernel_sizes: List of kernel sizes for multi-scale contrast
    """
    
    def __init__(
        self,
        channels: int,
        kernel_sizes: list = [3, 5, 7],
    ) -> None:
        super().__init__()
        
        self.channels = channels
        self.kernel_sizes = kernel_sizes
        
        self.convs = nn.ModuleList()
        for k in kernel_sizes:
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(channels, channels, kernel_size=k, padding=k // 2, groups=channels, bias=False),
                    nn.BatchNorm2d(channels),
                )
            )
        
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * len(kernel_sizes), channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        
        self.alpha = nn.Parameter(torch.ones(len(kernel_sizes)) / len(kernel_sizes))
        
    def forward(self, x: Tensor) -> Tensor:
        contrast_feats = []
        
        for i, conv in enumerate(self.convs):
            local_avg = conv(x)
            contrast = x - local_avg
            contrast_feats.append(contrast)
        
        weights = F.softmax(self.alpha, dim=0)
        weighted_feats = [w * f for w, f in zip(weights, contrast_feats)]
        
        concat_feat = torch.cat(weighted_feats, dim=1)
        out = self.fusion(concat_feat)
        
        return out + x


class SpatialAttentionEnhanced(nn.Module):
    """
    Enhanced Spatial Attention Module
    
    Improved spatial attention with:
    1. Multi-scale context aggregation
    2. Dilated convolutions for larger receptive field
    3. Learnable attention weights
    
    Args:
        kernel_size: Base kernel size
        dilations: List of dilation rates
    """
    
    def __init__(
        self,
        kernel_size: int = 7,
        dilations: list = [1, 2, 3],
    ) -> None:
        super().__init__()
        
        self.kernel_size = kernel_size
        self.dilations = dilations
        
        self.convs = nn.ModuleList()
        for d in dilations:
            padding = (kernel_size + (kernel_size - 1) * (d - 1) - 1) // 2
            self.convs.append(
                nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, dilation=d, bias=False)
            )
        
        self.fusion = nn.Conv2d(len(dilations), 1, kernel_size=1, bias=False)
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: Tensor) -> Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_out, max_out], dim=1)
        
        multi_scale_attn = []
        for conv in self.convs:
            multi_scale_attn.append(conv(spatial_input))
        
        concat_attn = torch.cat(multi_scale_attn, dim=1)
        attn = self.fusion(concat_attn)
        attn = self.sigmoid(attn)
        
        return x * attn


class DilatedConvBlock(nn.Module):
    """
    Dilated Convolution Block for multi-scale feature extraction
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        dilations: List of dilation rates
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dilations: list = [1, 2, 4],
    ) -> None:
        super().__init__()
        
        self.branches = nn.ModuleList()
        for d in dilations:
            padding = d
            self.branches.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels // len(dilations), 
                             kernel_size=3, padding=padding, dilation=d, bias=False),
                    nn.BatchNorm2d(out_channels // len(dilations)),
                    nn.ReLU(inplace=True),
                )
            )
        
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x: Tensor) -> Tensor:
        branch_outputs = [branch(x) for branch in self.branches]
        concat = torch.cat(branch_outputs, dim=1)
        return self.fusion(concat)


class FeatureRefinementModule(nn.Module):
    """
    Feature Refinement Module for small target detection
    
    Combines:
    1. TinyTargetAttention for local enhancement
    2. LocalContrastModule for contrast enhancement
    3. SpatialAttentionEnhanced for spatial refinement
    
    Args:
        channels: Number of input/output channels
    """
    
    def __init__(self, channels: int) -> None:
        super().__init__()
        
        self.tiny_attention = TinyTargetAttention(channels)
        self.local_contrast = LocalContrastModule(channels)
        self.spatial_attention = SpatialAttentionEnhanced()
        
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x: Tensor) -> Tensor:
        attn_feat = self.tiny_attention(x)
        contrast_feat = self.local_contrast(x)
        
        concat = torch.cat([attn_feat, contrast_feat], dim=1)
        fused = self.fusion(concat)
        
        refined = self.spatial_attention(fused)
        
        return refined


class SmallTargetHead(nn.Module):
    """
    Specialized detection head for small targets
    
    Features:
    1. Smaller kernel sizes to preserve spatial details
    2. Deeper feature processing
    3. Separate regression and classification branches
    
    Args:
        in_channels: Number of input channels
        num_classes: Number of detection classes
    """
    
    def __init__(
        self,
        in_channels: int,
        num_classes: int = 1,
    ) -> None:
        super().__init__()
        
        self.num_classes = num_classes
        
        hidden_channels = in_channels
        
        self.reg_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        
        self.obj_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        
        self.cls_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        
        self.reg_pred = nn.Conv2d(hidden_channels, 4, kernel_size=1)
        self.obj_pred = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.cls_pred = nn.Conv2d(hidden_channels, num_classes, kernel_size=1) if num_classes > 0 else None
        
        self._init_weights()
        
    def _init_weights(self):
        import math
        prior_prob = 0.05
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.obj_pred.bias, bias_value)
        if self.cls_pred is not None:
            nn.init.constant_(self.cls_pred.bias, bias_value)
            
    def forward(self, x: Tensor) -> dict:
        return {
            "reg": self.reg_pred(self.reg_conv(x)),
            "obj": self.obj_pred(self.obj_conv(x)),
            "cls": self.cls_pred(self.cls_conv(x)) if self.cls_pred is not None else None,
        }


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Tiny Target Modules")
    print("=" * 60)
    
    x = torch.randn(2, 256, 60, 80)
    
    tta = TinyTargetAttention(256)
    out = tta(x)
    print(f"TinyTargetAttention: {x.shape} -> {out.shape}")
    
    lcm = LocalContrastModule(256)
    out = lcm(x)
    print(f"LocalContrastModule: {x.shape} -> {out.shape}")
    
    sae = SpatialAttentionEnhanced()
    out = sae(x)
    print(f"SpatialAttentionEnhanced: {x.shape} -> {out.shape}")
    
    frm = FeatureRefinementModule(256)
    out = frm(x)
    print(f"FeatureRefinementModule: {x.shape} -> {out.shape}")
    
    sth = SmallTargetHead(256, num_classes=1)
    out = sth(x)
    print(f"SmallTargetHead: reg={out['reg'].shape}, obj={out['obj'].shape}, cls={out['cls'].shape if out['cls'] is not None else None}")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
