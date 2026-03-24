"""
MemISTD Small Target Detection Network
=======================================

Optimized for infrared small target detection (target size < 7x7 pixels)

Key Improvements:
1. Multi-scale detection at P0/P1/P2 (full resolution, 1/2, 1/4)
2. TinyTargetAttention module for small target enhancement
3. Enhanced feature extraction with larger receptive field control
4. Memory-augmented mechanism for target/background separation

Architecture:
    Input Image (B, C, H, W)
        |
    [U-Net Backbone] -> Multi-scale encoder features [E0, E1, E2]
        |
    [Feature Split Branch] -> Global, Target, Background
        |
    [Memory Modules] -> Target Memory + Background Memory
        |
    [Fusion Module] -> Attention-based fusion
        |
    [Multi-Scale FPN] -> [P0: H×W, P1: H/2×W/2, P2: H/4×W/4]
        |
    [Detection Heads] -> Detection at 3 scales
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, tensor
from typing import Optional, Tuple, Dict, List
import math
from einops import rearrange

from .tiny_target_modules import (
    TinyTargetAttention,
    FeatureRefinementModule,
    SmallTargetHead,
)


def get_activation(act_type: str = "relu"):
    """Get activation function"""
    if act_type == "relu":
        return nn.ReLU(inplace=True)
    elif act_type == "silu" or act_type == "swish":
        return nn.SiLU(inplace=True)
    elif act_type == "mish":
        return nn.Mish(inplace=True)
    elif act_type == "leaky_relu":
        return nn.LeakyReLU(0.1, inplace=True)
    else:
        return nn.ReLU(inplace=True)


def get_norm_layer(norm_type: str, num_channels: int, num_groups: int = 32):
    """Get normalization layer"""
    if norm_type == "bn":
        return nn.BatchNorm2d(num_channels)
    elif norm_type == "gn":
        num_groups = min(num_groups, num_channels)
        return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)
    elif norm_type == "in":
        return nn.InstanceNorm2d(num_channels)
    else:
        return nn.Identity()


class DoubleConv(nn.Module):
    """
    Double Convolution Block with residual connection

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        norm_type: Normalization type ('bn', 'gn', 'in')
        act_type: Activation type ('relu', 'silu', 'mish')
        use_residual: Whether to use residual connection
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_type: str = "bn",
        act_type: str = "relu",
        use_residual: bool = True,
    ) -> None:
        super().__init__()
        self.use_residual = use_residual and (in_channels == out_channels)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            get_norm_layer(norm_type, out_channels),
            get_activation(act_type),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            get_norm_layer(norm_type, out_channels),
        )
        self.act = get_activation(act_type)

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, bias=False),
                get_norm_layer(norm_type, out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        if self.use_residual:
            out = out + identity
        out = self.act(out)
        return out


class MemoryDecoder(nn.Module):
    """
    Memory Decoder Module

    Decodes memory module output features to image space.
    Input feature map has the same spatial size as the original image,
    so no upsampling is needed.

    Args:
        in_channels: Number of input feature channels (default: 128)
        out_channels: Number of output image channels (default: 1 for grayscale)
        hidden_channels: Number of hidden layer channels
    """

    def __init__(
        self,
        in_channels: int = 128,
        out_channels: int = 1,
        hidden_channels: int = 64,
    ) -> None:
        super().__init__()

        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden_channels, hidden_channels // 2,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels // 2),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden_channels // 2, hidden_channels //
                      4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels // 4),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden_channels // 4, out_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input feature map [B, in_channels, H, W]

        Returns:
            Decoded image [B, out_channels, H, W] in range [0, 1]
        """
        return self.decoder(x)


class CoordinateAttention(nn.Module):
    """
    Coordinate Attention Module

    Captures long-range spatial dependencies with low computational cost.
    Decomposes channel attention into two 1D feature encoding processes
    along horizontal and vertical directions.

    Reference: Hou et al., "Coordinate Attention for Efficient Mobile Network Design", CVPR 2021

    Args:
        inp: Number of input channels
        oup: Number of output channels
        reduction: Reduction ratio for intermediate channels
    """

    def __init__(self, inp: int, oup: int, reduction: int = 32) -> None:
        super(CoordinateAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        n, c, h, w = x.size()

        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_h * a_w

        return out


class _BaseMemoryBranch(nn.Module):
    """
    Base class for multi-scale memory branch

    Uses patch-based memory reading with unfold/fold operations.

    Args:
        in_channels: Number of input channels
        num_memories: Number of memory slots
        patch_size: Size of the patch for memory reading
    """

    def __init__(
        self, 
        in_channels: int, 
        num_memories: int, 
        patch_size: int,
        similarity_type: str = 'dot'
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.padding = patch_size // 2
        self.mem_dim = in_channels * patch_size * patch_size
        self.memory = nn.Parameter(torch.randn(num_memories, self.mem_dim))
        self.temperature = nn.Parameter(torch.ones(1) * 10.0)
        self.similarity_type = similarity_type

    def _read_memory(self, x: Tensor) -> Tuple[Tensor, Tuple[int, int, int, int]]:
        """
        Read from memory using patch-based attention.
        
        Args:
            x: Input feature map [B, C, H, W]
        
        Returns:
            read_out: Memory read output [B, H*W, mem_dim]
            shape: Original shape (B, C, H, W)
        """
        B, C, H, W = x.shape
        x_unfold = F.unfold(x, kernel_size=self.patch_size, padding=self.padding, stride=1)
        x_unfold = x_unfold.transpose(1, 2)
        
        if self.similarity_type == 'cosine':
            x_norm = F.normalize(x_unfold, dim=2)
            mem_norm = F.normalize(self.memory, dim=1)
            sim = torch.matmul(x_norm, mem_norm.t()) * self.temperature
        elif self.similarity_type == 'dot':
            # 缩放点积相似度，防止数值过大
            scale = 1.0 / (self.mem_dim ** 0.5)  # sqrt(D)
            sim = torch.matmul(x_unfold, self.memory.t()) * scale * self.temperature

        att = F.softmax(sim, dim=2)
        read_out = torch.matmul(att, self.memory)
        
        return read_out, (B, C, H, W)


class BackgroundBranch(_BaseMemoryBranch):
    """
    Background memory branch with fold-based reconstruction.

    Reconstructs background features by folding memory read output.
    """

    def forward(self, x: Tensor) -> Tensor:
        read_out, (B, C, H, W) = self._read_memory(x)
        read_out = read_out.transpose(1, 2)
        out_sum = F.fold(read_out, output_size=(H, W), kernel_size=self.patch_size, padding=self.padding, stride=1)
        ones = torch.ones(1, 1, H, W, device=x.device)
        ones_unfold = F.unfold(ones, kernel_size=self.patch_size, padding=self.padding, stride=1)
        divisor = F.fold(ones_unfold, output_size=(H, W), kernel_size=self.patch_size, padding=self.padding, stride=1)
        return out_sum / (divisor + 1e-8)


class TargetBranch(_BaseMemoryBranch):
    """
    Target memory branch with center-pixel extraction.

    Extracts center pixel from each patch for target detection.
    Feature normalization is applied in _read_memory to stabilize output value range.
    """

    # 采用中心对齐方案
    # def forward(self, x: Tensor) -> Tensor:
    #     read_out, (B, C, H, W) = self._read_memory(x)
    #     read_out = read_out.view(B, H * W, C, self.patch_size, self.patch_size)
    #     center_idx = self.patch_size // 2
    #     out_center = read_out[:, :, :, center_idx, center_idx]
    #     out_center = out_center.transpose(1, 2).view(B, C, H, W)
        
    #     return out_center
    
    def forward(self, x: Tensor) -> Tensor:  # 和背景一个处理逻辑
        read_out, (B, C, H, W) = self._read_memory(x)
        read_out = read_out.transpose(1, 2)
        out_sum = F.fold(read_out, output_size=(H, W), kernel_size=self.patch_size, padding=self.padding, stride=1)
        ones = torch.ones(1, 1, H, W, device=x.device)
        ones_unfold = F.unfold(ones, kernel_size=self.patch_size, padding=self.padding, stride=1)
        divisor = F.fold(ones_unfold, output_size=(H, W), kernel_size=self.patch_size, padding=self.padding, stride=1)
        return out_sum / (divisor + 1e-8)


class MultiScaleFusion(nn.Module):
    """
    Multi-scale fusion module for combining different patch size results.

    Supports arbitrary number of scales (adaptive to input list length).

    Args:
        channels: Number of channels
        num_scales: Number of scales (default 3 for [3,5,7])
        fusion_type: Fusion type ('cat' or 'attention')
    """

    def __init__(self, channels: int, num_scales: int = 3, fusion_type: str = 'attention') -> None:
        super().__init__()
        self.fusion_type = fusion_type
        self.num_scales = num_scales
        self.channels = channels

        if fusion_type == 'cat':
            self.fusion_conv = nn.Sequential(
                nn.Conv2d(channels * num_scales, channels,
                          kernel_size=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            )
        elif fusion_type == 'attention':
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Sequential(
                nn.Linear(channels, channels // 4),
                nn.ReLU(inplace=True),
                nn.Linear(channels // 4, channels * num_scales)
            )
            self.softmax = nn.Softmax(dim=1)

    def forward(self, features: List[Tensor]) -> Tensor:
        """
        Forward pass with list of features.

        Args:
            features: List of feature tensors, each [B, C, H, W]

        Returns:
            Fused feature tensor [B, C, H, W]
        """
        num_features = len(features)
        if num_features == 0:
            raise ValueError("features list cannot be empty")

        if num_features != self.num_scales:
            if hasattr(self, 'fusion_conv'):
                self.fusion_conv = nn.Sequential(
                    nn.Conv2d(self.channels * num_features,
                              self.channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(self.channels),
                    nn.ReLU(inplace=True)
                ).to(features[0].device)
            if hasattr(self, 'fc'):
                self.fc = nn.Sequential(
                    nn.Linear(self.channels, self.channels // 4),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.channels // 4, self.channels * num_features)
                ).to(features[0].device)
            self.num_scales = num_features

        if self.fusion_type == 'cat':
            return self.fusion_conv(torch.cat(features, dim=1))
        elif self.fusion_type == 'attention':
            B, C, H, W = features[0].shape
            stack = torch.stack(features, dim=1)
            U = sum(features)
            w = self.softmax(self.fc(self.avg_pool(U).view(B, C)).view(
                B, num_features, C)).unsqueeze(-1).unsqueeze(-1)
            return (stack * w).sum(dim=1)
        return features[0]


class ArithmeticFusion(nn.Module):
    """
    算术融合模块 - 用于全局特征、背景特征、目标特征的交互融合

    基于物理约束设计多种融合策略:

    1. physics_pure (纯物理策略):
       Out = Global - alpha * BC + beta * TG
       直接从全局特征中减去背景、加上目标

    2. physics_plus (增强物理策略):
       Out = Global + alpha * (Global - BC) + beta * TG
       先计算残差(Global-BC)，再增强

    3. gate (门控策略):
       基于背景特征生成软门控掩码，抑制背景区域

    4. spatial_reweight (空间重加权策略，推荐):
       通过空间注意力机制，自适应地增强目标区域、抑制背景区域
       - target_weight: 目标区域的空间权重 (增强)
       - background_weight: 背景区域的空间权重 (抑制)
       - 最终输出: Global * (1 + target_weight) * (1 - background_weight)

    参数:
        channels: 输入通道数
        strategy: 融合策略 ('physics_pure', 'physics_plus', 'gate', 'spatial_reweight')
    """

    def __init__(self, channels: int, strategy: str = 'spatial_reweight') -> None:
        super().__init__()
        self.strategy = strategy
        self.channels = channels

        # 可学习的融合权重参数
        # alpha: 控制背景抑制强度
        # beta: 控制目标增强强度
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(1.0))

        if strategy == 'spatial_reweight':
            # 空间重加权策略: 使用1x1卷积生成空间权重图
            self.target_proj = nn.Conv2d(
                channels, channels, kernel_size=1)     # 目标特征投影
            self.background_proj = nn.Conv2d(
                channels, channels, kernel_size=1)  # 背景特征投影
            self.spatial_attention = SpatialAttention(
                kernel_size=7)            # 空间注意力
        elif strategy == 'gate':
            # 门控策略: 使用1x1卷积生成背景门控掩码
            self.gate_conv = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, f_global: Tensor, f_bc: Tensor, f_tg: Tensor) -> Tensor:
        """
        前向传播 - 融合全局、背景、目标三种特征

        参数:
            f_global: 全局特征 [B, C, H, W]，来自ViT分支
            f_bc: 背景特征 [B, C, H, W]，来自背景记忆分支
            f_tg: 目标特征 [B, C, H, W]，来自目标记忆分支

        返回:
            融合后的特征 [B, C, H, W]
        """
        if self.strategy == 'physics_pure':
            # 纯物理策略: 直接加减
            # 原理: 原始图像 = 背景 + 目标，所以目标 = 原始 - 背景
            out = f_global - self.alpha * f_bc + self.beta * f_tg

        elif self.strategy == 'physics_plus':
            # 增强物理策略: 先计算残差再融合
            # residual = Global - BC 表示"去除背景后的信息"
            out = f_global + self.alpha * (f_global - f_bc) + self.beta * f_tg

        elif self.strategy == 'gate':
            # 门控策略: 基于背景特征生成软掩码
            # bg_mask 高的地方表示"这里是背景"，需要抑制
            bg_mask = torch.sigmoid(self.gate_conv(f_bc))
            suppressed_global = f_global * (1 - bg_mask * self.alpha)
            out = suppressed_global + self.beta * f_tg

        elif self.strategy == 'spatial_reweight':
            # 空间重加权策略 (推荐)
            # target_weight: 目标区域权重，高的地方增强  [0, 1]
            # background_weight: 背景区域权重，高的地方抑制 [0, 1]
            target_weight = torch.sigmoid(self.target_proj(f_tg))
            background_weight = torch.sigmoid(self.background_proj(f_bc))

            # 解耦式互斥抑制
            # 原逻辑的问题：gate_factor = exp(-k * tw) 在目标强时让背景权重趋近0，
            # 导致 (1 - bg_gated) ≈ 1，背景抑制完全失效。
            # 即使"目标强且背景也强"的区域，背景也无法被抑制。
            #
            # 修复思路：背景有效抑制强度 = 背景响应 × (1 - 目标响应)
            #   - 目标弱、背景强：effective_bg ≈ bg（大），抑制最强 ✅
            #   - 目标强、背景弱：effective_bg ≈ 0，不抑制目标自身 ✅
            #   - 目标强、背景也强：effective_bg ≈ 0，以目标增强为主，
            #     避免在目标像素处将其自身压低 ✅
            #
            # 三种典型值验证（tw=target_weight, bg=background_weight）：
            #   tw=0.0, bg=0.9 → eff=0.90, out = 1.0 * 0.10 = 0.10  (强抑制背景)
            #   tw=0.9, bg=0.1 → eff=0.01, out = 1.9 * 0.99 ≈ 1.88  (强增强目标)
            #   tw=0.9, bg=0.9 → eff=0.09, out = 1.9 * 0.91 ≈ 1.73  (增强目标为主)
            effective_bg_suppress = background_weight * (1.0 - target_weight)

            # 融合公式:
            # (1 + target_weight)        : 目标区域增强，范围 [1, 2]
            # (1 - effective_bg_suppress): 背景区域抑制，范围 [0, 1]
            out = f_global * (1 + target_weight) * (1 - effective_bg_suppress)

            out = self.spatial_attention(out)
        else:
            # 默认: 直接返回全局特征
            out = f_global

        return out


class DualMemorySystem(nn.Module):
    """
    Dual Memory System with multi-scale memory branches.

    Combines background and target memory branches with adaptive multi-scale fusion.

    Args:
        in_channels: Number of input channels
        num_bg_memories: Number of background memory slots (more for complex background)
        num_tg_memories: Number of target memory slots (fewer for simple targets)
        bg_patch_sizes: List of patch sizes for background memory branches (default: [3, 5, 7])
        tg_patch_sizes: List of patch sizes for target memory branches (default: [3, 5, 7])
        ms_fusion_type: Multi-scale fusion type ('cat' or 'attention')
    """

    def __init__(
        self,
        in_channels: int,
        num_bg_memories: int = 64,
        num_tg_memories: int = 8,
        bg_patch_sizes: List[int] = [3, 5, 7],
        tg_patch_sizes: List[int] = [3, 5, 7],
        ms_fusion_type: str = 'attention',
    ) -> None:
        super().__init__()

        self.bg_patch_sizes = bg_patch_sizes
        self.num_bg_scales = len(bg_patch_sizes)
        self.bg_branches = nn.ModuleList([
            BackgroundBranch(in_channels, num_bg_memories, ps) for ps in bg_patch_sizes
        ])
        self.bg_fusion = MultiScaleFusion(
            in_channels, num_scales=self.num_bg_scales, fusion_type=ms_fusion_type)

        self.tg_patch_sizes = tg_patch_sizes
        self.num_tg_scales = len(tg_patch_sizes)
        self.tg_branches = nn.ModuleList([
            TargetBranch(in_channels, num_tg_memories, ps) for ps in tg_patch_sizes
        ])
        self.tg_fusion = MultiScaleFusion(
            in_channels, num_scales=self.num_tg_scales, fusion_type=ms_fusion_type)

    def forward(self, bg: Tensor, tg: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward pass.

        Args:
            bg: Background input feature [B, C, H, W]
            tg: Target input feature [B, C, H, W]

        Returns:
            f_bc: Background feature
            f_tg: Target feature
        """
        bg_features = [branch(bg) for branch in self.bg_branches]
        f_bc = self.bg_fusion(bg_features)

        tg_features = [branch(tg) for branch in self.tg_branches]
        f_tg = self.tg_fusion(tg_features)

        return f_bc, f_tg


class ChannelAttention(nn.Module):
    """Channel Attention Module"""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        avg_out = self.mlp(self.avg_pool(x).view(x.size(0), -1))
        max_out = self.mlp(self.max_pool(x).view(x.size(0), -1))
        attention = self.sigmoid(
            avg_out + max_out).view(x.size(0), x.size(1), 1, 1)
        return x * attention


class SpatialAttention(nn.Module):
    """Spatial Attention Module"""

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size,
                              padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention_input = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(attention_input))
        return x * attention


class FeatureSplitBranch(nn.Module):
    """
    Feature Split Branch for separating global, target, and background features

    Args:
        in_channels: Number of input channels
        raw_channels: Channels for global feature
        target_channels: Channels for target feature
        background_channels: Channels for background feature
        use_attention: Whether to use channel attention
    """

    def __init__(
        self,
        in_channels: int = 512,
        raw_channels: int = 256,
        target_channels: int = 256,
        background_channels: int = 256,
        use_attention: bool = True,
    ) -> None:
        super().__init__()

        self.global_proj = nn.Sequential(
            nn.Conv2d(in_channels, raw_channels, kernel_size=1),
            nn.BatchNorm2d(raw_channels),
            nn.ReLU(inplace=True),
        )

        self.target_proj = nn.Sequential(
            nn.Conv2d(in_channels, target_channels, kernel_size=1),
            nn.BatchNorm2d(target_channels),
            nn.ReLU(inplace=True),
        )

        self.background_proj = nn.Sequential(
            nn.Conv2d(in_channels, background_channels, kernel_size=1),
            nn.BatchNorm2d(background_channels),
            nn.ReLU(inplace=True),
        )

        self.target_attention = ChannelAttention(
            target_channels) if use_attention else nn.Identity()
        self.background_attention = ChannelAttention(
            background_channels) if use_attention else nn.Identity()

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        global_feat = self.global_proj(x)
        target_feat = self.target_proj(x)
        target_feat = self.target_attention(target_feat)
        background_feat = self.background_proj(x)
        background_feat = self.background_attention(background_feat)

        return global_feat, target_feat, background_feat



class UNetBackboneSmallTarget(nn.Module):
    """
    U-Net Backbone optimized for small target detection

    Features:
    1. Encoder with 3 levels (E0, E1, E2) for P0/P1/P2 detection
    2. Skip connections preserved for high-resolution features
    3. Decoder with feature refinement

    Args:
        in_channels: Number of input channels
        base_channels: Base number of channels
        depth: Number of encoder levels (default: 3 for P0/P1/P2)
        norm_type: Normalization type
        act_type: Activation type
        use_residual: Whether to use residual connections
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        depth: int = 3,
        norm_type: str = "bn",
        act_type: str = "relu",
        use_residual: bool = True,
    ) -> None:
        super().__init__()

        self.depth = depth
        self.enc_channels = [base_channels * (2**i) for i in range(depth)]

        self.encoder_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()

        self.encoder_blocks.append(
            DoubleConv(
                in_channels, self.enc_channels[0], norm_type, act_type, use_residual)
        )

        for i in range(1, depth):
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            self.encoder_blocks.append(
                DoubleConv(
                    self.enc_channels[i - 1],
                    self.enc_channels[i],
                    norm_type,
                    act_type,
                    use_residual,
                )
            )

        self.decoder_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        for i in range(depth - 1, 0, -1):
            self.upsamples.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="bilinear",
                                align_corners=False),
                    nn.Conv2d(
                        self.enc_channels[i],
                        self.enc_channels[i - 1],
                        kernel_size=1,
                        bias=False,
                    ),
                    get_norm_layer(norm_type, self.enc_channels[i - 1]),
                    get_activation(act_type),
                )
            )

            self.decoder_blocks.append(
                DoubleConv(
                    self.enc_channels[i - 1] * 2,
                    self.enc_channels[i - 1],
                    norm_type,
                    act_type,
                    use_residual,
                )
            )

        self.output_proj = nn.Sequential(
            nn.Conv2d(
                self.enc_channels[0],
                self.enc_channels[-1],
                kernel_size=1,
                bias=False,
            ),
            get_norm_layer(norm_type, self.enc_channels[-1]),
            get_activation(act_type),
        )

        self.tiny_attention = TinyTargetAttention(self.enc_channels[-1])

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        encoder_features = []

        feat = self.encoder_blocks[0](x)
        encoder_features.append(feat)

        for i in range(1, self.depth):
            feat = self.pools[i - 1](feat)
            feat = self.encoder_blocks[i](feat)
            encoder_features.append(feat)

        bottleneck = encoder_features[-1]

        decoder_features = []
        feat = bottleneck

        for i in range(len(self.upsamples)):
            feat = self.upsamples[i](feat)
            skip_idx = self.depth - 2 - i
            skip = encoder_features[skip_idx]
            feat = torch.cat([feat, skip], dim=1)
            feat = self.decoder_blocks[i](feat)
            decoder_features.append(feat)

        output_features = self.output_proj(feat)
        output_features = self.tiny_attention(output_features)

        return output_features
        # return {
        #     "features": output_features,
        #     "encoder_features": encoder_features,
        #     "bottleneck": bottleneck,
        #     "decoder_features": decoder_features,
        # }


class MultiScaleFPN(nn.Module):
    """
    Multi-Scale Feature Pyramid Network for P0/P1/P2 detection

    Generates features at three scales:
    - P0: Full resolution (H × W)
    - P1: 1/2 resolution (H/2 × W/2)
    - P2: 1/4 resolution (H/4 × W/4)

    Args:
        in_channels: Number of input channels from backbone
        out_channels: Number of output channels for each scale
        num_scales: Number of scales (default: 3 for P0/P1/P2)
        norm_type: Normalization type
        act_type: Activation type
    """

    def __init__(
        self,
        in_channels: int = 256,
        out_channels: int = 256,
        num_scales: int = 3,
        norm_type: str = "bn",
        act_type: str = "relu",
    ) -> None:
        super().__init__()

        self.num_scales = num_scales
        self.out_channels = out_channels

        self.p0_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            get_norm_layer(norm_type, out_channels),
            get_activation(act_type),
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            get_norm_layer(norm_type, out_channels),
            get_activation(act_type),
        )

        self.p1_downsample = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3,
                      stride=2, padding=1, bias=False),
            get_norm_layer(norm_type, out_channels),
            get_activation(act_type),
        )
        self.p1_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            get_norm_layer(norm_type, out_channels),
            get_activation(act_type),
        )

        self.p2_downsample = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3,
                      stride=2, padding=1, bias=False),
            get_norm_layer(norm_type, out_channels),
            get_activation(act_type),
        )
        self.p2_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            get_norm_layer(norm_type, out_channels),
            get_activation(act_type),
        )

        self.p2_upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            get_norm_layer(norm_type, out_channels),
            get_activation(act_type),
        )
        self.p1_fusion = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels,
                      kernel_size=1, bias=False),
            get_norm_layer(norm_type, out_channels),
            get_activation(act_type),
        )

        self.p1_upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            get_norm_layer(norm_type, out_channels),
            get_activation(act_type),
        )
        self.p0_fusion = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels,
                      kernel_size=1, bias=False),
            get_norm_layer(norm_type, out_channels),
            get_activation(act_type),
        )

        self.p0_refine = FeatureRefinementModule(out_channels)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        p0 = self.p0_conv(x)

        p1 = self.p1_downsample(p0)
        p1 = self.p1_conv(p1)

        p2 = self.p2_downsample(p1)
        p2 = self.p2_conv(p2)

        p2_up = self.p2_upsample(p2)
        if p2_up.shape[2:] != p1.shape[2:]:
            p2_up = F.interpolate(
                p2_up, size=p1.shape[2:], mode="bilinear", align_corners=False)
        p1 = self.p1_fusion(torch.cat([p1, p2_up], dim=1))

        p1_up = self.p1_upsample(p1)
        if p1_up.shape[2:] != p0.shape[2:]:
            p1_up = F.interpolate(
                p1_up, size=p0.shape[2:], mode="bilinear", align_corners=False)
        p0 = self.p0_fusion(torch.cat([p0, p1_up], dim=1))

        p0 = self.p0_refine(p0)

        return {
            "P0": p0,
        }


class CoordAtt(nn.Module):
    """
    Coordinate Attention FPN - Lightweight FPN with CoordAtt

    Uses Coordinate Attention for feature enhancement instead of 
    traditional multi-scale fusion. Significantly reduces parameters.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        reduction: Reduction ratio for CoordAtt
        norm_type: Normalization type
        act_type: Activation type
    """

    def __init__(
        self,
        in_channels: int = 256,
        out_channels: int = 256,
        reduction: int = 16,
        norm_type: str = "bn",
        act_type: str = "relu",
    ) -> None:
        super().__init__()

        self.out_channels = out_channels

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            get_norm_layer(norm_type, out_channels),
            get_activation(act_type),
        )

        self.coord_att = CoordinateAttention(
            out_channels, out_channels, reduction)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        x = self.conv(x)
        x = self.coord_att(x)
        return {"P0": x}


class SingleScaleDetectionHead(nn.Module):
    """
    Single-Scale Detection Head for P0 only

    Detection head with:
    - Box regression branch
    - Objectness branch
    - Classification branch

    Args:
        in_channels: Number of input channels
        num_classes: Number of detection classes
    """

    def __init__(
        self,
        in_channels: int = 256,
        num_classes: int = 1,
    ) -> None:
        super().__init__()

        self.num_classes = num_classes

        self.head = SmallTargetHead(in_channels, num_classes)

    def forward(self, features: Dict[str, Tensor]) -> Dict[str, Tensor]:
        p0 = features["P0"]
        head_out = self.head(p0)

        return {
            "P0_reg": head_out["reg"],
            "P0_obj": head_out["obj"],
            "P0_cls": head_out["cls"],
        }


class MemISTDSmallTarget(nn.Module):
    """
    MemISTD Small Target Detection Network

    Optimized for infrared small target detection with:
    1. Multi-scale detection at P0/P1/P2 (full resolution, 1/2, 1/4)
    2. Memory-augmented target/background separation
    3. TinyTargetAttention for small target enhancement
    4. Feature refinement modules
    5. New DualMemorySystem with multi-scale memory branches

    Args:
        in_channels: Number of input channels (default: 1 for grayscale)
        num_classes: Number of detection classes
        base_channels: Base number of channels
        backbone_depth: Depth of backbone (default: 3 for P0/P1/P2)
        target_memory_slots: Number of target memory slots
        background_memory_slots: Number of background memory slots
        use_attention: Whether to use attention in feature split
        norm_type: Normalization type
        act_type: Activation type
        use_residual: Whether to use residual connections
        use_dual_memory: Whether to use new DualMemorySystem
        ms_fusion_type: Multi-scale fusion type ('cat' or 'attention')
        global_fusion_strategy: Arithmetic fusion strategy
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 1,
        base_channels: int = 64,
        backbone_depth: int = 3,
        target_memory_slots: int = 16,
        background_memory_slots: int = 64,
        use_attention: bool = True,
        norm_type: str = "bn",
        act_type: str = "relu",
        use_residual: bool = True,
        save_heatmaps: bool = False,
        ms_fusion_type: str = 'attention',
        global_fusion_strategy: str = 'spatial_reweight',
        bg_patch_sizes: List[int] = [3, 5, 7],
        tg_patch_sizes: List[int] = [3, 5, 7],
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.save_heatmaps = save_heatmaps

        self.backbone = UNetBackboneSmallTarget(
            in_channels=in_channels,
            base_channels=base_channels,
            depth=backbone_depth,
            norm_type=norm_type,
            act_type=act_type,
            use_residual=use_residual,
        )

        backbone_channels = base_channels * (2 ** (backbone_depth - 1))
        mem_channels = backbone_channels // 2

        self.feature_split = FeatureSplitBranch(
            in_channels=backbone_channels,
            raw_channels=mem_channels,
            target_channels=mem_channels,
            background_channels=mem_channels,
            use_attention=use_attention,
        )

        self.dual_memory = DualMemorySystem(
            in_channels=mem_channels,
            num_bg_memories=background_memory_slots,
            num_tg_memories=target_memory_slots,
            bg_patch_sizes=bg_patch_sizes,
            tg_patch_sizes=tg_patch_sizes,
            ms_fusion_type=ms_fusion_type,
        )

        self.final_arithmetic = ArithmeticFusion(
            mem_channels, strategy=global_fusion_strategy)

        self.neck = CoordAtt(
            in_channels=mem_channels,
            out_channels=mem_channels,
            reduction=16,
            norm_type=norm_type,
            act_type=act_type,
        )

        self.detection_head = SingleScaleDetectionHead(
            in_channels=mem_channels,
            num_classes=num_classes,
        )

        self.target_decoder = MemoryDecoder(
            in_channels=mem_channels,
            out_channels=in_channels,
        )
        self.background_decoder = MemoryDecoder(
            in_channels=mem_channels,
            out_channels=in_channels,
        )

    def forward(self, x: Tensor, decode_image: bool = False) -> Dict[str, object]:
        """
        Forward pass

        Args:
            x: Input image tensor [B, C, H, W]
            decode_image: If True, decode memory features to image space (only for training).

        Returns:
            Dictionary containing:
            - predictions: Detection predictions at each scale
            - multi_scale_features: Multi-scale features (P0, P1, P2)
            - Memory module outputs
            - Decoded images (if decode_image=True)
        """
        backbone_feat = self.backbone(x)

        global_feat, target_feat, background_feat = self.feature_split(
            backbone_feat)

        background_feat_memory, target_feat_memory = self.dual_memory(
            target_feat, background_feat)

        target_recon_img, background_recon_img = None, None
        if decode_image:
            target_recon_img = self.target_decoder(target_feat_memory)
            background_recon_img = self.background_decoder(
                background_feat_memory)

        fused_feat = self.final_arithmetic(
            global_feat, background_feat_memory, target_feat_memory)

        if self.save_heatmaps:
            # 保存原图（逆预处理变换）
            # 预处理: image = (image / 255.0 + 0.1246) / 1.0923
            # 逆变换: image = (x * 1.0923 - 0.1246) * 255
            # original_img = x * 1.0923 - 0.1246  # IRDST 逆变换
            original_img = x * 0.2457 + 0.4874  # ITSDT 
            original_img = original_img * 255.0
            original_img = torch.clamp(original_img, 0, 255)
            self.save_grayscale_image(original_img, "hot_map/original_img", "original.png")
            
            self.save_feature_heatmaps(global_feat, "hot_map/global_feat", "global")
            self.save_feature_heatmaps(target_recon_img, "hot_map/target_recon_img", "target")
            self.save_feature_heatmaps(background_recon_img, "hot_map/background_recon_img", "background")
            self.save_feature_heatmaps(fused_feat, "hot_map/fused_feat", "fused_feat")

        multi_scale_features = self.neck(fused_feat)

        predictions = self.detection_head(multi_scale_features)

        outputs = {
            "predictions": predictions,
            "target_feat_recon": target_feat_memory,
            "background_feat_recon": background_feat_memory,
            "target_recon_img": target_recon_img,
            "background_recon_img": background_recon_img,
            "original_img": x,
        }

        return outputs

    def compute_loss(
        self,
        outputs: Dict[str, object],
        labels: List[Tensor],
        yolox_loss: nn.Module,
        residual_recon_loss: Optional[nn.Module] = None,
        residual_recon_weight: float = 1.0,
    ) -> Dict[str, Tensor]:
        """
        Compute all losses inside the model

        Args:
            outputs: Output from forward()
            labels: Ground truth labels (list of tensors)
            yolox_loss: YOLOLoss instance
            residual_recon_loss: ResidualReconstructionLoss instance (optional)
            residual_recon_weight: Weight for residual reconstruction loss

        Returns:
            Dictionary containing all loss components
        """
        predictions = outputs["predictions"]
        pred_list = self.get_predictions_list(predictions)

        loss_dict = yolox_loss(pred_list, labels, return_components=True)
        loss_yolo = loss_dict["total_loss"]

        total_loss = loss_yolo

        loss_residual_recon = torch.tensor(0.0, device=loss_yolo.device)
        loss_global = torch.tensor(0.0, device=loss_yolo.device)
        loss_tgt_sparse = torch.tensor(0.0, device=loss_yolo.device)
        loss_tgt_content = torch.tensor(0.0, device=loss_yolo.device)
        loss_bg_inpaint = torch.tensor(0.0, device=loss_yolo.device)

        if residual_recon_loss is not None and outputs.get("target_recon_img") is not None:
            target_mask = outputs.get("target_mask")
            if target_mask is not None:
                residual_losses = residual_recon_loss(
                    img=outputs["original_img"],
                    rec_bg=outputs["background_recon_img"],
                    rec_tgt=outputs["target_recon_img"],
                    mask=target_mask,
                )
                loss_residual_recon = residual_losses["loss_residual_recon"] * \
                    residual_recon_weight
                loss_global = residual_losses["loss_global"]
                loss_tgt_sparse = residual_losses["loss_tgt_sparse"]
                loss_tgt_content = residual_losses["loss_tgt_content"]
                loss_bg_inpaint = residual_losses["loss_bg_inpaint"]

                total_loss = total_loss + loss_residual_recon

        loss_dict_result = {
            "total_loss": total_loss,
            "yolo_loss": loss_yolo,
            "loss_box": loss_dict["loss_box"],
            "loss_obj": loss_dict["loss_obj"],
            "loss_cls": loss_dict["loss_cls"],
            "residual_recon_loss": loss_residual_recon,
            "loss_global": loss_global,
            "loss_tgt_sparse": loss_tgt_sparse,
            "loss_tgt_content": loss_tgt_content,
            "loss_bg_inpaint": loss_bg_inpaint,
            "num_fg": loss_dict["num_fg"],
        }

        return loss_dict_result

    @torch.no_grad()
    def detect(
        self,
        x: Tensor,
        conf_thres: float = 0.05,
        nms_thres: float = 0.5,
        max_detections: int = 300,
        debug: bool = False,
    ) -> Tensor:
        """
        Inference detection function

        Directly outputs detection results after NMS.

        Args:
            x: Input image tensor [B, C, H, W]
            conf_thres: Confidence threshold for filtering detections
            nms_thres: NMS IoU threshold
            max_detections: Maximum number of detections per image
            debug: If True, print debug information

        Returns:
            detections: [B, N, 6] detection results
                - Each detection: [x1, y1, x2, y2, score, class_id]
                - N is the number of detections (padded to max_detections if fewer)
                - If no detections, returns zeros tensor
        """
        from torchvision.ops import batched_nms

        self.eval()

        B = x.shape[0]
        device = x.device

        outputs = self.forward(x, decode_image=False)
        predictions = outputs["predictions"]

        # 单尺度检测，只有 P0，stride = 1
        stride = 1
        all_detections = [[] for _ in range(B)]

        reg = predictions["P0_reg"]
        obj = predictions["P0_obj"]
        cls = predictions["P0_cls"]

        _, _, H, W = reg.shape

        yv, xv = torch.meshgrid(
            [torch.arange(H), torch.arange(W)], indexing="ij")
        grid = torch.stack((xv, yv), 2).view(1, H, W, 2).type_as(reg)
        grid = grid + 0.5
        grid = grid.expand(B, -1, -1, -1)

        if debug:
            print(f"[DEBUG] 推理时anchor中心示例 (P0, 前5个):")
            print(f"  grid[0,0,0,:] = {grid[0, 0, 0, :].tolist()}")
            print(f"  grid[0,0,1,:] = {grid[0, 0, 1, :].tolist()}")
            print(f"  注意: anchor中心 = (grid_index + 0.5 + pred_offset) * stride")

        reg = reg.permute(0, 2, 3, 1).contiguous()
        obj = obj.permute(0, 2, 3, 1).contiguous()
        if cls is not None:
            cls = cls.permute(0, 2, 3, 1).contiguous()

        cx = (reg[..., 0] + grid[..., 0]) * stride
        cy = (reg[..., 1] + grid[..., 1]) * stride
        w = torch.exp(reg[..., 2].clamp(max=10)) * stride
        h = torch.exp(reg[..., 3].clamp(max=10)) * stride

        w = torch.clamp(w, min=2.0, max=1000.0)
        h = torch.clamp(h, min=2.0, max=1000.0)

        obj_conf = torch.sigmoid(obj[..., 0])

        if debug:
            print(
                f"[DEBUG] P0 cls.shape: {cls.shape if cls is not None else None}")

        if cls is not None:
            cls_conf = torch.sigmoid(cls[..., 0])
            # 对于单类别检测，只使用 obj_conf
            if cls.shape[-1] == 1:
                scores = obj_conf
                if debug:
                    print(f"[DEBUG] P0 单类别检测，scores = obj_conf")
            else:
                scores = obj_conf * cls_conf
                if debug:
                    print(f"[DEBUG] P0 多类别检测，scores = obj_conf * cls_conf")
        else:
            scores = obj_conf

        if debug:
            print(f"[DEBUG] P0 坐标解码:")
            print(
                f"  cx: min={cx.min().item():.1f}, max={cx.max().item():.1f}")
            print(
                f"  cy: min={cy.min().item():.1f}, max={cy.max().item():.1f}")
            print(f"  w: min={w.min().item():.1f}, max={w.max().item():.1f}")
            print(f"  h: min={h.min().item():.1f}, max={h.max().item():.1f}")
            print(
                f"  obj_conf: min={obj_conf.min().item():.6f}, max={obj_conf.max().item():.6f}")
            print(
                f"  scores: min={scores.min().item():.6f}, max={scores.max().item():.6f}")
            print(f"  conf_thres: {conf_thres}")
            above_thres = (scores > conf_thres).sum().item()
            print(f"  scores > conf_thres 的数量: {above_thres}")

        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        x1, x2 = torch.minimum(x1, x2), torch.maximum(x1, x2)
        y1, y2 = torch.minimum(y1, y2), torch.maximum(y1, y2)

        valid_coord_mask = (
            torch.isfinite(x1) & torch.isfinite(y1) &
            torch.isfinite(x2) & torch.isfinite(y2) &
            (x1 >= 0) & (y1 >= 0) & (x2 > x1) & (y2 > y1) &
            (x1 < 10000) & (y1 < 10000) & (x2 < 10000) & (y2 < 10000)
        )

        x1 = torch.clamp(x1, min=0, max=10000)
        y1 = torch.clamp(y1, min=0, max=10000)
        x2 = torch.clamp(x2, min=0, max=10001)
        y2 = torch.clamp(y2, min=0, max=10001)
        x2 = torch.maximum(x2, x1 + 1)
        y2 = torch.maximum(y2, y1 + 1)

        for b in range(B):
            img_scores = scores[b].view(-1)
            img_valid_mask = valid_coord_mask[b].view(-1)
            mask = (img_scores > conf_thres) & img_valid_mask

            if mask.sum() == 0:
                continue

            img_x1 = x1[b].view(-1)[mask]
            img_y1 = y1[b].view(-1)[mask]
            img_x2 = x2[b].view(-1)[mask]
            img_y2 = y2[b].view(-1)[mask]
            img_scores_filtered = img_scores[mask]

            boxes = torch.stack([img_x1, img_y1, img_x2, img_y2], dim=1)

            all_detections[b].append((boxes, img_scores_filtered))

        final_detections = []
        for b in range(B):
            if len(all_detections[b]) == 0:
                final_detections.append(torch.zeros(
                    max_detections, 6, device=device))
                continue

            boxes = torch.cat([d[0] for d in all_detections[b]], dim=0)
            scores = torch.cat([d[1] for d in all_detections[b]], dim=0)

            boxes = boxes.float()
            boxes = torch.clamp(boxes, min=0.0, max=10000.0)
            scores = torch.clamp(scores, min=0.0, max=1.0)

            valid_boxes_mask = (
                (boxes[:, 2] - boxes[:, 0] >= 1.0) &
                (boxes[:, 3] - boxes[:, 1] >= 1.0) &
                torch.isfinite(boxes).all(dim=1) &
                torch.isfinite(scores)
            )

            if valid_boxes_mask.sum() == 0:
                final_detections.append(torch.zeros(
                    max_detections, 6, device=device))
                continue

            boxes = boxes[valid_boxes_mask]
            scores = scores[valid_boxes_mask]

            labels = torch.zeros(
                scores.shape[0], dtype=torch.long, device=device)

            try:
                keep = batched_nms(boxes, scores, labels, nms_thres)
            except RuntimeError:
                final_detections.append(torch.zeros(
                    max_detections, 6, device=device))
                continue

            if len(keep) > max_detections:
                keep = keep[:max_detections]

            kept_boxes = boxes[keep]
            kept_scores = scores[keep]
            kept_labels = labels[keep]

            n_detections = kept_boxes.shape[0]
            detections = torch.zeros(max_detections, 6, device=device)
            detections[:n_detections, :4] = kept_boxes
            detections[:n_detections, 4] = kept_scores
            detections[:n_detections, 5] = kept_labels.float()

            final_detections.append(detections)

        result = torch.stack(final_detections, dim=0)

        return result

    @staticmethod
    def save_feature_heatmaps(feature: Tensor, save_dir: str, prefix: str = "channel"):
        """
        保存特征图为热力图

        Args:
            feature: 特征张量 [B, C, H, W]
            save_dir: 保存目录
            prefix: 文件名前缀
        """
        import os
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        os.makedirs(save_dir, exist_ok=True)

        feat_np = feature[0].detach().cpu().numpy()

        for c in range(feat_np.shape[0]):
            fig, ax = plt.subplots(figsize=(8, 8))
            im = ax.imshow(feat_np[c], cmap='jet')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.axis('off')
            plt.savefig(os.path.join(
                save_dir, f'{prefix}_channel_{c:03d}.png'),
                bbox_inches='tight', pad_inches=0)
            plt.close(fig)

        print(f"Saved {feat_np.shape[0]} heatmaps to {save_dir}")

    def save_grayscale_image(self, image: Tensor, save_dir: str, filename: str = "image.png"):
        """
        保存单通道灰度图

        Args:
            image: 单通道图像张量 [B, 1, H, W] 或 [1, H, W]，像素值范围 [0, 255]
            save_dir: 保存目录
            filename: 文件名
        """
        import os
        import numpy as np
        from PIL import Image

        os.makedirs(save_dir, exist_ok=True)

        img_tensor = image[0] if image.dim() == 4 else image
        img_np = img_tensor.squeeze().detach().cpu().numpy()
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)

        img = Image.fromarray(img_np, mode='L')
        img.save(os.path.join(save_dir, filename))
        print(f"Saved grayscale image to {os.path.join(save_dir, filename)}")

    def get_predictions_list(self, predictions: Dict[str, Tensor]) -> List[Tensor]:
        """
        Convert predictions dict to list format for loss computation

        Args:
            predictions: Dictionary of predictions from detection head

        Returns:
            List of tensors for each scale [P0_pred]
        """
        pred_list = []
        for scale in ["P0"]:
            reg = predictions[f"{scale}_reg"]
            obj = predictions[f"{scale}_obj"]
            cls = predictions[f"{scale}_cls"]
            if cls is not None:
                scale_pred = torch.cat([reg, obj, cls], dim=1)
            else:
                scale_pred = torch.cat([reg, obj], dim=1)
            pred_list.append(scale_pred)
        return pred_list

    @torch.no_grad()
    def detect_with_info(
        self,
        x: Tensor,
        conf_thres: float = 0.05,
        nms_thres: float = 0.5,
        max_detections: int = 300,
    ) -> Dict[str, Tensor]:
        """
        Inference detection function with additional information

        Args:
            x: Input image tensor [B, C, H, W]
            conf_thres: Confidence threshold
            nms_thres: NMS IoU threshold
            max_detections: Maximum detections per image

        Returns:
            Dictionary containing:
            - detections: [B, N, 6] detection results
            - num_detections: [B] number of valid detections per image
        """
        detections = self.detect(x, conf_thres, nms_thres, max_detections)

        num_detections = (detections[:, :, 4] > 0).sum(dim=1)

        return {
            "detections": detections,
            "num_detections": num_detections,
        }


def decode_predictions(
    predictions: List[Tensor],
    strides: List[int] = [1, 2, 4],
    conf_thres: float = 0.05,
    nms_thres: float = 0.5,
) -> List[Optional[Tensor]]:
    """
    Decode predictions from multi-scale outputs

    Args:
        predictions: List of predictions [P0, P1, P2]
        strides: Strides for each scale
        conf_thres: Confidence threshold
        nms_thres: NMS threshold

    Returns:
        List of detections per image
    """
    from torchvision.ops import batched_nms

    batch_size = predictions[0].shape[0]
    all_detections = [[] for _ in range(batch_size)]

    for pred, stride in zip(predictions, strides):
        B, C, H, W = pred.shape

        yv, xv = torch.meshgrid(
            [torch.arange(H), torch.arange(W)], indexing="ij")
        grid = torch.stack((xv, yv), 2).view(1, H, W, 2).type_as(pred)
        grid = grid + 0.5

        pred = pred.permute(0, 2, 3, 1).contiguous()

        pred[..., :2] = (pred[..., :2] + grid) * stride
        pred[..., 2:4] = torch.exp(pred[..., 2:4]) * stride
        pred[..., 2:4] = torch.clamp(pred[..., 2:4], min=2.0)

        pred = pred.view(B, -1, C // 6 * 6)

        for i in range(batch_size):
            img_pred = pred[i]

            obj_conf = img_pred[:, 4]
            conf_mask = obj_conf > conf_thres

            if conf_mask.sum() == 0:
                continue

            img_pred = img_pred[conf_mask]

            boxes = img_pred[:, :4]
            scores = img_pred[:, 4]

            x1 = boxes[:, 0] - boxes[:, 2] / 2
            y1 = boxes[:, 1] - boxes[:, 3] / 2
            x2 = boxes[:, 0] + boxes[:, 2] / 2
            y2 = boxes[:, 1] + boxes[:, 3] / 2
            boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=1)

            all_detections[i].append((boxes_xyxy, scores))

    final_detections = []
    for i in range(batch_size):
        if len(all_detections[i]) == 0:
            final_detections.append(None)
            continue

        boxes = torch.cat([d[0] for d in all_detections[i]], dim=0)
        scores = torch.cat([d[1] for d in all_detections[i]], dim=0)

        keep = batched_nms(
            boxes, scores, torch.zeros_like(scores).int(), nms_thres)

        final_detections.append((boxes[keep], scores[keep]))

    return final_detections


if __name__ == "__main__":
    # 当直接运行此文件时，添加父目录到路径
    import sys
    import os
    sys.path.insert(0, os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))

    print("=" * 80)
    print("MemISTD Small Target Detection Network - Test")
    print("=" * 80)

    model = MemISTDSmallTarget(
        in_channels=1,
        num_classes=1,
        base_channels=64,
        backbone_depth=3,
        target_memory_slots=20,
        background_memory_slots=200,
        use_attention=True,
    )

    x = torch.randn(2, 1, 480, 720)
    print(f"\nInput: {x.shape}")

    out = model(x)

    print(f"\n[OK] Multi-scale features:")
    for k, v in out["multi_scale_features"].items():
        print(f"  {k}: {v.shape}")

    print(f"\n[OK] Predictions:")
    for k, v in out["predictions"].items():
        if v is not None:
            print(f"  {k}: {v.shape}")

    print(f"\n[OK] Backbone features: {out['backbone_features'].shape}")
    print(f"[OK] Target feat raw: {out['target_feat_raw'].shape}")
    print(f"[OK] Background feat raw: {out['background_feat_raw'].shape}")

    pred_list = model.get_predictions_list(out["predictions"])
    print(f"\n[OK] Predictions list: {[p.shape for p in pred_list]}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)
    print(f"\n[STATS] Total parameters: {total_params:,}")
    print(f"[STATS] Trainable parameters: {trainable_params:,}")

    print("\n" + "=" * 80)
    print("All tests passed!")
    print("=" * 80)
