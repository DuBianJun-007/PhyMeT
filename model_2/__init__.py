"""
MemISTD Small Target Detection Model Package
=============================================

Optimized for infrared small target detection (target size < 7x7 pixels)

Key Improvements:
1. Multi-scale detection at P0/P1/P2 (full resolution, 1/2, 1/4)
2. TinyTargetAttention module for small target enhancement
3. NWD Loss + Focal Loss for small target optimization
4. Enhanced SimOTA with larger center_radius
"""

from .memistd_small_target import MemISTDSmallTarget
from .losses import (
    FocalLoss, 
    NWDLoss, 
    CombinedDetectionLoss,
    ResidualReconstructionLoss,
)
from .tiny_target_modules import (
    TinyTargetAttention,
    LocalContrastModule,
    SpatialAttentionEnhanced,
)
from .yolox_loss_optimized import YOLOLossOptimized

__all__ = [
    "MemISTDSmallTarget",
    "FocalLoss",
    "NWDLoss",
    "CombinedDetectionLoss",
    "ResidualReconstructionLoss",
    "TinyTargetAttention",
    "LocalContrastModule",
    "SpatialAttentionEnhanced",
    "YOLOLossOptimized",
]

__version__ = "1.0.0"
