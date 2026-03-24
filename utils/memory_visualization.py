"""
Memory Module Visualization Tool
=================================

Visualize memory module reconstruction effects:
1. Feature map visualization (raw vs reconstructed)
2. Memory attention weights visualization
3. Target/Background separation visualization
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict, List, Optional, Tuple


def visualize_feature_maps(
    outputs: Dict[str, torch.Tensor],
    target_mask: Optional[torch.Tensor] = None,
    save_path: str = "./vis_features",
    sample_idx: int = 0,
):
    """
    Visualize raw and reconstructed feature maps from memory modules.
    
    Args:
        outputs: Model outputs containing:
            - target_feat_raw: Target feature before memory
            - target_feat_recon: Target feature after memory reconstruction
            - background_feat_raw: Background feature before memory
            - background_feat_recon: Background feature after memory reconstruction
        target_mask: Ground truth target mask [B, 1, H, W]
        save_path: Directory to save visualizations
        sample_idx: Which sample in the batch to visualize
    """
    os.makedirs(save_path, exist_ok=True)
    
    target_raw = outputs.get("target_feat_raw")
    target_recon = outputs.get("target_feat_recon")
    bg_raw = outputs.get("background_feat_raw")
    bg_recon = outputs.get("background_feat_recon")
    
    if target_raw is None:
        print("Warning: target_feat_raw not found in outputs")
        return
    
    def get_channel_stats(feat):
        if feat is None:
            return None, None, None
        feat_np = feat[sample_idx].detach().cpu().numpy()
        mean_map = feat_np.mean(axis=0)
        max_map = feat_np.max(axis=0)
        std_map = feat_np.std(axis=0)
        return mean_map, max_map, std_map
    
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    
    titles = [
        ["Target Raw (Mean)", "Target Raw (Max)", "Target Recon (Mean)", "Target Recon (Max)"],
        ["Target Diff (Mean)", "Target Diff (Max)", "Target Recon Error", ""],
        ["BG Raw (Mean)", "BG Raw (Max)", "BG Recon (Mean)", "BG Recon (Max)"],
        ["BG Diff (Mean)", "BG Diff (Max)", "BG Recon Error", "Target Mask"],
    ]
    
    target_raw_mean, target_raw_max, _ = get_channel_stats(target_raw)
    target_recon_mean, target_recon_max, _ = get_channel_stats(target_recon)
    bg_raw_mean, bg_raw_max, _ = get_channel_stats(bg_raw)
    bg_recon_mean, bg_recon_max, _ = get_channel_stats(bg_recon)
    
    if target_raw_mean is not None and target_recon_mean is not None:
        target_diff_mean = np.abs(target_raw_mean - target_recon_mean)
        target_diff_max = np.abs(target_raw_max - target_recon_max)
        target_error = (target_raw_mean - target_recon_mean) ** 2
    
    if bg_raw_mean is not None and bg_recon_mean is not None:
        bg_diff_mean = np.abs(bg_raw_mean - bg_recon_mean)
        bg_diff_max = np.abs(bg_raw_max - bg_recon_max)
        bg_error = (bg_raw_mean - bg_recon_mean) ** 2
    
    data = [
        [target_raw_mean, target_raw_max, target_recon_mean, target_recon_max],
        [target_diff_mean, target_diff_max, target_error, None],
        [bg_raw_mean, bg_raw_max, bg_recon_mean, bg_recon_max],
        [bg_diff_mean, bg_diff_max, bg_error, None],
    ]
    
    if target_mask is not None:
        mask_np = target_mask[sample_idx, 0].detach().cpu().numpy()
        data[3][3] = mask_np
    
    for i in range(4):
        for j in range(4):
            ax = axes[i, j]
            if data[i][j] is not None:
                im = ax.imshow(data[i][j], cmap='jet')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(titles[i][j], fontsize=10)
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"feature_maps_sample{sample_idx}.png"), dpi=150)
    plt.close()
    print(f"Feature maps saved to {save_path}/feature_maps_sample{sample_idx}.png")


def visualize_memory_attention(
    outputs: Dict[str, torch.Tensor],
    save_path: str = "./vis_features",
    sample_idx: int = 0,
    top_k: int = 5,
):
    """
    Visualize memory attention weights.
    
    Args:
        outputs: Model outputs containing:
            - target_similarity: Attention weights for target memory
            - background_similarity: Attention weights for background memory
        save_path: Directory to save visualizations
        sample_idx: Which sample in the batch to visualize
        top_k: Number of top memory slots to visualize
    """
    os.makedirs(save_path, exist_ok=True)
    
    target_sim = outputs.get("target_similarity")
    bg_sim = outputs.get("background_similarity")
    
    if target_sim is None and bg_sim is None:
        print("Warning: No similarity tensors found")
        return
    
    def process_similarity(sim, name):
        if sim is None:
            return None, None
        
        sim_np = sim.detach().cpu().numpy()
        
        if sim_np.ndim == 2:
            num_slots = sim_np.shape[1]
            mean_weights = sim_np.mean(axis=0)
            
            top_indices = np.argsort(mean_weights)[-top_k:][::-1]
            
            return mean_weights, top_indices
        return None, None
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    if target_sim is not None:
        target_weights, target_top_idx = process_similarity(target_sim, "target")
        if target_weights is not None:
            ax = axes[0, 0]
            ax.bar(range(len(target_weights)), target_weights, color='red', alpha=0.7)
            ax.set_xlabel('Memory Slot Index')
            ax.set_ylabel('Mean Attention Weight')
            ax.set_title(f'Target Memory Attention\nTop {top_k}: {target_top_idx.tolist()}')
            
            ax = axes[0, 1]
            ax.hist(target_weights, bins=50, color='red', alpha=0.7)
            ax.set_xlabel('Attention Weight')
            ax.set_ylabel('Frequency')
            ax.set_title('Target Memory Weight Distribution')
    
    if bg_sim is not None:
        bg_weights, bg_top_idx = process_similarity(bg_sim, "background")
        if bg_weights is not None:
            ax = axes[1, 0]
            ax.bar(range(len(bg_weights)), bg_weights, color='blue', alpha=0.7)
            ax.set_xlabel('Memory Slot Index')
            ax.set_ylabel('Mean Attention Weight')
            ax.set_title(f'Background Memory Attention\nTop {top_k}: {bg_top_idx.tolist()}')
            
            ax = axes[1, 1]
            ax.hist(bg_weights, bins=50, color='blue', alpha=0.7)
            ax.set_xlabel('Attention Weight')
            ax.set_ylabel('Frequency')
            ax.set_title('Background Memory Weight Distribution')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"memory_attention_sample{sample_idx}.png"), dpi=150)
    plt.close()
    print(f"Memory attention saved to {save_path}/memory_attention_sample{sample_idx}.png")


def visualize_memory_vectors(
    outputs: Dict[str, torch.Tensor],
    save_path: str = "./vis_features",
    method: str = "pca",
):
    """
    Visualize memory vectors using dimensionality reduction.
    
    Args:
        outputs: Model outputs containing:
            - target_memory_matrix: Target memory matrix [mem_dim, fea_dim]
            - background_memory_matrix: Background memory matrix [mem_dim, fea_dim]
        save_path: Directory to save visualizations
        method: Dimensionality reduction method ('pca' or 'tsne')
    """
    os.makedirs(save_path, exist_ok=True)
    
    M_t = outputs.get("target_memory_matrix")
    M_b = outputs.get("background_memory_matrix")
    
    if M_t is None and M_b is None:
        print("Warning: No memory matrices found")
        return
    
    M_t_np = M_t.detach().cpu().numpy() if M_t is not None else None
    M_b_np = M_b.detach().cpu().numpy() if M_b is not None else None
    
    if method == "pca":
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2)
    elif method == "tsne":
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=42)
    else:
        reducer = PCA(n_components=2)
    
    all_vectors = []
    labels = []
    colors = []
    
    if M_t_np is not None:
        all_vectors.append(M_t_np)
        labels.extend(['Target'] * len(M_t_np))
        colors.extend(['red'] * len(M_t_np))
    
    if M_b_np is not None:
        all_vectors.append(M_b_np)
        labels.extend(['Background'] * len(M_b_np))
        colors.extend(['blue'] * len(M_b_np))
    
    if len(all_vectors) == 0:
        return
    
    all_vectors = np.vstack(all_vectors)
    embedded = reducer.fit_transform(all_vectors)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for label, color in [('Target', 'red'), ('Background', 'blue')]:
        mask = np.array([l == label for l in labels])
        ax.scatter(embedded[mask, 0], embedded[mask, 1], 
                   c=color, label=label, alpha=0.6, s=50)
    
    ax.set_xlabel(f'{method.upper()} Component 1')
    ax.set_ylabel(f'{method.upper()} Component 2')
    ax.set_title('Memory Vector Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"memory_vectors_{method}.png"), dpi=150)
    plt.close()
    print(f"Memory vectors saved to {save_path}/memory_vectors_{method}.png")


def visualize_reconstruction_quality(
    outputs: Dict[str, torch.Tensor],
    target_mask: Optional[torch.Tensor] = None,
    save_path: str = "./vis_features",
    sample_idx: int = 0,
):
    """
    Visualize reconstruction quality with target mask overlay.
    
    Args:
        outputs: Model outputs
        target_mask: Ground truth target mask
        save_path: Directory to save visualizations
        sample_idx: Which sample to visualize
    """
    os.makedirs(save_path, exist_ok=True)
    
    target_raw = outputs.get("target_feat_raw")
    target_recon = outputs.get("target_feat_recon")
    bg_raw = outputs.get("background_feat_raw")
    bg_recon = outputs.get("background_feat_recon")
    
    if target_mask is None:
        print("Warning: target_mask is required for this visualization")
        return
    
    def compute_recon_error(raw, recon, mask, is_target=True):
        if raw is None or recon is None:
            return None, None, None
        
        raw_np = raw[sample_idx].mean(dim=0, keepdim=True).detach().cpu()
        recon_np = recon[sample_idx].mean(dim=0, keepdim=True).detach().cpu()
        
        mask_resized = F.interpolate(
            target_mask[sample_idx:sample_idx+1], 
            size=raw_np.shape[-2:], 
            mode='bilinear', 
            align_corners=False
        )
        mask_np = mask_resized[0, 0].numpy()
        
        error = ((raw_np - recon_np) ** 2)[0].numpy()
        
        if is_target:
            region_mask = mask_np > 0.5
        else:
            region_mask = mask_np <= 0.5
        
        region_error = error * region_mask
        mean_error = region_error[region_mask].mean() if region_mask.sum() > 0 else 0
        
        return error, region_error, mean_error
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    target_error, target_region_error, target_mean_error = compute_recon_error(
        target_raw, target_recon, target_mask, is_target=True
    )
    bg_error, bg_region_error, bg_mean_error = compute_recon_error(
        bg_raw, bg_recon, target_mask, is_target=False
    )
    
    mask_np = target_mask[sample_idx, 0].detach().cpu().numpy()
    
    if target_error is not None:
        im0 = axes[0, 0].imshow(target_error, cmap='hot')
        axes[0, 0].set_title(f'Target Recon Error\nMean: {target_mean_error:.4f}')
        plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)
        
        im1 = axes[0, 1].imshow(target_region_error, cmap='hot')
        axes[0, 1].set_title('Target Error (Target Region)')
        plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)
    
    if bg_error is not None:
        im2 = axes[0, 2].imshow(bg_error, cmap='hot')
        axes[0, 2].set_title(f'BG Recon Error\nMean: {bg_mean_error:.4f}')
        plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)
        
        im3 = axes[0, 3].imshow(bg_region_error, cmap='hot')
        axes[0, 3].set_title('BG Error (Background Region)')
        plt.colorbar(im3, ax=axes[0, 3], fraction=0.046)
    
    im4 = axes[1, 0].imshow(mask_np, cmap='gray')
    axes[1, 0].set_title('Target Mask (GT)')
    plt.colorbar(im4, ax=axes[1, 0], fraction=0.046)
    
    if target_error is not None and bg_error is not None:
        combined_error = target_error + bg_error
        im5 = axes[1, 1].imshow(combined_error, cmap='hot')
        axes[1, 1].set_title('Combined Error')
        plt.colorbar(im5, ax=axes[1, 1], fraction=0.046)
        
        if target_raw is not None and bg_raw is not None:
            target_raw_mean = target_raw[sample_idx].mean(dim=0).detach().cpu().numpy()
            bg_raw_mean = bg_raw[sample_idx].mean(dim=0).detach().cpu().numpy()
            
            im6 = axes[1, 2].imshow(target_raw_mean, cmap='jet')
            axes[1, 2].set_title('Target Raw Feature')
            plt.colorbar(im6, ax=axes[1, 2], fraction=0.046)
            
            im7 = axes[1, 3].imshow(bg_raw_mean, cmap='jet')
            axes[1, 3].set_title('Background Raw Feature')
            plt.colorbar(im7, ax=axes[1, 3], fraction=0.046)
    
    for ax in axes.flat:
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"reconstruction_quality_sample{sample_idx}.png"), dpi=150)
    plt.close()
    print(f"Reconstruction quality saved to {save_path}/reconstruction_quality_sample{sample_idx}.png")


def add_visualization_hook(model, save_dir: str = "./vis_features", every_n_steps: int = 100):
    """
    Add visualization hooks to model for training visualization.
    
    Args:
        model: The MemISTDSmallTarget model
        save_dir: Directory to save visualizations
        every_n_steps: Visualize every N steps
    """
    import weakref
    
    weak_model = weakref.ref(model)
    step_counter = [0]
    
    def visualization_hook(module, inputs, outputs):
        step_counter[0] += 1
        
        if step_counter[0] % every_n_steps != 0:
            return
        
        model_ref = weak_model()
        if model_ref is None:
            return
        
        if isinstance(outputs, dict):
            save_path = os.path.join(save_dir, f"step_{step_counter[0]}")
            os.makedirs(save_path, exist_ok=True)
            
            visualize_feature_maps(outputs, save_path=save_path)
            visualize_memory_attention(outputs, save_path=save_path)
    
    return visualization_hook


class MemoryVisualizer:
    """
    Integrated visualizer for memory module analysis.
    """
    
    def __init__(self, save_dir: str = "./vis_memory"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def visualize_batch(
        self,
        outputs: Dict[str, torch.Tensor],
        target_mask: Optional[torch.Tensor] = None,
        sample_idx: int = 0,
        prefix: str = "",
    ):
        """Run all visualizations on a batch."""
        save_path = os.path.join(self.save_dir, prefix) if prefix else self.save_dir
        
        print(f"\n{'='*60}")
        print("Memory Module Visualization")
        print(f"{'='*60}")
        
        visualize_feature_maps(outputs, target_mask, save_path, sample_idx)
        visualize_memory_attention(outputs, save_path, sample_idx)
        visualize_reconstruction_quality(outputs, target_mask, save_path, sample_idx)
        visualize_memory_vectors(outputs, save_path, method="pca")
        
        self._print_statistics(outputs, target_mask, sample_idx)
    
    def _print_statistics(self, outputs, target_mask, sample_idx):
        """Print memory module statistics."""
        print(f"\n{'='*60}")
        print("Memory Module Statistics")
        print(f"{'='*60}")
        
        target_raw = outputs.get("target_feat_raw")
        target_recon = outputs.get("target_feat_recon")
        bg_raw = outputs.get("background_feat_raw")
        bg_recon = outputs.get("background_feat_recon")
        
        if target_raw is not None and target_recon is not None:
            target_mse = ((target_raw - target_recon) ** 2).mean().item()
            print(f"Target Reconstruction MSE: {target_mse:.6f}")
        
        if bg_raw is not None and bg_recon is not None:
            bg_mse = ((bg_raw - bg_recon) ** 2).mean().item()
            print(f"Background Reconstruction MSE: {bg_mse:.6f}")
        
        M_t = outputs.get("target_memory_matrix")
        M_b = outputs.get("background_memory_matrix")
        
        if M_t is not None and M_b is not None:
            M_t_norm = F.normalize(M_t, p=2, dim=1)
            M_b_norm = F.normalize(M_b, p=2, dim=1)
            cosine_sim = torch.mm(M_t_norm, M_b_norm.t())
            orth_loss = (cosine_sim ** 2).sum().item()
            print(f"Orthogonality Loss: {orth_loss:.6f}")
            print(f"Mean Cross-Similarity: {cosine_sim.abs().mean().item():.6f}")
        
        target_sim = outputs.get("target_similarity")
        bg_sim = outputs.get("background_similarity")
        
        if target_sim is not None:
            print(f"Target Memory Active Slots: {(target_sim.max(dim=1)[0] > 0.3).sum().item()}")
        
        if bg_sim is not None:
            print(f"Background Memory Active Slots: {(bg_sim.max(dim=1)[0] > 0.0025).sum().item()}")


if __name__ == "__main__":
    print("Memory Visualization Tool")
    print("=" * 60)
    print("\nUsage:")
    print("  from utils.memory_visualization import MemoryVisualizer")
    print("  visualizer = MemoryVisualizer(save_dir='./vis_memory')")
    print("  visualizer.visualize_batch(outputs, target_mask)")
