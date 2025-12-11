"""
General utilities:
- directory helpers
- metrics (IoU)
- visualization helpers (voxel -> slices / marching cubes -> .ply)
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
import trimesh

def ensure_dirs(cfg):
    os.makedirs(cfg.results_dir, exist_ok=True)
    os.makedirs(cfg.model_dir, exist_ok=True)
    os.makedirs(cfg.vis_dir, exist_ok=True)
    os.makedirs(cfg.loss_dir, exist_ok=True)

def voxel_iou(pred, target, threshold=0.5):
    """
    pred, target : numpy arrays with binary occupancy or probabilities
    """
    pred_bin = (pred >= threshold).astype(np.uint8)
    target_bin = (target >= 0.5).astype(np.uint8)
    inter = np.logical_and(pred_bin, target_bin).sum()
    union = np.logical_or(pred_bin, target_bin).sum()
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return inter / union

def save_voxel_slices(voxel, out_path, slices=6):
    """
    Save a grid of XY slices for quick visualization.
    voxel: (D,H,W) or (1,D,H,W)
    """
    if voxel.ndim == 4:
        voxel = voxel[0]
    D = voxel.shape[0]
    step = max(1, D // slices)
    fig, axs = plt.subplots(1, slices, figsize=(slices*2,2))
    for i in range(slices):
        idx = min(D-1, i*step)
        axs[i].imshow(voxel[idx], cmap='gray')
        axs[i].axis('off')
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close(fig)

def voxel_to_mesh_and_save(voxel, out_path, level=0.5):
    """
    Convert voxel grid to triangular mesh using marching cubes and save as .ply
    voxel: (D,H,W) numpy
    """
    # marching cubes expects voxel values; transpose to ensure axes orientation OK
    verts, faces, normals, _ = measure.marching_cubes(voxel.astype(np.float32), level=level)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    mesh.export(out_path)
