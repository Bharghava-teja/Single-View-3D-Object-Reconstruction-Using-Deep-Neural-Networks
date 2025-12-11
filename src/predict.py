"""
Simple inference script:
- loads a trained model .pth
- runs on an input image and saves voxel visualization and mesh
"""

import os
import torch
import numpy as np
from .model import Reconstructor
from dataset.preprocess import load_and_preprocess_image
from .config import Config
from .utils import save_voxel_slices, voxel_to_mesh_and_save

def infer_single(image_path, model_path, out_dir=None):
    cfg = Config()
    device = cfg.device
    model = Reconstructor(latent_dim=cfg.latent_dim, voxel_size=cfg.voxel_size)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    img = load_and_preprocess_image(image_path, size=cfg.image_size)  # C,H,W numpy
    img_t = torch.from_numpy(img).unsqueeze(0).to(device)  # 1,C,H,W
    with torch.no_grad():
        pred = model(img_t)
    pred_np = pred.detach().cpu().numpy()[0,0]
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(model_path), "inference")
    os.makedirs(out_dir, exist_ok=True)
    slices_path = os.path.join(out_dir, "pred_slices.png")
    save_voxel_slices(pred_np, slices_path)
    try:
        mesh_path = os.path.join(out_dir, "pred_mesh.ply")
        voxel_to_mesh_and_save((pred_np>=0.5).astype(np.uint8), mesh_path)
    except Exception as e:
        print("Mesh export failed. Install scikit-image and trimesh to enable mesh export.", e)
    print("Saved outputs to:", out_dir)

if __name__ == "__main__":
    # quick demo usage
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="input image path")
    parser.add_argument("--model", required=True, help="trained model .pth")
    parser.add_argument("--out", default=None, help="output directory")
    args = parser.parse_args()
    infer_single(args.image, args.model, args.out)
