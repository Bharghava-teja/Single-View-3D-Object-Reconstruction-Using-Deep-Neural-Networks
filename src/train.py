"""
Training script:
- Loads dataset (if available) otherwise uses dummy data generator
- Trains the Reconstructor model with BCE loss on voxels
- Saves model checkpoint and example visualizations
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .config import Config
from .model import Reconstructor
from .dataset import ImageVoxelDataset
from .utils import ensure_dirs, save_voxel_slices, voxel_iou
from dataset.preprocess import load_and_preprocess_image


def train(cfg: Config):
    device = cfg.device
    print(f"Using device: {device}")
    # Prepare dataset
    dataset_exists = os.path.exists(cfg.image_dir) and len(os.listdir(cfg.image_dir))>0
    if dataset_exists:
        ds = ImageVoxelDataset(cfg.image_dir, cfg.voxel_dir, image_size=cfg.image_size, voxel_size=cfg.voxel_size)
        loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)
        print(f"Found {len(ds)} samples.")
    else:
        print("No dataset found in dataset/images. Using dummy data for quick test.")
        # Create a tiny synthetic dataset: random noise images with dummy voxels
        class DummyDS(torch.utils.data.Dataset):
            def __init__(self, n=32):
                import numpy as np
                self.n = n
            def __len__(self): return self.n
            def __getitem__(self, idx):
                img = (np.random.rand(3, cfg.image_size[0], cfg.image_size[1]).astype('float32'))
                # create a centered sphere voxel
                z = np.zeros((cfg.voxel_size,cfg.voxel_size,cfg.voxel_size), dtype=np.uint8)
                cx = cy = cz = cfg.voxel_size//2
                r = cfg.voxel_size//4
                for x in range(cfg.voxel_size):
                    for y in range(cfg.voxel_size):
                        for zc in range(cfg.voxel_size):
                            if (x-cx)**2 + (y-cy)**2 + (zc-cz)**2 <= r*r:
                                z[x,y,zc] = 1
                return torch.from_numpy(img).float(), torch.from_numpy(z).unsqueeze(0).float()
        loader = DataLoader(DummyDS(64), batch_size=cfg.batch_size, shuffle=True)

    # Model
    model = Reconstructor(latent_dim=cfg.latent_dim, voxel_size=cfg.voxel_size).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    global_step = 0
    for epoch in range(cfg.epochs):
        epoch_loss = 0.0
        epoch_iou = 0.0
        n_seen = 0
        t0 = time.time()
        for imgs, voxels in loader:
            imgs = imgs.to(device)
            voxels = voxels.to(device)
            # If images are HxW in dataset loader they should be CxHxW already
            preds = model(imgs)
            loss = criterion(preds, voxels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # metrics
            bs = imgs.size(0)
            epoch_loss += loss.item() * bs
            # compute iou on CPU numpy
            preds_np = preds.detach().cpu().numpy()
            vox_np = voxels.detach().cpu().numpy()
            batch_iou = 0.0
            for i in range(bs):
                batch_iou += voxel_iou(preds_np[i,0], vox_np[i,0])
            epoch_iou += batch_iou
            n_seen += bs
            global_step += 1

        epoch_loss /= n_seen
        epoch_iou /= n_seen
        t1 = time.time()
        print(f"Epoch {epoch+1}/{cfg.epochs} - Loss: {epoch_loss:.4f} - IoU: {epoch_iou:.4f} - Time: {t1-t0:.1f}s")

        # Save checkpoint and example visualization
        ckpt_path = os.path.join(cfg.model_dir, f"reconstructor_epoch{epoch+1}.pth")
        torch.save(model.state_dict(), ckpt_path)
        # save an example from last batch
        try:
            sample_pred = preds.detach().cpu().numpy()[0,0]
            sample_gt = voxels.detach().cpu().numpy()[0,0]
            vis_path = os.path.join(cfg.vis_dir, f"epoch{epoch+1}_slices.png")
            save_voxel_slices(sample_pred, vis_path)
            # also save mesh if trimesh available
            try:
                from .utils import voxel_to_mesh_and_save
                mesh_path = os.path.join(cfg.vis_dir, f"epoch{epoch+1}_mesh.ply")
                voxel_to_mesh_and_save((sample_pred >= 0.5).astype('uint8'), mesh_path)
            except Exception as e:
                # optional mesh saving failed (e.g., trimesh not installed)
                pass
        except Exception as e:
            print("Could not save visualization:", e)

    print("Training complete. Models saved to:", cfg.model_dir)
