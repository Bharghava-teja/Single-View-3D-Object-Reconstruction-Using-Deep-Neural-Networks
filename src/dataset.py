"""
Custom PyTorch Dataset for image -> voxel pairs.
Expects:
- dataset/images/*.png or jpg
- dataset/voxels/*.npy  (same filename base as image)
If voxels missing, dataset will produce dummy voxels (useful for quick testing).
"""

import os
from torch.utils.data import Dataset
import numpy as np
import torch
from dataset.preprocess import load_and_preprocess_image, dummy_voxel_from_image

class ImageVoxelDataset(Dataset):
    def __init__(self, image_dir, voxel_dir, image_size=(128,128), voxel_size=32):
        self.image_dir = image_dir
        self.voxel_dir = voxel_dir
        self.image_size = image_size
        self.voxel_size = voxel_size

        # find images
        exts = [".png", ".jpg", ".jpeg"]
        self.samples = []
        for fn in os.listdir(self.image_dir) if os.path.exists(self.image_dir) else []:
            if any(fn.lower().endswith(e) for e in exts):
                img_path = os.path.join(self.image_dir, fn)
                base = os.path.splitext(fn)[0]
                voxel_path = os.path.join(self.voxel_dir, base + ".npy")
                self.samples.append((img_path, voxel_path))
        # If empty, create dummy list with no files (train script can fallback)
        if not self.samples:
            self.samples = []

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, voxel_path = self.samples[idx]
        img = load_and_preprocess_image(img_path, size=self.image_size)  # C,H,W (numpy)
        if os.path.exists(voxel_path):
            vox = np.load(voxel_path)
            # Ensure shape and binary values
            if vox.dtype != np.uint8 and vox.dtype != np.bool_:
                vox = (vox > 0.5).astype(np.uint8)
        else:
            vox = dummy_voxel_from_image(img, size=self.voxel_size)

        # Convert to torch tensors
        img_t = torch.from_numpy(img).float()
        vox_t = torch.from_numpy(vox).float()
        # Ensure shape: (1,D,H,W)
        if vox_t.dim() == 3:
            vox_t = vox_t.unsqueeze(0)
        return img_t, vox_t
