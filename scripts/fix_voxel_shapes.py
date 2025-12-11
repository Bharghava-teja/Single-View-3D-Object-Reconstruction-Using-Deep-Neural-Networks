"""
Fix voxel grid sizes: make all .npy files uniform 32x32x32 shape.
"""

import os
import numpy as np
from tqdm import tqdm

VOX_DIR = "dataset/voxels"
TARGET_SHAPE = (32, 32, 32)

def resize_or_pad(vox):
    D, H, W = vox.shape

    # Case 1: larger than target -> center crop
    if D > 32 or H > 32 or W > 32:
        start_d = max(0, (D - 32) // 2)
        start_h = max(0, (H - 32) // 2)
        start_w = max(0, (W - 32) // 2)
        vox = vox[start_d:start_d+32, start_h:start_h+32, start_w:start_w+32]

    # Case 2: smaller than target -> pad with zeros
    pad_d = max(0, (32 - vox.shape[0]) // 2)
    pad_h = max(0, (32 - vox.shape[1]) // 2)
    pad_w = max(0, (32 - vox.shape[2]) // 2)
    vox = np.pad(
        vox,
        ((pad_d, 32 - vox.shape[0] - pad_d),
         (pad_h, 32 - vox.shape[1] - pad_h),
         (pad_w, 32 - vox.shape[2] - pad_w)),
        mode="constant",
    )
    return vox

def fix_all_voxels():
    count = 0
    for fn in tqdm(os.listdir(VOX_DIR), desc="Fixing voxel shapes"):
        if not fn.endswith(".npy"):
            continue
        path = os.path.join(VOX_DIR, fn)
        try:
            vox = np.load(path)
            if vox.shape != TARGET_SHAPE:
                vox_fixed = resize_or_pad(vox)
                np.save(path, vox_fixed.astype(np.uint8))
                count += 1
        except Exception as e:
            print("Error with", fn, ":", e)
    print(f"✅ Fixed {count} voxel files to shape {TARGET_SHAPE}")

if __name__ == "__main__":
    fix_all_voxels()
