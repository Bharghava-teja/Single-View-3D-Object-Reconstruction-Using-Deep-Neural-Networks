"""
Convert Pix3D dataset into your project format:
    dataset/images/
    dataset/voxels/

Requirements:
- trimesh
- numpy
- pillow (for image copying)
- json
"""

import os
import json
import numpy as np
import trimesh
from PIL import Image
from tqdm import tqdm

# Paths (edit if needed)
PIX3D_ROOT = "Pix3D"
OUT_ROOT = "dataset"
IMG_OUT = os.path.join(OUT_ROOT, "images")
VOX_OUT = os.path.join(OUT_ROOT, "voxels")

os.makedirs(IMG_OUT, exist_ok=True)
os.makedirs(VOX_OUT, exist_ok=True)

# Load metadata
meta_path = os.path.join(PIX3D_ROOT, "pix3d.json")
print(f"Loading metadata from: {meta_path}")
with open(meta_path, "r") as f:
    meta = json.load(f)

count = 0
for item in tqdm(meta, desc="Processing Pix3D"):
    try:
        img_path = os.path.join(PIX3D_ROOT, item["img"])
        obj_path = os.path.join(PIX3D_ROOT, item["model"])

        if not os.path.exists(img_path) or not os.path.exists(obj_path):
            continue

        # Image output path
        base_name = f"{os.path.splitext(os.path.basename(img_path))[0]}_{count}"
        img_out_path = os.path.join(IMG_OUT, f"{base_name}.jpg")

        # Copy and resize image
        img = Image.open(img_path).convert("RGB")
        img = img.resize((128, 128))
        img.save(img_out_path)

        # Convert 3D model to voxel grid (32x32x32)
        mesh = trimesh.load(obj_path, force='mesh')
        voxel = mesh.voxelized(pitch=0.02).matrix.astype(np.uint8)

        # Normalize voxel size to 32³
        v = voxel
        D = v.shape[0]
        if D > 32:
            factor = D // 32
            v = v[::factor, ::factor, ::factor]
        elif D < 32:
            pad = (32 - D) // 2
            v = np.pad(v, pad_width=((pad, pad), (pad, pad), (pad, pad)), mode='constant')

        np.save(os.path.join(VOX_OUT, f"{base_name}.npy"), v)
        count += 1

    except Exception as e:
        print(f"Error with {item['img']}: {e}")

print(f"\n✅ Conversion complete! Generated {count} image-voxel pairs in {OUT_ROOT}/")
