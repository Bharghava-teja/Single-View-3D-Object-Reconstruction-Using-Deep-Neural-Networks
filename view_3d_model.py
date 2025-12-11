"""
view_3d_model.py
------------------------------------
Simple script to view reconstructed 3D model (.ply) using Trimesh viewer.
"""

import trimesh
import os
import sys

# Default model path (you can replace this with any .ply file)
DEFAULT_MODEL_PATH = os.path.join("results", "saved_models", "inference", "pred_mesh.ply")

def view_model(model_path=DEFAULT_MODEL_PATH):
    if not os.path.exists(model_path):
        print(f"❌ Model not found at: {model_path}")
        print("Please make sure you have run inference and that pred_mesh.ply exists.")
        sys.exit(1)

    print(f"✅ Loading model from: {model_path}")
    mesh = trimesh.load(model_path)

    print("🟢 Opening 3D viewer window... (use mouse to rotate/zoom)")
    mesh.show()

if __name__ == "__main__":
    # Allow custom path via command-line argument
    model_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL_PATH
    view_model(model_path)
