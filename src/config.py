import os

class Config:
    def __init__(self):
        # Paths
        self.root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.dataset_dir = os.path.join(self.root, "dataset")
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.voxel_dir = os.path.join(self.dataset_dir, "voxels")
        self.results_dir = os.path.join(self.root, "results")
        self.model_dir = os.path.join(self.results_dir, "saved_models")
        self.vis_dir = os.path.join(self.results_dir, "visualizations")
        self.loss_dir = os.path.join(self.results_dir, "loss_curves")

        # Data params
        self.image_size = (128, 128)      # HxW for input images
        self.voxel_size = 32             # cubic voxel resolution (32x32x32)

        # Training params
        self.batch_size = 8
        self.epochs = 30                  # small default for quick test; increase for real training
        self.lr = 1e-4

        # Model params
        self.latent_dim = 512

        # Device
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
