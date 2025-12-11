"""
Encoder-Decoder model:
- 2D CNN encoder -> latent vector
- FC -> reshape -> ConvTranspose3d decoder -> voxel occupancy (sigmoid)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder2D(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()
        # Input: (B,3,128,128)
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),  # 64x64
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, 4, 2, 1), # 32x32
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1), #16x16
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1), #8x8
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1), #4x4
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Linear(512*4*4, latent_dim)

    def forward(self, x):
        B = x.shape[0]
        x = self.conv(x)
        x = x.view(B, -1)
        x = self.fc(x)
        return x

class Decoder3D(nn.Module):
    def __init__(self, latent_dim=512, voxel_size=32):
        super().__init__()
        self.voxel_size = voxel_size
        # We'll map latent to a small 3D feature cube, e.g., 4x4x4 with channels
        self.init_dim = 256
        self.init_size = 4  # 4x4x4
        self.fc = nn.Linear(latent_dim, self.init_dim * self.init_size**3)

        # Decoder with ConvTranspose3d to upsample to voxel_size
        self.deconv = nn.Sequential(
            nn.ConvTranspose3d(self.init_dim, 128, kernel_size=4, stride=2, padding=1), #8
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1), #16
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1), #32
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),

            nn.Conv3d(32, 1, kernel_size=3, padding=1),  # final occupancy logits
        )

    def forward(self, z):
        B = z.shape[0]
        x = self.fc(z)
        x = x.view(B, self.init_dim, self.init_size, self.init_size, self.init_size)
        x = self.deconv(x)  # (B,1,voxel_size,voxel_size,voxel_size)
        x = torch.sigmoid(x)  # occupancy probabilities
        return x

class Reconstructor(nn.Module):
    def __init__(self, latent_dim=512, voxel_size=32):
        super().__init__()
        self.encoder = Encoder2D(latent_dim=latent_dim)
        self.decoder = Decoder3D(latent_dim=latent_dim, voxel_size=voxel_size)

    def forward(self, img):
        z = self.encoder(img)
        vox = self.decoder(z)
        return vox
