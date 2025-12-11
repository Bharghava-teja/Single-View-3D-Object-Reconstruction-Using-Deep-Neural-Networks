"""
Main entrypoint: quick demo pipeline:
- checks dataset presence
- trains a short demo model (or a real run if dataset provided)
- saves model and visualizations
"""
import os
from src.config import Config
from src.train import train
from src.utils import ensure_dirs

if __name__ == "__main__":
    cfg = Config()
    ensure_dirs(cfg)
    train(cfg)
