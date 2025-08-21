#!/usr/bin/env python3
"""
Mask2Former Training Script
Trains instance segmentation models on custom COCO-format datasets.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "Mask2Former"))

from detectron2.config import get_cfg
from detectron2.engine import launch
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.model_zoo import model_zoo
from mask2former import add_maskformer2_config

from src.training import Mask2FormerTrainer
from src.utils import register_datasets


def setup_cfg():
    """
    Setup configuration for Mask2Former training.
    
    Returns:
        Configuration object with all settings
    """
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    
    # Load base configuration
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    ))
    
    # Model configuration
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 2  # Adjust for your dataset
    
    # Dataset configuration
    cfg.DATASETS.TRAIN = ("custom_train",)
    cfg.DATASETS.TEST = ("custom_val",)
    
    # Training parameters
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 1000
    cfg.SOLVER.STEPS = (600, 900)
    cfg.SOLVER.CHECKPOINT_PERIOD = 500
    cfg.TEST.EVAL_PERIOD = 200
    
    # Gradient clipping
    cfg.SOLVER.CLIP_GRADIENTS.ENABLED = True
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "norm"
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 1.0
    cfg.SOLVER.CLIP_GRADIENTS.NORM_TYPE = 2.0
    
    # Input configuration
    cfg.INPUT.MASK_FORMAT = "bitmask"
    cfg.INPUT.MIN_SIZE_TRAIN = (640, 672, 704, 736, 768, 800)
    cfg.INPUT.MIN_SIZE_TEST = 800
    
    # Output directory
    cfg.OUTPUT_DIR = "outputs/experiments/latest"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    # DataLoader
    cfg.DATALOADER.NUM_WORKERS = 2
    
    return cfg


def main():
    """Main training function."""
    print("\n" + "="*60)
    print("MASK2FORMER TRAINING")
    print("="*60)
    
    # Register datasets
    print("\n📊 Registering datasets...")
    train_ok, val_ok = register_datasets()
    
    if not train_ok:
        print("❌ Training dataset not found!")
        return
    
    # Setup configuration
    print("\n⚙️ Setting up configuration...")
    cfg = setup_cfg()
    
    # Print configuration summary
    print("\n📋 Configuration Summary:")
    print(f"  • Model: Mask2Former")
    print(f"  • Classes: {cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES}")
    print(f"  • Batch Size: {cfg.SOLVER.IMS_PER_BATCH}")
    print(f"  • Max Iterations: {cfg.SOLVER.MAX_ITER}")
    print(f"  • Output: {cfg.OUTPUT_DIR}")
    
    # Initialize trainer
    print("\n🏋️ Initializing trainer...")
    trainer = Mask2FormerTrainer(cfg)
    trainer.resume_or_load(resume=False)
    
    # Start training
    print("\n🚀 Starting training...")
    print("="*60 + "\n")
    
    return trainer.train()


if __name__ == "__main__":
    launch(main, 1)  # num_gpus as positional argument