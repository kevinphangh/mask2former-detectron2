#!/usr/bin/env python3
"""
Mask2Former Training with Swin-Small backbone and transfer learning.
Optimized for small datasets with pre-trained COCO weights.
"""

import os
import sys
from pathlib import Path
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "Mask2Former"))

from detectron2.config import get_cfg
from detectron2.engine import launch
from detectron2.projects.deeplab import add_deeplab_config
from mask2former import add_maskformer2_config

from src.training import Mask2FormerTrainer
from src.utils import register_datasets


def setup_cfg():
    """
    Setup configuration for Mask2Former with Swin-Small backbone.
    Uses transfer learning from COCO pre-trained weights.
    """
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    
    # Load Mask2Former with Swin-Small configuration
    config_file = project_root / "Mask2Former/configs/coco/instance-segmentation/swin/maskformer2_swin_small_bs16_50ep.yaml"
    cfg.merge_from_file(str(config_file))
    
    # TRANSFER LEARNING: Use pre-trained COCO weights
    # Download URL for Swin-Small Mask2Former trained on COCO
    cfg.MODEL.WEIGHTS = "https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/instance/maskformer2_swin_small_bs16_50ep/model_final_1e7f22.pkl"
    
    # Model configuration
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # IMPORTANT: Set number of classes for your dataset
    # Dataset has 2 classes: low_diff and pickable_surface
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 2  # Two classes in dataset
    
    # Swin-Small specific settings
    cfg.MODEL.SWIN.EMBED_DIM = 96
    cfg.MODEL.SWIN.DEPTHS = [2, 2, 18, 2]
    cfg.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
    cfg.MODEL.SWIN.WINDOW_SIZE = 7
    cfg.MODEL.SWIN.APE = False  # Absolute position embedding
    cfg.MODEL.SWIN.DROP_PATH_RATE = 0.3
    cfg.MODEL.SWIN.PATCH_NORM = True
    
    # Mask2Former specific settings
    cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES = 100
    cfg.MODEL.MASK_FORMER.DEC_LAYERS = 9  # Number of decoder layers
    
    # Dataset configuration
    cfg.DATASETS.TRAIN = ("custom_train",)
    cfg.DATASETS.TEST = ("custom_val",) if Path("data/valid/_annotations.coco.json").exists() else ()
    
    # TRAINING PARAMETERS (optimized for transfer learning)
    # Smaller batch size for stability
    cfg.SOLVER.IMS_PER_BATCH = 2
    
    # Lower learning rate for fine-tuning
    cfg.SOLVER.BASE_LR = 0.0001  # Reduced for transfer learning
    
    # Training schedule
    cfg.SOLVER.MAX_ITER = 3000  # Reduced iterations for transfer learning
    cfg.SOLVER.STEPS = (2000, 2700)  # Learning rate decay milestones
    cfg.SOLVER.GAMMA = 0.1  # LR decay factor
    
    # Warmup for stable training
    cfg.SOLVER.WARMUP_FACTOR = 0.001
    cfg.SOLVER.WARMUP_ITERS = 100
    cfg.SOLVER.WARMUP_METHOD = "linear"
    
    # Weight decay and optimizer
    cfg.SOLVER.WEIGHT_DECAY = 0.05
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1  # Lower LR for backbone (transfer learning)
    
    # Gradient clipping for stability
    cfg.SOLVER.CLIP_GRADIENTS.ENABLED = True
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "norm"
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 0.1
    cfg.SOLVER.CLIP_GRADIENTS.NORM_TYPE = 2.0
    
    # Mixed precision training for efficiency
    cfg.SOLVER.AMP.ENABLED = True
    
    # Checkpoint and evaluation
    cfg.SOLVER.CHECKPOINT_PERIOD = 500
    cfg.TEST.EVAL_PERIOD = 500
    
    # Input configuration
    cfg.INPUT.MASK_FORMAT = "bitmask"
    cfg.INPUT.MIN_SIZE_TRAIN = (480, 512, 544, 576, 608, 640)
    cfg.INPUT.MAX_SIZE_TRAIN = 1333
    cfg.INPUT.MIN_SIZE_TEST = 640
    cfg.INPUT.MAX_SIZE_TEST = 1333
    
    # Data augmentation for better generalization
    cfg.INPUT.RANDOM_FLIP = "horizontal"
    cfg.INPUT.FORMAT = "RGB"
    
    # Output directory
    cfg.OUTPUT_DIR = "outputs/experiments/swin_small_transfer"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    # DataLoader
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = True
    
    # Test configuration
    cfg.TEST.DETECTIONS_PER_IMAGE = 100
    
    return cfg


def main():
    """Main training function with transfer learning."""
    print("\n" + "="*70)
    print("MASK2FORMER TRAINING WITH SWIN-SMALL BACKBONE")
    print("Transfer Learning from COCO Pre-trained Weights")
    print("="*70)
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"\n✓ GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("\n⚠ No GPU detected. Training will be slow.")
    
    # Register datasets
    print("\n📊 Registering datasets...")
    train_ok, val_ok = register_datasets()
    
    if not train_ok:
        print("❌ Training dataset not found!")
        return
    
    # Setup configuration
    print("\n⚙️ Setting up Swin-Small configuration...")
    cfg = setup_cfg()
    
    # Print configuration summary
    print("\n" + "="*70)
    print("📋 TRAINING CONFIGURATION")
    print("="*70)
    print(f"  Model: Mask2Former with Swin-Small backbone")
    print(f"  Pre-trained weights: COCO Instance Segmentation")
    print(f"  Transfer Learning: ✓ Enabled")
    print(f"  Number of classes: {cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES}")
    print(f"  Batch size: {cfg.SOLVER.IMS_PER_BATCH}")
    print(f"  Learning rate: {cfg.SOLVER.BASE_LR}")
    print(f"  Backbone LR multiplier: {cfg.SOLVER.BACKBONE_MULTIPLIER}")
    print(f"  Max iterations: {cfg.SOLVER.MAX_ITER}")
    print(f"  Mixed precision: {'✓ Enabled' if cfg.SOLVER.AMP.ENABLED else '✗ Disabled'}")
    print(f"  Output directory: {cfg.OUTPUT_DIR}")
    print("="*70)
    
    # Initialize trainer
    print("\n🏋️ Initializing Mask2Former trainer...")
    trainer = Mask2FormerTrainer(cfg)
    
    # Load weights and resume training if checkpoint exists
    trainer.resume_or_load(resume=False)
    
    # Start training
    print("\n🚀 Starting training with transfer learning...")
    print("   This will download pre-trained weights (~400MB) on first run.")
    print("="*70 + "\n")
    
    # Train the model
    return trainer.train()


if __name__ == "__main__":
    # Launch training on single GPU
    launch(main, 1)