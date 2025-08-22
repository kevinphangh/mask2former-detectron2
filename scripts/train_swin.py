#!/usr/bin/env python3
"""
Mask2Former Training with Official Pre-trained Models from Facebook Research.
Uses transfer learning from official Mask2Former COCO weights.
Repository: https://github.com/facebookresearch/Mask2Former
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

# Official Mask2Former Pre-trained Models (Facebook Research)
# Source: https://github.com/facebookresearch/Mask2Former/blob/main/MODEL_ZOO.md
OFFICIAL_MODELS = {
    "swin_tiny": {
        "config": "Mask2Former/configs/coco/instance-segmentation/swin/maskformer2_swin_tiny_bs16_50ep.yaml",
        "weights": "https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/instance/maskformer2_swin_tiny_bs16_50ep/model_final_86143f.pkl",
        "description": "Swin-Tiny (28M params, fastest)"
    },
    "swin_small": {
        "config": "Mask2Former/configs/coco/instance-segmentation/swin/maskformer2_swin_small_bs16_50ep.yaml",
        "weights": "https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/instance/maskformer2_swin_small_bs16_50ep/model_final_1e7f22.pkl",
        "description": "Swin-Small (50M params, recommended)"
    },
    "swin_base": {
        "config": "Mask2Former/configs/coco/instance-segmentation/swin/maskformer2_swin_base_IN21k_384_bs16_50ep.yaml",
        "weights": "https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/instance/maskformer2_swin_base_IN21k_384_bs16_50ep/model_final_83d103.pkl",
        "description": "Swin-Base (88M params, best accuracy)"
    },
    "resnet50": {
        "config": "Mask2Former/configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml",
        "weights": "https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/instance/maskformer2_R50_bs16_50ep/model_final_3c8ec9.pkl",
        "description": "ResNet-50 (44M params)"
    },
    "resnet101": {
        "config": "Mask2Former/configs/coco/instance-segmentation/maskformer2_R101_bs16_50ep.yaml",
        "weights": "https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/instance/maskformer2_R101_bs16_50ep/model_final_eba159.pkl",
        "description": "ResNet-101 (63M params)"
    }
}

# ==================== SELECT YOUR MODEL ====================
# Choose from official Mask2Former pre-trained models:
#   - "swin_tiny"   : 28M params, fastest, good for testing
#   - "swin_small"  : 50M params, recommended balance
#   - "swin_base"   : 88M params, best accuracy (needs 16GB+ GPU)
#   - "resnet50"    : 44M params, classic CNN backbone
#   - "resnet101"   : 63M params, deeper ResNet variant
# ============================================================
MODEL_NAME = "swin_small"  # <-- CHANGE THIS TO SELECT MODEL


def setup_cfg():
    """
    Setup configuration for official Mask2Former pre-trained models.
    Uses transfer learning from official Facebook Research COCO weights.
    """
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    
    # Load official Mask2Former configuration
    model_info = OFFICIAL_MODELS[MODEL_NAME]
    config_file = project_root / model_info["config"]
    cfg.merge_from_file(str(config_file))
    
    # Use official pre-trained weights from Facebook Research
    cfg.MODEL.WEIGHTS = model_info["weights"]
    
    # Check if using local downloaded model
    local_model_path = project_root / f"models/maskformer2_{MODEL_NAME}.pkl"
    if local_model_path.exists():
        print(f"✓ Using local model: {local_model_path}")
        cfg.MODEL.WEIGHTS = str(local_model_path)
    
    # Model configuration
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # IMPORTANT: Set number of classes for your dataset
    # Dataset has 2 classes: low_diff and pickable_surface
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 2  # Two classes in dataset
    
    # Model-specific settings (only for Swin models)
    if "swin" in MODEL_NAME:
        if MODEL_NAME == "swin_tiny":
            cfg.MODEL.SWIN.EMBED_DIM = 96
            cfg.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
            cfg.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
        elif MODEL_NAME == "swin_small":
            cfg.MODEL.SWIN.EMBED_DIM = 96
            cfg.MODEL.SWIN.DEPTHS = [2, 2, 18, 2]
            cfg.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
        elif MODEL_NAME == "swin_base":
            cfg.MODEL.SWIN.EMBED_DIM = 128
            cfg.MODEL.SWIN.DEPTHS = [2, 2, 18, 2]
            cfg.MODEL.SWIN.NUM_HEADS = [4, 8, 16, 32]
        
        cfg.MODEL.SWIN.WINDOW_SIZE = 7
        cfg.MODEL.SWIN.APE = False  # Absolute position embedding
        cfg.MODEL.SWIN.DROP_PATH_RATE = 0.3
        cfg.MODEL.SWIN.PATCH_NORM = True
    
    # Mask2Former specific settings
    cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES = 100
    # Decoder layers: 9 for Swin models, 6 for ResNet models
    cfg.MODEL.MASK_FORMER.DEC_LAYERS = 9 if "swin" in MODEL_NAME else 6
    
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
    
    # Output directory based on selected model
    cfg.OUTPUT_DIR = f"outputs/experiments/{MODEL_NAME}_transfer"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    # DataLoader
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = True
    
    # Test configuration
    cfg.TEST.DETECTIONS_PER_IMAGE = 100
    
    return cfg


def main():
    """Main training function with official Mask2Former models."""
    model_info = OFFICIAL_MODELS[MODEL_NAME]
    print("\n" + "="*70)
    print("MASK2FORMER TRAINING WITH OFFICIAL PRE-TRAINED MODELS")
    print(f"Model: {model_info['description']}")
    print("Transfer Learning from Official Facebook Research COCO Weights")
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
    print(f"\n⚙️ Setting up {MODEL_NAME.replace('_', '-').title()} configuration...")
    cfg = setup_cfg()
    
    # Print configuration summary
    print("\n" + "="*70)
    print("📋 TRAINING CONFIGURATION")
    print("="*70)
    print(f"  Model: Official Mask2Former - {model_info['description']}")
    print(f"  Pre-trained weights: Facebook Research COCO Instance Segmentation")
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
    print("\n🚀 Starting training with official Mask2Former models...")
    print(f"   Using: {model_info['description']}")
    print("   Weights will be downloaded from Facebook Research on first run.")
    print("="*70 + "\n")
    
    # Train the model
    return trainer.train()


if __name__ == "__main__":
    # Launch training on single GPU
    launch(main, 1)