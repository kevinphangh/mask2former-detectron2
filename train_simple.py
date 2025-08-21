#!/usr/bin/env python3.12
"""
Simple training script that works with the existing setup.
Trains a Mask R-CNN model as a fallback since Mask2Former requires additional setup.
"""

import os
import sys
import json
from pathlib import Path

# Check if detectron2 is available, if not use the basic training
try:
    from detectron2.config import get_cfg
    from detectron2.data import MetadataCatalog
    from detectron2.data.datasets import register_coco_instances
    from detectron2.engine import DefaultTrainer
    from detectron2 import model_zoo
    USE_DETECTRON2 = True
except ImportError:
    USE_DETECTRON2 = False
    print("Detectron2 not available. Will use basic training approach.")

def train_with_detectron2():
    """Train using Detectron2"""
    # Register dataset
    data_path = Path("data")
    train_json = data_path / "train" / "_annotations.coco.json"
    val_json = data_path / "valid" / "_annotations.coco.json"
    
    if "cylinders_train" not in MetadataCatalog.list():
        register_coco_instances("cylinders_train", {}, str(train_json), str(data_path / "train"))
    if "cylinders_val" not in MetadataCatalog.list() and val_json.exists():
        register_coco_instances("cylinders_val", {}, str(val_json), str(data_path / "valid"))
    
    # Configure
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("cylinders_train",)
    cfg.DATASETS.TEST = ("cylinders_val",) if val_json.exists() else ()
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 1000
    cfg.OUTPUT_DIR = "outputs/simple_training"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    # Train
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    
    print(f"Training complete! Model saved to {cfg.OUTPUT_DIR}")

def train_basic():
    """Basic training without Detectron2 - just analyze the dataset"""
    print("\n" + "="*60)
    print("DATASET ANALYSIS")
    print("="*60)
    
    data_path = Path("data")
    
    # Analyze train dataset
    train_json = data_path / "train" / "_annotations.coco.json"
    if train_json.exists():
        with open(train_json, 'r') as f:
            train_data = json.load(f)
        print(f"\nTraining dataset:")
        print(f"  • Images: {len(train_data['images'])}")
        print(f"  • Annotations: {len(train_data['annotations'])}")
        print(f"  • Categories: {train_data['categories']}")
    
    # Analyze validation dataset
    val_json = data_path / "valid" / "_annotations.coco.json"
    if val_json.exists():
        with open(val_json, 'r') as f:
            val_data = json.load(f)
        print(f"\nValidation dataset:")
        print(f"  • Images: {len(val_data['images'])}")
        print(f"  • Annotations: {len(val_data['annotations'])}")
    
    # Analyze test dataset
    test_json = data_path / "test" / "_annotations.coco.json"
    if test_json.exists():
        with open(test_json, 'r') as f:
            test_data = json.load(f)
        print(f"\nTest dataset:")
        print(f"  • Images: {len(test_data['images'])}")
        print(f"  • Annotations: {len(test_data['annotations'])}")
    
    print("\n" + "="*60)
    print("To properly train Mask2Former, you need to:")
    print("1. Install gcc/g++ compiler: apt-get install gcc g++ build-essential")
    print("2. Install Detectron2: pip install 'git+https://github.com/facebookresearch/detectron2.git'")
    print("3. Compile CUDA kernels in Mask2Former/mask2former/modeling/pixel_decoder/ops")
    print("4. Run: python train_mask2former.py")
    print("="*60)

if __name__ == "__main__":
    if USE_DETECTRON2:
        print("Starting training with Detectron2...")
        train_with_detectron2()
    else:
        train_basic()