#!/usr/bin/env python3
"""
Training script for Mask2Former with proper mask loading.
"""

import argparse
import os
import sys
import torch
from pathlib import Path

# Add Mask2Former to path
sys.path.insert(0, str(Path(__file__).parent / "Mask2Former"))

from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer, launch
from detectron2.evaluation import COCOEvaluator
from detectron2.projects.deeplab import add_deeplab_config
from mask2former import add_maskformer2_config
from detectron2.data import DatasetMapper, build_detection_train_loader
import detectron2.data.transforms as T
from detectron2.model_zoo import model_zoo
import copy
import torch


def custom_mapper_with_masks(dataset_dict, mapper):
    """Wrapper to ensure masks are in the correct format for Mask2Former."""
    dataset_dict = mapper(dataset_dict)
    
    # Convert BitMasks to tensor if present
    if "instances" in dataset_dict and hasattr(dataset_dict["instances"], "gt_masks"):
        gt_masks = dataset_dict["instances"].gt_masks
        # If it's a BitMasks object, extract the tensor
        if hasattr(gt_masks, 'tensor'):
            dataset_dict["instances"].gt_masks = gt_masks.tensor
        elif not isinstance(gt_masks, torch.Tensor):
            # Convert to tensor if it's not already
            dataset_dict["instances"].gt_masks = torch.as_tensor(gt_masks)
    
    return dataset_dict


class Mask2FormerTrainer(DefaultTrainer):
    """Custom trainer for Mask2Former."""
    
    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        return COCOEvaluator(dataset_name, output_dir=cfg.OUTPUT_DIR)
    
    @classmethod
    def build_train_loader(cls, cfg):
        """Build train loader with mask loading."""
        base_mapper = DatasetMapper(
            cfg, 
            is_train=True,
            augmentations=[
                T.ResizeShortestEdge(
                    short_edge_length=cfg.INPUT.MIN_SIZE_TRAIN,
                    max_size=cfg.INPUT.MAX_SIZE_TRAIN,
                    sample_style=cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
                ),
                T.RandomFlip()
            ],
            use_instance_mask=True,  # Ensure masks are loaded
            instance_mask_format="bitmask",  # Convert to bitmask format
            recompute_boxes=False
        )
        # Wrap the mapper to convert masks to tensor format
        mapper = lambda x: custom_mapper_with_masks(x, base_mapper)
        return build_detection_train_loader(cfg, mapper=mapper)


def main():
    print("\n" + "="*60)
    print("MASK2FORMER TRAINING (SIMPLIFIED)")
    print("="*60)
    
    # Register datasets
    print("\n📊 Registering datasets...")
    
    # Clear existing registrations
    for name in ["cylinders_train", "cylinders_val"]:
        if name in DatasetCatalog.list():
            DatasetCatalog.remove(name)
            MetadataCatalog.remove(name)
    
    register_coco_instances(
        "cylinders_train", 
        {}, 
        "data/train/_annotations.coco.json",
        "data/train"
    )
    print("✓ Registered training dataset")
    
    if Path("data/valid/_annotations.coco.json").exists():
        register_coco_instances(
            "cylinders_val",
            {},
            "data/valid/_annotations.coco.json",
            "data/valid"
        )
        print("✓ Registered validation dataset")
    
    # Setup configuration
    print("\n⚙️ Setting up configuration...")
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    
    # Use base Mask2Former config
    cfg.merge_from_file("Mask2Former/configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml")
    
    # Override settings for our dataset
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 2  # Number of classes
    cfg.DATASETS.TRAIN = ("cylinders_train",)
    cfg.DATASETS.TEST = ("cylinders_val",) if Path("data/valid/_annotations.coco.json").exists() else ()
    
    # Training parameters
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 1000
    cfg.SOLVER.STEPS = (600, 900)
    cfg.SOLVER.CHECKPOINT_PERIOD = 500
    cfg.TEST.EVAL_PERIOD = 200
    
    # Fix gradient clipping
    cfg.SOLVER.CLIP_GRADIENTS.ENABLED = True
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "norm"
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 1.0
    cfg.SOLVER.CLIP_GRADIENTS.NORM_TYPE = 2.0
    
    # Input configuration
    cfg.INPUT.MASK_FORMAT = "bitmask"  # Mask2Former needs bitmask format
    cfg.INPUT.MIN_SIZE_TRAIN = (640, 672, 704, 736, 768, 800)
    cfg.INPUT.MIN_SIZE_TEST = 800
    
    # Output
    cfg.OUTPUT_DIR = "outputs/mask2former_v2"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    # DataLoader
    cfg.DATALOADER.NUM_WORKERS = 2
    
    print("\n📋 Configuration Summary:")
    print(f"  • Model: Mask2Former")
    print(f"  • Classes: {cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES}")
    print(f"  • Batch Size: {cfg.SOLVER.IMS_PER_BATCH}")
    print(f"  • Max Iterations: {cfg.SOLVER.MAX_ITER}")
    print(f"  • Output: {cfg.OUTPUT_DIR}")
    
    # Build trainer
    print("\n🏋️ Initializing trainer...")
    trainer = Mask2FormerTrainer(cfg)
    trainer.resume_or_load(resume=False)
    
    # Start training
    print("\n🚀 Starting training...")
    print("="*60 + "\n")
    
    return trainer.train()


if __name__ == "__main__":
    launch(main, 1)  # num_gpus as positional argument