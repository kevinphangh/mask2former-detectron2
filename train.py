#!/usr/bin/env python3
"""
Training script for Mask R-CNN cylinder detection using Detectron2.
This script trains a Mask R-CNN model on a COCO-format dataset for instance segmentation.
"""

import argparse
import os
import torch
from pathlib import Path

from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import COCOEvaluator
from detectron2 import model_zoo
from detectron2.utils.logger import setup_logger


class Trainer(DefaultTrainer):
    """Custom trainer with COCO evaluation."""
    
    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        """Build COCO evaluator for validation."""
        return COCOEvaluator(dataset_name, output_dir=cfg.OUTPUT_DIR)


def register_datasets(data_dir):
    """
    Register COCO format datasets for training and validation.
    
    Args:
        data_dir: Path to directory containing train/valid subdirectories
    """
    data_path = Path(data_dir)
    
    # Register training dataset
    train_json = data_path / "train" / "_annotations.coco.json"
    if train_json.exists():
        if "cylinders_train" not in MetadataCatalog.list():
            register_coco_instances(
                "cylinders_train", 
                {}, 
                str(train_json),
                str(data_path / "train")
            )
            print(f"✓ Registered training dataset from {train_json}")
    else:
        raise FileNotFoundError(f"Training annotations not found: {train_json}")
    
    # Register validation dataset
    val_json = data_path / "valid" / "_annotations.coco.json"
    if val_json.exists():
        if "cylinders_val" not in MetadataCatalog.list():
            register_coco_instances(
                "cylinders_val",
                {},
                str(val_json),
                str(data_path / "valid")
            )
            print(f"✓ Registered validation dataset from {val_json}")
    else:
        print(f"⚠️ Validation annotations not found: {val_json}")


def setup_cfg(args):
    """
    Create and setup configuration for training.
    
    Args:
        args: Command line arguments
        
    Returns:
        cfg: Detectron2 configuration
    """
    cfg = get_cfg()
    
    # Load base configuration
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    else:
        # Default to Mask R-CNN with ResNet-50-FPN
        cfg.merge_from_file(model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        ))
    
    # Set datasets
    cfg.DATASETS.TRAIN = ("cylinders_train",)
    cfg.DATASETS.TEST = ("cylinders_val",) if "cylinders_val" in MetadataCatalog.list() else ()
    
    # Model configuration
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = args.num_classes
    
    # Load pretrained weights
    if args.weights:
        cfg.MODEL.WEIGHTS = args.weights
    else:
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
    
    # Training configuration
    cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    cfg.SOLVER.BASE_LR = args.learning_rate
    cfg.SOLVER.MAX_ITER = args.max_iter
    cfg.SOLVER.STEPS = tuple([int(args.max_iter * 0.6), int(args.max_iter * 0.9)])
    cfg.SOLVER.CHECKPOINT_PERIOD = args.checkpoint_period
    
    # Evaluation
    cfg.TEST.EVAL_PERIOD = args.eval_period
    
    # Output directory
    cfg.OUTPUT_DIR = args.output_dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    # DataLoader
    cfg.DATALOADER.NUM_WORKERS = args.num_workers
    
    # Mixed precision training
    cfg.SOLVER.AMP.ENABLED = args.amp
    
    # Additional options from command line
    cfg.merge_from_list(args.opts)
    
    cfg.freeze()
    return cfg


def main(args):
    """Main training function."""
    # Setup logger
    setup_logger()
    
    # Print system info
    print("\n" + "="*60)
    print("MASK R-CNN TRAINING WITH DETECTRON2")
    print("="*60)
    
    if torch.cuda.is_available():
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠️ No GPU detected, training will be slow")
    
    # Register datasets
    print("\n📊 Registering datasets...")
    register_datasets(args.data_dir)
    
    # Setup configuration
    print("\n⚙️ Setting up configuration...")
    cfg = setup_cfg(args)
    
    # Print configuration summary
    print("\n📋 Configuration Summary:")
    print(f"  • Model: {cfg.MODEL.META_ARCHITECTURE}")
    print(f"  • Backbone: {cfg.MODEL.BACKBONE.NAME if hasattr(cfg.MODEL, 'BACKBONE') else 'ResNet-50-FPN'}")
    print(f"  • Classes: {cfg.MODEL.ROI_HEADS.NUM_CLASSES}")
    print(f"  • Batch Size: {cfg.SOLVER.IMS_PER_BATCH}")
    print(f"  • Learning Rate: {cfg.SOLVER.BASE_LR}")
    print(f"  • Max Iterations: {cfg.SOLVER.MAX_ITER}")
    print(f"  • Checkpoint Period: {cfg.SOLVER.CHECKPOINT_PERIOD}")
    print(f"  • Evaluation Period: {cfg.TEST.EVAL_PERIOD}")
    print(f"  • Output Directory: {cfg.OUTPUT_DIR}")
    
    # Setup default training
    default_setup(cfg, args)
    
    # Build trainer
    print("\n🏋️ Initializing trainer...")
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    
    # Start training
    print("\n🚀 Starting training...")
    print("="*60)
    
    return trainer.train()


def get_parser():
    """Get argument parser."""
    parser = argparse.ArgumentParser(
        description="Train Mask R-CNN for cylinder detection"
    )
    
    # Basic options
    parser.add_argument(
        "--config-file", 
        default="", 
        metavar="FILE", 
        help="Path to config file"
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Path to dataset directory (default: data)"
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Output directory for checkpoints and logs (default: outputs)"
    )
    
    # Model options
    parser.add_argument(
        "--num-classes",
        type=int,
        default=1,
        help="Number of classes (default: 1)"
    )
    parser.add_argument(
        "--weights",
        default="",
        help="Path to pretrained weights or checkpoint"
    )
    
    # Training options
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size (default: 2)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.00025,
        help="Base learning rate (default: 0.00025)"
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=3000,
        help="Maximum training iterations (default: 3000)"
    )
    parser.add_argument(
        "--checkpoint-period",
        type=int,
        default=500,
        help="Save checkpoint every N iterations (default: 500)"
    )
    parser.add_argument(
        "--eval-period",
        type=int,
        default=500,
        help="Evaluate every N iterations (default: 500)"
    )
    
    # System options
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of data loading workers (default: 2)"
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable automatic mixed precision training"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from last checkpoint"
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs to use (default: 1)"
    )
    parser.add_argument(
        "--num-machines",
        type=int,
        default=1,
        help="Number of machines for distributed training"
    )
    parser.add_argument(
        "--machine-rank",
        type=int,
        default=0,
        help="Rank of this machine in distributed training"
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    print(f"Command Line Args: {args}")
    
    # Launch training (supports distributed training)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url="auto",
        args=(args,),
    )