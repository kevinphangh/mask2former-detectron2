#!/usr/bin/env python3
"""
Official Mask2Former Training Script for Custom COCO Datasets.
Based on the official Facebook Research Mask2Former implementation.
Repository: https://github.com/facebookresearch/Mask2Former
"""

import os
import sys
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "Mask2Former"))

# Suppress warnings
try:
    from shapely.errors import ShapelyDeprecationWarning
    import warnings
    warnings.filterwarnings('ignore', category=ShapelyDeprecationWarning)
except:
    pass

import copy
import itertools
import logging
from collections import OrderedDict
from typing import Any, Dict, List, Set

import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_train_loader
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import COCOEvaluator, DatasetEvaluators, verify_results
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger
import detectron2.utils.comm as comm

# Import Mask2Former modules
from mask2former import (
    COCOInstanceNewBaselineDatasetMapper,
    add_maskformer2_config,
)

# Import custom dataset registration
from src.utils import register_datasets

# ==================== CONFIGURATION ====================
# Official Mask2Former Pre-trained Models from Facebook Research
# Model Zoo: https://github.com/facebookresearch/Mask2Former/blob/main/MODEL_ZOO.md
OFFICIAL_MODELS = {
    "swin_tiny": {
        "config": "configs/coco/instance-segmentation/swin/maskformer2_swin_tiny_bs16_50ep.yaml",
        "weights": "https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/instance/maskformer2_swin_tiny_bs16_50ep/model_final_86143f.pkl",
        "description": "Swin-Tiny (28M params, AP 45.0)"
    },
    "swin_small": {
        "config": "configs/coco/instance-segmentation/swin/maskformer2_swin_small_bs16_50ep.yaml",
        "weights": "https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/instance/maskformer2_swin_small_bs16_50ep/model_final_1e7f22.pkl",
        "description": "Swin-Small (50M params, AP 46.3)"
    },
    "swin_base": {
        "config": "configs/coco/instance-segmentation/swin/maskformer2_swin_base_IN21k_384_bs16_50ep.yaml",
        "weights": "https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/instance/maskformer2_swin_base_IN21k_384_bs16_50ep/model_final_83d103.pkl",
        "description": "Swin-Base-IN21k (88M params, AP 48.1)"
    },
    "resnet50": {
        "config": "configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml",
        "weights": "https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/instance/maskformer2_R50_bs16_50ep/model_final_3c8ec9.pkl",
        "description": "ResNet-50 (44M params, AP 43.7)"
    },
    "resnet101": {
        "config": "configs/coco/instance-segmentation/maskformer2_R101_bs16_50ep.yaml",
        "weights": "https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/instance/maskformer2_R101_bs16_50ep/model_final_eba159.pkl",
        "description": "ResNet-101 (63M params, AP 44.2)"
    }
}

# ============================================================
# SELECT YOUR MODEL HERE
# ============================================================
MODEL_NAME = "swin_small"  # Options: swin_tiny, swin_small, swin_base, resnet50, resnet101


class Mask2FormerTrainer(DefaultTrainer):
    """
    Extension of the Trainer class adapted to official Mask2Former.
    Based on the official implementation from Facebook Research.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator for COCO instance segmentation.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, output_dir=output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        """
        Build training data loader with Mask2Former dataset mapper.
        """
        # Use the official COCO instance mapper from Mask2Former
        mapper = COCOInstanceNewBaselineDatasetMapper(cfg, True)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        Build learning rate scheduler (polynomial decay).
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        """
        Build optimizer with weight decay settings from official implementation.
        """
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = weight_decay

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                if (
                    "relative_position_bias_table" in module_param_name
                    or "absolute_pos_embed" in module_param_name
                ):
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer


def setup_cfg(args=None):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    
    # Add config from Detectron2 and Mask2Former
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    
    # Load official Mask2Former configuration
    model_info = OFFICIAL_MODELS[MODEL_NAME]
    config_file = project_root / "Mask2Former" / model_info["config"]
    cfg.merge_from_file(str(config_file))
    
    # Use official pre-trained weights
    cfg.MODEL.WEIGHTS = model_info["weights"]
    
    # Check for local model file
    local_model_path = project_root / f"models/maskformer2_{MODEL_NAME}.pkl"
    if local_model_path.exists():
        print(f"✓ Using local model: {local_model_path}")
        cfg.MODEL.WEIGHTS = str(local_model_path)
    
    # Override for custom dataset
    cfg.DATASETS.TRAIN = ("custom_train",)
    cfg.DATASETS.TEST = ("custom_val",) if Path("data/valid/_annotations.coco.json").exists() else ()
    
    # IMPORTANT: Set number of classes for your dataset
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 2  # Adjust based on your dataset
    
    # Training hyperparameters (adjust based on your GPU and dataset)
    cfg.SOLVER.IMS_PER_BATCH = 2  # Batch size
    cfg.SOLVER.BASE_LR = 0.0001  # Learning rate for transfer learning
    cfg.SOLVER.MAX_ITER = 3000  # Training iterations
    cfg.SOLVER.STEPS = (2000, 2700)  # LR decay steps
    cfg.SOLVER.CHECKPOINT_PERIOD = 500
    cfg.TEST.EVAL_PERIOD = 500
    
    # Set weight decay parameters (from official config)
    cfg.SOLVER.WEIGHT_DECAY = 0.05
    cfg.SOLVER.WEIGHT_DECAY_NORM = 0.0
    cfg.SOLVER.WEIGHT_DECAY_EMBED = 0.0
    
    # Backbone learning rate multiplier for transfer learning
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1
    
    # Input configuration
    cfg.INPUT.MIN_SIZE_TRAIN = (480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800)
    cfg.INPUT.MIN_SIZE_TEST = 800
    cfg.INPUT.MAX_SIZE_TRAIN = 1333
    cfg.INPUT.MAX_SIZE_TEST = 1333
    
    # Dataset mapper for instance segmentation
    cfg.INPUT.DATASET_MAPPER_NAME = "coco_instance_lsj"
    cfg.INPUT.MASK_FORMAT = "bitmask"
    
    # Output directory
    cfg.OUTPUT_DIR = f"outputs/mask2former_{MODEL_NAME}"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    # DataLoader
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = True
    
    # Merge command line arguments if provided
    if args:
        cfg.merge_from_list(args.opts)
    
    cfg.freeze()
    return cfg


def main(args):
    """
    Main training function.
    """
    # Print header
    model_info = OFFICIAL_MODELS[MODEL_NAME]
    print("\n" + "="*70)
    print("OFFICIAL MASK2FORMER TRAINING")
    print(f"Model: {model_info['description']}")
    print("Repository: https://github.com/facebookresearch/Mask2Former")
    print("="*70)
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"\n✓ GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("\n⚠ No GPU detected. Training will be slow.")
    
    # Register custom datasets
    print("\n📊 Registering datasets...")
    train_ok, val_ok = register_datasets()
    
    if not train_ok:
        print("❌ Training dataset not found!")
        print("   Please ensure data/train/_annotations.coco.json exists")
        return
    
    print(f"  ✓ Training dataset registered: custom_train")
    if val_ok:
        print(f"  ✓ Validation dataset registered: custom_val")
    else:
        print(f"  ⚠ No validation dataset found")
    
    # Setup configuration
    print(f"\n⚙️ Setting up {MODEL_NAME.replace('_', '-').title()} configuration...")
    cfg = setup_cfg(args)
    
    # Print configuration summary
    print("\n" + "="*70)
    print("📋 TRAINING CONFIGURATION")
    print("="*70)
    print(f"  Model: {model_info['description']}")
    print(f"  Pre-trained weights: Official Facebook Research")
    print(f"  Number of classes: {cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES}")
    print(f"  Batch size: {cfg.SOLVER.IMS_PER_BATCH}")
    print(f"  Learning rate: {cfg.SOLVER.BASE_LR}")
    print(f"  Backbone LR multiplier: {cfg.SOLVER.BACKBONE_MULTIPLIER}")
    print(f"  Max iterations: {cfg.SOLVER.MAX_ITER}")
    print(f"  Mixed precision: {'✓ Enabled' if cfg.SOLVER.AMP.ENABLED else '✗ Disabled'}")
    print(f"  Output directory: {cfg.OUTPUT_DIR}")
    print("="*70)
    
    # Setup logger
    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="mask2former")
    
    # Build trainer
    print("\n🏋️ Initializing Mask2Former trainer...")
    trainer = Mask2FormerTrainer(cfg)
    
    # Resume or load model
    trainer.resume_or_load(resume=args.resume)
    
    # Start training
    print("\n🚀 Starting training...")
    print(f"   Using official model: {model_info['description']}")
    print("   Weights will be downloaded from Facebook Research on first run.")
    print("="*70 + "\n")
    
    # Train
    return trainer.train()


if __name__ == "__main__":
    parser = default_argument_parser()
    args = parser.parse_args()
    
    # Launch training
    print("Launching training...")
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )