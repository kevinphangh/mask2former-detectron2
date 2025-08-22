#!/usr/bin/env python3
"""
Mask2Former Training Script - Simplified Version
Train instance segmentation models on custom COCO datasets.
"""

import os
import sys
import copy
import itertools
import logging
from pathlib import Path
from collections import OrderedDict
from typing import Any, Dict, List, Set
import warnings

# Suppress known deprecation warnings
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.autocast.*")
warnings.filterwarnings("ignore", message=".*Importing from timm.models.layers.*")
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_train_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import COCOEvaluator, DatasetEvaluators, verify_results
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger
import detectron2.utils.comm as comm

# Add Mask2Former to path
sys.path.insert(0, str(Path(__file__).parent / "Mask2Former"))
from mask2former import COCOInstanceNewBaselineDatasetMapper, add_maskformer2_config

# ==================== CONFIGURATION ====================
# Select your model: "swin_tiny" (6GB GPU), "swin_small" (8GB), "swin_base" (16GB)
MODEL_NAME = "swin_tiny"

MODELS = {
    "swin_tiny": {
        "config": "Mask2Former/configs/coco/instance-segmentation/swin/maskformer2_swin_tiny_bs16_50ep.yaml",
        "weights": "https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/instance/maskformer2_swin_tiny_bs16_50ep/model_final_86143f.pkl",
    },
    "swin_small": {
        "config": "Mask2Former/configs/coco/instance-segmentation/swin/maskformer2_swin_small_bs16_50ep.yaml",
        "weights": "https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/instance/maskformer2_swin_small_bs16_50ep/model_final_1e7f22.pkl",
    },
    "swin_base": {
        "config": "Mask2Former/configs/coco/instance-segmentation/swin/maskformer2_swin_base_IN21k_384_bs16_50ep.yaml",
        "weights": "https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/instance/maskformer2_swin_base_IN21k_384_bs16_50ep/model_final_83d103.pkl",
    }
}


def register_datasets():
    """Register COCO format datasets."""
    train_registered = False
    val_registered = False
    
    # Clear existing registrations
    for name in ["custom_train", "custom_val"]:
        if name in DatasetCatalog.list():
            DatasetCatalog.remove(name)
            MetadataCatalog.remove(name)
    
    # Register training dataset
    train_json = Path("data/train/_annotations.coco.json")
    if train_json.exists():
        register_coco_instances("custom_train", {}, str(train_json), "data/train")
        train_registered = True
        print("✓ Registered training dataset")
    
    # Register validation dataset
    val_json = Path("data/valid/_annotations.coco.json")
    if val_json.exists():
        register_coco_instances("custom_val", {}, str(val_json), "data/valid")
        val_registered = True
        print("✓ Registered validation dataset")
    
    return train_registered, val_registered


class Mask2FormerTrainer(DefaultTrainer):
    """Trainer class for Mask2Former."""

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, output_dir=output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = COCOInstanceNewBaselineDatasetMapper(cfg, True)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = weight_decay

        norm_module_types = (
            torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm, torch.nn.GroupNorm, torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d, torch.nn.InstanceNorm3d, torch.nn.LayerNorm,
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
                if ("relative_position_bias_table" in module_param_name or 
                    "absolute_pos_embed" in module_param_name):
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (cfg.SOLVER.CLIP_GRADIENTS.ENABLED and 
                     cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model" and 
                     clip_norm_val > 0.0)

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
    """Create configs and perform basic setups."""
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    
    # Load model configuration
    model_info = MODELS[MODEL_NAME]
    cfg.merge_from_file(model_info["config"])
    cfg.MODEL.WEIGHTS = model_info["weights"]
    
    # Check for local model
    local_model = Path(f"models/maskformer2_{MODEL_NAME}.pkl")
    if local_model.exists():
        print(f"✓ Using local model: {local_model}")
        cfg.MODEL.WEIGHTS = str(local_model)
    
    # Dataset configuration
    cfg.DATASETS.TRAIN = ("custom_train",)
    cfg.DATASETS.TEST = ("custom_val",) if Path("data/valid/_annotations.coco.json").exists() else ()
    
    # IMPORTANT: Set number of classes for your dataset
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 2  # Adjust based on your dataset
    
    # Training parameters (optimized for 6GB GPU)
    cfg.SOLVER.IMS_PER_BATCH = 2  # Batch size
    cfg.SOLVER.BASE_LR = 0.00005  # Learning rate
    
    # Choose between iteration-based or epoch-based configuration
    USE_EPOCHS = False  # Set to True to use epoch-based configuration
    
    if USE_EPOCHS:
        # Epoch-based configuration (84 images, batch size 2 = 42 iters/epoch)
        IMAGES_PER_EPOCH = 84
        ITERS_PER_EPOCH = IMAGES_PER_EPOCH // cfg.SOLVER.IMS_PER_BATCH
        TARGET_EPOCHS = 50
        cfg.SOLVER.MAX_ITER = TARGET_EPOCHS * ITERS_PER_EPOCH  # 50 epochs
        cfg.SOLVER.STEPS = (30 * ITERS_PER_EPOCH, 40 * ITERS_PER_EPOCH)  # LR decay at 30, 40 epochs
        cfg.SOLVER.CHECKPOINT_PERIOD = 5 * ITERS_PER_EPOCH  # Every 5 epochs
        cfg.TEST.EVAL_PERIOD = 2 * ITERS_PER_EPOCH  # Every 2 epochs
    else:
        # Iteration-based configuration (current)
        cfg.SOLVER.MAX_ITER = 3000  # ~71 epochs with batch size 2
        cfg.SOLVER.STEPS = (2000, 2700)  # LR decay steps
        cfg.SOLVER.CHECKPOINT_PERIOD = 500  # Every ~12 epochs
        cfg.TEST.EVAL_PERIOD = 500  # Every ~12 epochs
    
    # Weight decay
    cfg.SOLVER.WEIGHT_DECAY = 0.05
    cfg.SOLVER.WEIGHT_DECAY_NORM = 0.0
    cfg.SOLVER.WEIGHT_DECAY_EMBED = 0.0
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1
    
    # Input sizes (reduced for memory)
    cfg.INPUT.MIN_SIZE_TRAIN = (384, 416, 448, 480, 512)
    cfg.INPUT.MIN_SIZE_TEST = 512
    cfg.INPUT.MAX_SIZE_TRAIN = 768
    cfg.INPUT.MAX_SIZE_TEST = 768
    
    # Dataset mapper
    cfg.INPUT.DATASET_MAPPER_NAME = "coco_instance_lsj"
    cfg.INPUT.MASK_FORMAT = "bitmask"
    
    # Output directory
    cfg.OUTPUT_DIR = f"outputs/{MODEL_NAME}"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    # DataLoader
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = True
    
    if args:
        cfg.merge_from_list(args.opts)
    
    cfg.freeze()
    return cfg


def main(args):
    """Main training function."""
    print("\n" + "="*70)
    print(f"MASK2FORMER TRAINING - {MODEL_NAME.upper()}")
    print("="*70)
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("⚠ No GPU detected")
    
    # Register datasets
    print("\n📊 Registering datasets...")
    train_ok, val_ok = register_datasets()
    
    if not train_ok:
        print("❌ Training dataset not found!")
        return
    
    # Setup configuration
    print(f"\n⚙️ Setting up configuration...")
    cfg = setup_cfg(args)
    
    print("\n📋 Configuration:")
    print(f"  Classes: {cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES}")
    print(f"  Batch size: {cfg.SOLVER.IMS_PER_BATCH}")
    print(f"  Learning rate: {cfg.SOLVER.BASE_LR}")
    print(f"  Max iterations: {cfg.SOLVER.MAX_ITER}")
    print(f"  Output: {cfg.OUTPUT_DIR}")
    
    # Setup logger
    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="mask2former")
    
    # Build trainer
    print("\n🚀 Starting training...")
    trainer = Mask2FormerTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    
    return trainer.train()


if __name__ == "__main__":
    parser = default_argument_parser()
    args = parser.parse_args()
    
    print("Launching Mask2Former training...")
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )