#!/usr/bin/env python3
"""
Mask2Former Training Script V2 - Configuration-based
Train instance segmentation models using external YAML configuration.
"""

import os
import sys
import copy
import itertools
import logging
from pathlib import Path
from datetime import datetime
from collections import OrderedDict
from typing import Any, Dict, List, Set
import warnings
import yaml

# Suppress known deprecation warnings
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.autocast.*")
warnings.filterwarnings("ignore", message=".*Importing from timm.models.layers.*")
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_train_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch, hooks
from detectron2.evaluation import COCOEvaluator, DatasetEvaluators, verify_results
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger
from detectron2.utils.events import EventWriter, get_event_storage
import detectron2.utils.comm as comm

# Add Mask2Former to path
sys.path.insert(0, str(Path(__file__).parent / "Mask2Former"))
from mask2former import COCOInstanceNewBaselineDatasetMapper, add_maskformer2_config

# Model configurations
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


def load_custom_config(config_path: str) -> dict:
    """Load custom configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def register_datasets(custom_cfg: dict):
    """Register COCO format datasets."""
    train_registered = False
    val_registered = False
    
    # Clear existing registrations
    for name in [custom_cfg['DATASET']['TRAIN_NAME'], custom_cfg['DATASET']['VAL_NAME']]:
        if name in DatasetCatalog.list():
            DatasetCatalog.remove(name)
            MetadataCatalog.remove(name)
    
    # Register training dataset
    train_json = Path(custom_cfg['DATASET']['TRAIN_JSON'])
    if train_json.exists():
        register_coco_instances(
            custom_cfg['DATASET']['TRAIN_NAME'], 
            {}, 
            str(train_json), 
            custom_cfg['DATASET']['TRAIN_DIR']
        )
        train_registered = True
        print("✓ Registered training dataset")
    
    # Register validation dataset
    val_json = Path(custom_cfg['DATASET']['VAL_JSON'])
    if val_json.exists():
        register_coco_instances(
            custom_cfg['DATASET']['VAL_NAME'], 
            {}, 
            str(val_json), 
            custom_cfg['DATASET']['VAL_DIR']
        )
        val_registered = True
        print("✓ Registered validation dataset")
    
    return train_registered, val_registered


class EpochPrintHook(hooks.HookBase):
    """Hook to print epoch information during training."""
    
    def __init__(self, max_epochs, iters_per_epoch):
        self.max_epochs = max_epochs
        self.iters_per_epoch = iters_per_epoch
    
    def after_step(self):
        # Only print every 20 iterations (matching detectron2's default)
        if self.trainer.iter > 0 and (self.trainer.iter + 1) % 20 == 0:
            current_epoch = (self.trainer.iter // self.iters_per_epoch) + 1
            iter_in_epoch = (self.trainer.iter % self.iters_per_epoch) + 1
            
            # Print epoch info on its own line for clarity
            epoch_str = f"\n📊 [Epoch {current_epoch}/{self.max_epochs}] Progress: {iter_in_epoch}/{self.iters_per_epoch} iterations"
            print(epoch_str)


class Mask2FormerTrainer(DefaultTrainer):
    """Trainer class for Mask2Former with epoch tracking."""

    def __init__(self, cfg, custom_cfg=None):
        """Initialize trainer with optional epoch tracking."""
        self.custom_cfg = custom_cfg
        
        # Set up epoch tracking BEFORE calling super().__init__
        if custom_cfg and custom_cfg.get('TRAINING', {}).get('MODE') == 'epochs':
            self.epochs_mode = True
            self.images_per_epoch = custom_cfg['TRAINING']['IMAGES_PER_EPOCH']
            self.batch_size = custom_cfg['TRAINING']['BATCH_SIZE']
            self.iters_per_epoch = self.images_per_epoch // self.batch_size
            self.max_epochs = custom_cfg['TRAINING']['MAX_EPOCHS']
        else:
            self.epochs_mode = False
        
        # Now call super().__init__ which will call build_hooks
        super().__init__(cfg)
    
    def build_hooks(self):
        """Build hooks, adding epoch logger if in epoch mode."""
        ret = super().build_hooks()
        if self.epochs_mode:
            ret.insert(-1, EpochPrintHook(self.max_epochs, self.iters_per_epoch))
        return ret
    
    def run_step(self):
        """Override to add epoch logging."""
        super().run_step()
        
        # Add epoch info to logs if in epoch mode
        if self.epochs_mode:
            current_epoch = (self.iter // self.iters_per_epoch) + 1
            iter_in_epoch = self.iter % self.iters_per_epoch
            
            # Add epoch metrics to storage for logging
            storage = self.storage
            storage.put_scalar("train/epoch", current_epoch, smoothing_hint=False)
            storage.put_scalar("train/epoch_progress", 
                             (iter_in_epoch / self.iters_per_epoch) * 100, 
                             smoothing_hint=False)

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


def setup_cfg(custom_cfg: dict, args=None):
    """Create configs from custom YAML configuration."""
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    
    # Load model configuration
    model_name = custom_cfg['MODEL']['NAME']
    model_info = MODELS[model_name]
    cfg.merge_from_file(model_info["config"])
    
    # Set model weights
    if custom_cfg['MODEL']['WEIGHTS']:
        cfg.MODEL.WEIGHTS = custom_cfg['MODEL']['WEIGHTS']
    else:
        cfg.MODEL.WEIGHTS = model_info["weights"]
        # Check for local model
        local_model = Path(f"models/maskformer2_{model_name}.pkl")
        if local_model.exists():
            print(f"✓ Using local model: {local_model}")
            cfg.MODEL.WEIGHTS = str(local_model)
    
    # Dataset configuration
    cfg.DATASETS.TRAIN = (custom_cfg['DATASET']['TRAIN_NAME'],)
    val_json = Path(custom_cfg['DATASET']['VAL_JSON'])
    cfg.DATASETS.TEST = (custom_cfg['DATASET']['VAL_NAME'],) if val_json.exists() else ()
    
    # Set number of classes
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = custom_cfg['MODEL']['NUM_CLASSES']
    
    # Training parameters
    cfg.SOLVER.IMS_PER_BATCH = custom_cfg['TRAINING']['BATCH_SIZE']
    cfg.SOLVER.BASE_LR = custom_cfg['OPTIMIZER']['BASE_LR']
    
    # Configure based on training mode
    if custom_cfg['TRAINING']['MODE'] == 'epochs':
        # Epoch-based configuration
        images_per_epoch = custom_cfg['TRAINING']['IMAGES_PER_EPOCH']
        batch_size = custom_cfg['TRAINING']['BATCH_SIZE']
        iters_per_epoch = images_per_epoch // batch_size
        
        cfg.SOLVER.MAX_ITER = custom_cfg['TRAINING']['MAX_EPOCHS'] * iters_per_epoch
        
        # Convert epoch steps to iterations
        lr_epochs = custom_cfg['OPTIMIZER']['LR_DECAY_EPOCHS']
        cfg.SOLVER.STEPS = tuple(e * iters_per_epoch for e in lr_epochs)
        
        # Convert evaluation periods
        cfg.SOLVER.CHECKPOINT_PERIOD = custom_cfg['EVALUATION']['CHECKPOINT_PERIOD_EPOCHS'] * iters_per_epoch
        cfg.TEST.EVAL_PERIOD = custom_cfg['EVALUATION']['EVAL_PERIOD_EPOCHS'] * iters_per_epoch
    else:
        # Iteration-based configuration
        cfg.SOLVER.MAX_ITER = custom_cfg['TRAINING']['MAX_ITERATIONS']
        cfg.SOLVER.STEPS = tuple(custom_cfg['OPTIMIZER']['LR_DECAY_STEPS'])
        cfg.SOLVER.CHECKPOINT_PERIOD = custom_cfg['EVALUATION']['CHECKPOINT_PERIOD_ITERS']
        cfg.TEST.EVAL_PERIOD = custom_cfg['EVALUATION']['EVAL_PERIOD_ITERS']
    
    # Optimizer settings
    cfg.SOLVER.WEIGHT_DECAY = custom_cfg['OPTIMIZER']['WEIGHT_DECAY']
    cfg.SOLVER.WEIGHT_DECAY_NORM = custom_cfg['OPTIMIZER']['WEIGHT_DECAY_NORM']
    cfg.SOLVER.WEIGHT_DECAY_EMBED = custom_cfg['OPTIMIZER']['WEIGHT_DECAY_EMBED']
    cfg.SOLVER.BACKBONE_MULTIPLIER = custom_cfg['OPTIMIZER']['BACKBONE_MULTIPLIER']
    cfg.SOLVER.GAMMA = custom_cfg['OPTIMIZER']['GAMMA']
    cfg.SOLVER.OPTIMIZER = custom_cfg['OPTIMIZER']['TYPE']
    cfg.SOLVER.WARMUP_ITERS = custom_cfg['OPTIMIZER']['WARMUP_ITERS']
    
    # Gradient clipping
    cfg.SOLVER.CLIP_GRADIENTS.ENABLED = custom_cfg['OPTIMIZER']['CLIP_GRADIENTS']
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = custom_cfg['OPTIMIZER']['CLIP_VALUE']
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "full_model"
    
    # Input configuration
    cfg.INPUT.MIN_SIZE_TRAIN = custom_cfg['INPUT']['MIN_SIZE_TRAIN']
    cfg.INPUT.MIN_SIZE_TEST = custom_cfg['INPUT']['MIN_SIZE_TEST']
    cfg.INPUT.MAX_SIZE_TRAIN = custom_cfg['INPUT']['MAX_SIZE_TRAIN']
    cfg.INPUT.MAX_SIZE_TEST = custom_cfg['INPUT']['MAX_SIZE_TEST']
    
    # LSJ augmentation
    if custom_cfg['INPUT']['USE_LSJ']:
        cfg.INPUT.DATASET_MAPPER_NAME = "coco_instance_lsj"
        cfg.INPUT.IMAGE_SIZE = custom_cfg['INPUT']['IMAGE_SIZE']
        cfg.INPUT.MIN_SCALE = custom_cfg['INPUT']['LSJ_MIN_SCALE']
        cfg.INPUT.MAX_SCALE = custom_cfg['INPUT']['LSJ_MAX_SCALE']
    
    cfg.INPUT.MASK_FORMAT = "bitmask"
    
    # Output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = custom_cfg['OUTPUT']['DIR'].format(model_name=model_name, timestamp=timestamp)
    cfg.OUTPUT_DIR = output_dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    # Performance settings
    cfg.SOLVER.AMP.ENABLED = custom_cfg['PERFORMANCE']['USE_AMP']
    cfg.CUDNN_BENCHMARK = custom_cfg['PERFORMANCE']['CUDNN_BENCHMARK']
    
    # DataLoader
    cfg.DATALOADER.NUM_WORKERS = custom_cfg['TRAINING']['NUM_WORKERS']
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = True
    
    # Advanced Mask2Former settings
    cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES = custom_cfg['ADVANCED']['NUM_QUERIES']
    cfg.MODEL.MASK_FORMER.DEC_LAYERS = custom_cfg['ADVANCED']['DEC_LAYERS']
    cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD = custom_cfg['ADVANCED']['DIM_FEEDFORWARD']
    cfg.MODEL.MASK_FORMER.HIDDEN_DIM = custom_cfg['ADVANCED']['HIDDEN_DIM']
    cfg.MODEL.MASK_FORMER.DROPOUT = custom_cfg['ADVANCED']['DROPOUT']
    cfg.MODEL.MASK_FORMER.NHEADS = custom_cfg['ADVANCED']['NHEADS']
    cfg.MODEL.MASK_FORMER.CLASS_WEIGHT = custom_cfg['ADVANCED']['CLASS_WEIGHT']
    cfg.MODEL.MASK_FORMER.DICE_WEIGHT = custom_cfg['ADVANCED']['DICE_WEIGHT']
    cfg.MODEL.MASK_FORMER.MASK_WEIGHT = custom_cfg['ADVANCED']['MASK_WEIGHT']
    cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT = custom_cfg['ADVANCED']['NO_OBJECT_WEIGHT']
    cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS = custom_cfg['ADVANCED']['TRAIN_NUM_POINTS']
    cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO = custom_cfg['ADVANCED']['OVERSAMPLE_RATIO']
    cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO = custom_cfg['ADVANCED']['IMPORTANCE_SAMPLE_RATIO']
    
    # Seed
    if custom_cfg['ADVANCED']['SEED'] != -1:
        cfg.SEED = custom_cfg['ADVANCED']['SEED']
    
    # Visualization
    cfg.VIS_PERIOD = custom_cfg['LOGGING']['VIS_PERIOD']
    
    if args:
        cfg.merge_from_list(args.opts)
    
    cfg.freeze()
    return cfg


def main(args):
    """Main training function."""
    # Load custom configuration
    config_path = args.custom_config if hasattr(args, 'custom_config') else "configs/custom_training_config.yaml"
    if not Path(config_path).exists():
        print(f"❌ Config file not found: {config_path}")
        print("   Please specify with --custom-config or create configs/custom_training_config.yaml")
        return
    
    print(f"📋 Loading configuration from: {config_path}")
    custom_cfg = load_custom_config(config_path)
    
    print("\n" + "="*70)
    print(f"MASK2FORMER TRAINING - {custom_cfg['MODEL']['NAME'].upper()}")
    print("="*70)
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("⚠ No GPU detected")
    
    # Register datasets
    print("\n📊 Registering datasets...")
    train_ok, val_ok = register_datasets(custom_cfg)
    
    if not train_ok:
        print("❌ Training dataset not found!")
        return
    
    # Setup configuration
    print(f"\n⚙️ Setting up configuration...")
    cfg = setup_cfg(custom_cfg, args)
    
    # Save custom config to output directory
    output_config_path = Path(cfg.OUTPUT_DIR) / "custom_config.yaml"
    with open(output_config_path, 'w') as f:
        yaml.dump(custom_cfg, f, default_flow_style=False)
    print(f"✓ Saved configuration to: {output_config_path}")
    
    print("\n📋 Training Configuration:")
    print(f"  Model: {custom_cfg['MODEL']['NAME']}")
    print(f"  Classes: {custom_cfg['MODEL']['NUM_CLASSES']}")
    print(f"  Batch size: {custom_cfg['TRAINING']['BATCH_SIZE']}")
    print(f"  Learning rate: {custom_cfg['OPTIMIZER']['BASE_LR']}")
    print(f"  Training mode: {custom_cfg['TRAINING']['MODE']}")
    
    if custom_cfg['TRAINING']['MODE'] == 'epochs':
        iters_per_epoch = custom_cfg['TRAINING']['IMAGES_PER_EPOCH'] // custom_cfg['TRAINING']['BATCH_SIZE']
        total_iters = custom_cfg['TRAINING']['MAX_EPOCHS'] * iters_per_epoch
        print(f"  Max epochs: {custom_cfg['TRAINING']['MAX_EPOCHS']}")
        print(f"  Iterations per epoch: {iters_per_epoch}")
        print(f"  Total iterations: {total_iters}")
        print(f"  Eval every: {custom_cfg['EVALUATION']['EVAL_PERIOD_EPOCHS']} epochs ({custom_cfg['EVALUATION']['EVAL_PERIOD_EPOCHS'] * iters_per_epoch} iters)")
    else:
        print(f"  Max iterations: {custom_cfg['TRAINING']['MAX_ITERATIONS']}")
        print(f"  Eval every: {custom_cfg['EVALUATION']['EVAL_PERIOD_ITERS']} iterations")
    
    print(f"  Output: {cfg.OUTPUT_DIR}")
    
    # Setup logger
    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="mask2former")
    
    # Build trainer
    print("\n🚀 Starting training...")
    trainer = Mask2FormerTrainer(cfg, custom_cfg)
    trainer.resume_or_load(resume=args.resume)
    
    return trainer.train()


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--custom-config", default="configs/custom_training_config.yaml",
                       help="path to custom YAML config file")
    args = parser.parse_args()
    
    print("Launching Mask2Former training with custom configuration...")
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )