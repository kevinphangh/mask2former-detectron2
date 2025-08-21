"""Mask2Former trainer implementation."""

from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from .data_loader import build_data_loader


class Mask2FormerTrainer(DefaultTrainer):
    """Custom trainer for Mask2Former with proper mask handling."""
    
    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        """Build COCO evaluator for validation."""
        return COCOEvaluator(dataset_name, output_dir=cfg.OUTPUT_DIR)
    
    @classmethod
    def build_train_loader(cls, cfg):
        """Build training data loader with mask format conversion."""
        return build_data_loader(cfg)