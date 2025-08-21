"""Data loading utilities for Mask2Former training."""

import torch
from detectron2.data import DatasetMapper, build_detection_train_loader
import detectron2.data.transforms as T


def custom_mapper_with_masks(dataset_dict, mapper):
    """
    Wrapper to ensure masks are in the correct format for Mask2Former.
    
    Converts polygon masks to tensor format required by Mask2Former.
    
    Args:
        dataset_dict: Dictionary containing dataset sample
        mapper: Base dataset mapper
    
    Returns:
        Modified dataset dictionary with tensor masks
    """
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


def build_data_loader(cfg):
    """
    Build train loader with proper mask loading for Mask2Former.
    
    Args:
        cfg: Detectron2 configuration
    
    Returns:
        DataLoader for training
    """
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