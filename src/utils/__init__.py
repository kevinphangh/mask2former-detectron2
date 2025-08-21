"""Utility functions for Mask2Former training."""

from .dataset import register_datasets, clear_dataset_catalog

__all__ = ["register_datasets", "clear_dataset_catalog"]