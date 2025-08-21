"""Dataset registration and management utilities."""

from pathlib import Path
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances


def clear_dataset_catalog(dataset_names):
    """
    Clear existing dataset registrations from catalog.
    
    Args:
        dataset_names: List of dataset names to clear
    """
    for name in dataset_names:
        if name in DatasetCatalog.list():
            DatasetCatalog.remove(name)
            MetadataCatalog.remove(name)


def register_datasets(data_dir="data", train_name="custom_train", val_name="custom_val"):
    """
    Register COCO format datasets for training and validation.
    
    Args:
        data_dir: Root directory containing train/valid subdirectories
        train_name: Name for training dataset registration
        val_name: Name for validation dataset registration
    
    Returns:
        Tuple of (train_registered, val_registered) booleans
    """
    data_path = Path(data_dir)
    train_registered = False
    val_registered = False
    
    # Clear existing registrations
    clear_dataset_catalog([train_name, val_name])
    
    # Register training dataset
    train_json = data_path / "train" / "_annotations.coco.json"
    if train_json.exists():
        register_coco_instances(
            train_name, 
            {}, 
            str(train_json),
            str(data_path / "train")
        )
        train_registered = True
        print(f"✓ Registered training dataset: {train_name}")
    else:
        print(f"⚠ Training annotations not found: {train_json}")
    
    # Register validation dataset
    val_json = data_path / "valid" / "_annotations.coco.json"
    if val_json.exists():
        register_coco_instances(
            val_name,
            {},
            str(val_json),
            str(data_path / "valid")
        )
        val_registered = True
        print(f"✓ Registered validation dataset: {val_name}")
    else:
        print(f"⚠ Validation annotations not found: {val_json}")
    
    return train_registered, val_registered