"""
Dataset registration and utilities for COCO format datasets.
"""

import os
import json
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer
import cv2
import random
import matplotlib.pyplot as plt


def register_cylinder_datasets(data_dir="data"):
    """
    Register COCO format datasets for cylinder detection.
    
    Args:
        data_dir: Path to directory containing train/valid/test subdirectories
    """
    # Define dataset names and paths
    datasets = {
        "cylinders_train": {
            "json_file": os.path.join(data_dir, "train", "_annotations.coco.json"),
            "image_root": os.path.join(data_dir, "train")
        },
        "cylinders_val": {
            "json_file": os.path.join(data_dir, "valid", "_annotations.coco.json"),
            "image_root": os.path.join(data_dir, "valid")
        },
        "cylinders_test": {
            "json_file": os.path.join(data_dir, "test", "_annotations.coco.json"),
            "image_root": os.path.join(data_dir, "test")
        }
    }
    
    # Register each dataset
    for name, info in datasets.items():
        if os.path.exists(info["json_file"]):
            register_coco_instances(
                name, 
                {}, 
                info["json_file"], 
                info["image_root"]
            )
            print(f"✓ Registered dataset: {name}")
            
            # Get and print dataset statistics
            dataset_dicts = DatasetCatalog.get(name)
            print(f"  - Images: {len(dataset_dicts)}")
            
            # Count total annotations
            total_annotations = sum(len(d.get("annotations", [])) for d in dataset_dicts)
            print(f"  - Annotations: {total_annotations}")
        else:
            print(f"✗ Dataset not found: {name} (missing {info['json_file']})")
    
    # Set metadata for better visualization
    for name in datasets.keys():
        if name in DatasetCatalog.list():
            MetadataCatalog.get(name).thing_classes = ["pickable_surface"]
            MetadataCatalog.get(name).thing_colors = [(0, 255, 0)]  # Green for pickable_surface


def visualize_dataset_samples(dataset_name, num_samples=3, save_dir="outputs/visualizations"):
    """
    Visualize random samples from a registered dataset.
    
    Args:
        dataset_name: Name of registered dataset
        num_samples: Number of samples to visualize
        save_dir: Directory to save visualizations
    """
    os.makedirs(save_dir, exist_ok=True)
    
    dataset_dicts = DatasetCatalog.get(dataset_name)
    metadata = MetadataCatalog.get(dataset_name)
    
    # Randomly sample images
    samples = random.sample(dataset_dicts, min(num_samples, len(dataset_dicts)))
    
    for idx, d in enumerate(samples):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.8)
        
        # Draw annotations
        out = visualizer.draw_dataset_dict(d)
        
        # Save visualization
        output_path = os.path.join(save_dir, f"{dataset_name}_sample_{idx}.jpg")
        cv2.imwrite(output_path, out.get_image()[:, :, ::-1])
        print(f"Saved visualization: {output_path}")
        
        # Display if in notebook/interactive environment
        try:
            plt.figure(figsize=(12, 8))
            plt.imshow(out.get_image())
            plt.axis('off')
            plt.title(f"{dataset_name} - Sample {idx}")
            plt.tight_layout()
            plt.savefig(output_path.replace('.jpg', '_plot.jpg'))
            plt.show()
        except:
            pass


def get_dataset_statistics(dataset_name):
    """
    Get detailed statistics about a dataset.
    
    Args:
        dataset_name: Name of registered dataset
        
    Returns:
        Dictionary with dataset statistics
    """
    dataset_dicts = DatasetCatalog.get(dataset_name)
    
    stats = {
        "num_images": len(dataset_dicts),
        "num_annotations": 0,
        "avg_annotations_per_image": 0,
        "min_annotations": float('inf'),
        "max_annotations": 0,
        "image_sizes": set(),
        "categories": set()
    }
    
    for d in dataset_dicts:
        annotations = d.get("annotations", [])
        num_anns = len(annotations)
        
        stats["num_annotations"] += num_anns
        stats["min_annotations"] = min(stats["min_annotations"], num_anns)
        stats["max_annotations"] = max(stats["max_annotations"], num_anns)
        
        # Image size
        stats["image_sizes"].add((d["height"], d["width"]))
        
        # Categories
        for ann in annotations:
            stats["categories"].add(ann.get("category_id", 0))
    
    if stats["num_images"] > 0:
        stats["avg_annotations_per_image"] = stats["num_annotations"] / stats["num_images"]
    
    # Convert sets to lists for JSON serialization
    stats["image_sizes"] = list(stats["image_sizes"])
    stats["categories"] = list(stats["categories"])
    
    return stats


def verify_dataset_registration():
    """
    Verify that datasets are properly registered and accessible.
    """
    print("\n" + "="*50)
    print("DATASET VERIFICATION")
    print("="*50)
    
    registered_datasets = DatasetCatalog.list()
    cylinder_datasets = [d for d in registered_datasets if "cylinders" in d]
    
    if not cylinder_datasets:
        print("❌ No cylinder datasets found!")
        print("Please run register_cylinder_datasets() first.")
        return False
    
    print(f"Found {len(cylinder_datasets)} cylinder datasets:")
    for dataset_name in cylinder_datasets:
        print(f"\n📊 {dataset_name}:")
        try:
            stats = get_dataset_statistics(dataset_name)
            print(f"  - Images: {stats['num_images']}")
            print(f"  - Total annotations: {stats['num_annotations']}")
            print(f"  - Avg annotations/image: {stats['avg_annotations_per_image']:.2f}")
            print(f"  - Annotation range: [{stats['min_annotations']}, {stats['max_annotations']}]")
            print(f"  - Image sizes: {stats['image_sizes']}")
            print(f"  - Categories: {stats['categories']}")
        except Exception as e:
            print(f"  ❌ Error accessing dataset: {e}")
            return False
    
    print("\n✅ All datasets verified successfully!")
    return True


if __name__ == "__main__":
    # Register datasets
    register_cylinder_datasets()
    
    # Verify registration
    if verify_dataset_registration():
        # Visualize some samples
        print("\nGenerating visualizations...")
        visualize_dataset_samples("cylinders_train", num_samples=3)