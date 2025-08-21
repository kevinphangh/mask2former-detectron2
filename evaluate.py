#!/usr/bin/env python3
"""
Evaluation script for trained Mask R-CNN model.
Computes COCO metrics on validation/test datasets.
"""

import argparse
import os
import torch
from pathlib import Path
import json

from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2 import model_zoo


def register_dataset(name, json_file, image_root):
    """Register a COCO format dataset."""
    if name not in MetadataCatalog.list():
        register_coco_instances(name, {}, json_file, image_root)
        print(f"✓ Registered dataset: {name}")
    return name


def setup_cfg(args):
    """Setup configuration for evaluation."""
    cfg = get_cfg()
    
    # Load configuration
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    else:
        # Default Mask R-CNN configuration
        cfg.merge_from_file(model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        ))
    
    # Set model weights and classes
    cfg.MODEL.WEIGHTS = args.weights
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = args.num_classes
    
    # Set device
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    
    cfg.freeze()
    return cfg


def evaluate_dataset(predictor, dataset_name, cfg, output_dir):
    """
    Evaluate model on a dataset.
    
    Args:
        predictor: Trained model predictor
        dataset_name: Name of registered dataset
        cfg: Configuration
        output_dir: Directory to save results
        
    Returns:
        results: Dictionary of evaluation metrics
    """
    print(f"\n📊 Evaluating on {dataset_name}...")
    
    # Create evaluator
    evaluator = COCOEvaluator(
        dataset_name, 
        output_dir=output_dir,
        use_fast_impl=True
    )
    
    # Build data loader
    val_loader = build_detection_test_loader(cfg, dataset_name)
    
    # Run evaluation
    results = inference_on_dataset(predictor.model, val_loader, evaluator)
    
    return results


def print_results(results, dataset_name):
    """Print evaluation results in a formatted way."""
    print(f"\n📈 Results for {dataset_name}:")
    print("="*50)
    
    # Print bbox results if available
    if "bbox" in results:
        bbox_results = results["bbox"]
        print("\n🔲 Bounding Box Detection:")
        print(f"  • mAP @ IoU=0.50:0.95: {bbox_results.get('AP', 0):.2f}%")
        print(f"  • mAP @ IoU=0.50:     {bbox_results.get('AP50', 0):.2f}%")
        print(f"  • mAP @ IoU=0.75:     {bbox_results.get('AP75', 0):.2f}%")
        print(f"  • mAP (small):        {bbox_results.get('APs', 0):.2f}%")
        print(f"  • mAP (medium):       {bbox_results.get('APm', 0):.2f}%")
        print(f"  • mAP (large):        {bbox_results.get('APl', 0):.2f}%")
    
    # Print segmentation results if available
    if "segm" in results:
        segm_results = results["segm"]
        print("\n🎭 Instance Segmentation:")
        print(f"  • mAP @ IoU=0.50:0.95: {segm_results.get('AP', 0):.2f}%")
        print(f"  • mAP @ IoU=0.50:     {segm_results.get('AP50', 0):.2f}%")
        print(f"  • mAP @ IoU=0.75:     {segm_results.get('AP75', 0):.2f}%")
        print(f"  • mAP (small):        {segm_results.get('APs', 0):.2f}%")
        print(f"  • mAP (medium):       {segm_results.get('APm', 0):.2f}%")
        print(f"  • mAP (large):        {segm_results.get('APl', 0):.2f}%")
    
    print("="*50)


def save_results(results, output_file):
    """Save evaluation results to JSON file."""
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to: {output_file}")


def main(args):
    """Main evaluation function."""
    print("\n" + "="*60)
    print("MASK R-CNN EVALUATION WITH DETECTRON2")
    print("="*60)
    
    # Check if weights exist
    if not Path(args.weights).exists():
        raise FileNotFoundError(f"Model weights not found: {args.weights}")
    
    # Setup configuration
    print("\n⚙️ Setting up configuration...")
    cfg = setup_cfg(args)
    
    # Register datasets
    datasets_to_eval = []
    
    if args.dataset_name:
        # Use provided dataset name
        datasets_to_eval.append(args.dataset_name)
    else:
        # Register datasets from files
        data_dir = Path(args.data_dir)
        
        # Validation dataset
        val_json = data_dir / "valid" / "_annotations.coco.json"
        if val_json.exists():
            name = register_dataset(
                "cylinders_val",
                str(val_json),
                str(data_dir / "valid")
            )
            datasets_to_eval.append(name)
        
        # Test dataset
        test_json = data_dir / "test" / "_annotations.coco.json"
        if test_json.exists():
            name = register_dataset(
                "cylinders_test",
                str(test_json),
                str(data_dir / "test")
            )
            datasets_to_eval.append(name)
    
    if not datasets_to_eval:
        print("⚠️ No datasets found for evaluation")
        return
    
    # Create predictor
    print(f"\n📦 Loading model from: {args.weights}")
    predictor = DefaultPredictor(cfg)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Evaluate on each dataset
    all_results = {}
    for dataset_name in datasets_to_eval:
        results = evaluate_dataset(
            predictor, 
            dataset_name, 
            cfg, 
            str(output_dir)
        )
        all_results[dataset_name] = results
        print_results(results, dataset_name)
    
    # Save all results
    if args.save_json:
        output_file = output_dir / "evaluation_results.json"
        save_results(all_results, output_file)
    
    print("\n✅ Evaluation complete!")
    print("="*60)


def get_parser():
    """Get argument parser."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained Mask R-CNN model on COCO format datasets"
    )
    
    # Model options
    parser.add_argument(
        "--weights",
        required=True,
        help="Path to model weights (.pth file)"
    )
    parser.add_argument(
        "--config-file",
        default="",
        help="Path to config file (optional)"
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=1,
        help="Number of classes the model was trained on (default: 1)"
    )
    
    # Dataset options
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Path to dataset directory (default: data)"
    )
    parser.add_argument(
        "--dataset-name",
        help="Name of specific dataset to evaluate (optional)"
    )
    
    # Output options
    parser.add_argument(
        "--output-dir",
        default="evaluation_results",
        help="Directory to save evaluation results (default: evaluation_results)"
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="Save detailed results as JSON"
    )
    
    # System options
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU evaluation even if GPU is available"
    )
    
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)