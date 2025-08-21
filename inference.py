#!/usr/bin/env python3
"""
Inference script for trained Mask R-CNN model.
Performs instance segmentation on images and saves visualizations.
"""

import argparse
import cv2
import os
from pathlib import Path
import torch
import json

from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2 import model_zoo


def setup_cfg(args):
    """Setup configuration for inference."""
    cfg = get_cfg()
    
    # Load configuration
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    else:
        # Default Mask R-CNN configuration
        cfg.merge_from_file(model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        ))
    
    # Set model weights
    cfg.MODEL.WEIGHTS = args.weights
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = args.num_classes
    
    # Set score threshold for predictions
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    
    cfg.freeze()
    return cfg


def register_dataset_metadata(num_classes):
    """Register metadata for visualization."""
    if num_classes == 1:
        class_names = ["cylinder"]
        class_colors = [(0, 255, 0)]
    elif num_classes == 2:
        class_names = ["low_diff", "pickable_surface"]
        class_colors = [(255, 0, 0), (0, 255, 0)]
    else:
        class_names = [f"class_{i}" for i in range(num_classes)]
        class_colors = None
    
    # Register metadata for visualization
    metadata = MetadataCatalog.get("inference_dataset")
    metadata.thing_classes = class_names
    if class_colors:
        metadata.thing_colors = class_colors
    
    return metadata


def predict_image(predictor, image_path, metadata, args):
    """
    Run prediction on a single image.
    
    Args:
        predictor: Detectron2 predictor
        image_path: Path to input image
        metadata: Dataset metadata for visualization
        args: Command line arguments
        
    Returns:
        outputs: Model predictions
        vis_output: Visualization output
    """
    # Read image
    im = cv2.imread(str(image_path))
    if im is None:
        print(f"⚠️ Could not read image: {image_path}")
        return None, None
    
    # Run prediction
    outputs = predictor(im)
    
    # Create visualization
    v = Visualizer(
        im[:, :, ::-1],
        metadata=metadata,
        scale=args.vis_scale,
        instance_mode=ColorMode.IMAGE if args.overlay else ColorMode.SEGMENTATION
    )
    
    instances = outputs["instances"].to("cpu")
    vis_output = v.draw_instance_predictions(instances)
    
    return outputs, vis_output


def save_results(image_path, outputs, vis_output, output_dir, save_json=False):
    """
    Save prediction results.
    
    Args:
        image_path: Path to input image
        outputs: Model predictions
        vis_output: Visualization output
        output_dir: Directory to save results
        save_json: Whether to save predictions as JSON
    """
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save visualization
    vis_path = output_dir / f"{image_path.stem}_prediction.jpg"
    vis_image = vis_output.get_image()
    cv2.imwrite(str(vis_path), vis_image[:, :, ::-1])
    print(f"  ✓ Saved visualization: {vis_path}")
    
    # Save predictions as JSON if requested
    if save_json:
        instances = outputs["instances"].to("cpu")
        predictions = {
            "image": str(image_path),
            "num_instances": len(instances),
            "predictions": []
        }
        
        for i in range(len(instances)):
            pred = {
                "score": float(instances.scores[i]),
                "class": int(instances.pred_classes[i]),
                "bbox": instances.pred_boxes[i].tensor.numpy().tolist()[0] if instances.has("pred_boxes") else None,
            }
            predictions["predictions"].append(pred)
        
        json_path = output_dir / f"{image_path.stem}_predictions.json"
        with open(json_path, "w") as f:
            json.dump(predictions, f, indent=2)
        print(f"  ✓ Saved predictions: {json_path}")


def main(args):
    """Main inference function."""
    print("\n" + "="*60)
    print("MASK R-CNN INFERENCE WITH DETECTRON2")
    print("="*60)
    
    # Check if weights exist
    if not Path(args.weights).exists():
        raise FileNotFoundError(f"Model weights not found: {args.weights}")
    
    # Setup configuration
    print("\n⚙️ Setting up configuration...")
    cfg = setup_cfg(args)
    
    # Register metadata for visualization
    metadata = register_dataset_metadata(args.num_classes)
    
    # Create predictor
    print(f"📦 Loading model from: {args.weights}")
    predictor = DefaultPredictor(cfg)
    
    # Get list of images to process
    image_paths = []
    if args.input_image:
        image_paths = [Path(args.input_image)]
    elif args.input_dir:
        input_dir = Path(args.input_dir)
        image_paths = list(input_dir.glob("*.jpg")) + \
                     list(input_dir.glob("*.png")) + \
                     list(input_dir.glob("*.jpeg"))
        print(f"📁 Found {len(image_paths)} images in {input_dir}")
    
    if not image_paths:
        print("⚠️ No images found to process")
        return
    
    # Process images
    print(f"\n🔍 Processing {len(image_paths)} image(s)...")
    print(f"   Confidence threshold: {args.confidence_threshold}")
    print(f"   Output directory: {args.output_dir}")
    print()
    
    for image_path in image_paths:
        print(f"Processing: {image_path.name}")
        
        # Run prediction
        outputs, vis_output = predict_image(predictor, image_path, metadata, args)
        
        if outputs is None:
            continue
        
        # Print statistics
        instances = outputs["instances"]
        num_instances = len(instances)
        print(f"  Found {num_instances} instance(s)")
        
        if num_instances > 0:
            scores = instances.scores.cpu().numpy()
            print(f"  Confidence scores: {scores}")
        
        # Save results
        save_results(
            image_path, 
            outputs, 
            vis_output, 
            args.output_dir,
            save_json=args.save_json
        )
        print()
    
    print("✅ Inference complete!")
    print("="*60)


def get_parser():
    """Get argument parser."""
    parser = argparse.ArgumentParser(
        description="Run inference with trained Mask R-CNN model"
    )
    
    # Input options
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--input-image",
        help="Path to single input image"
    )
    group.add_argument(
        "--input-dir",
        help="Directory containing images to process"
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
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum confidence threshold for predictions (default: 0.5)"
    )
    
    # Output options
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory to save results (default: results)"
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="Save predictions as JSON files"
    )
    
    # Visualization options
    parser.add_argument(
        "--vis-scale",
        type=float,
        default=1.0,
        help="Visualization scale factor (default: 1.0)"
    )
    parser.add_argument(
        "--overlay",
        action="store_true",
        help="Overlay predictions on original image instead of segmentation view"
    )
    
    # System options
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU inference even if GPU is available"
    )
    
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)