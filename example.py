#!/usr/bin/env python3
"""
Example script demonstrating the complete workflow:
1. Dataset registration
2. Training
3. Evaluation
4. Inference
"""

import os
from pathlib import Path
import torch
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.utils.visualizer import Visualizer
from detectron2 import model_zoo
import cv2


def register_datasets():
    """Register your COCO format datasets."""
    print("📊 Registering datasets...")
    
    # Register training dataset
    register_coco_instances(
        "cylinders_train", 
        {}, 
        "data/train/_annotations.coco.json",
        "data/train"
    )
    
    # Register validation dataset
    register_coco_instances(
        "cylinders_val",
        {},
        "data/valid/_annotations.coco.json",
        "data/valid"
    )
    
    print("✓ Datasets registered")


def setup_config():
    """Setup configuration for training."""
    cfg = get_cfg()
    
    # Load base Mask R-CNN config
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    ))
    
    # Set datasets
    cfg.DATASETS.TRAIN = ("cylinders_train",)
    cfg.DATASETS.TEST = ("cylinders_val",)
    
    # Model configuration
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Single class
    
    # Use pretrained COCO weights
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )
    
    # Training configuration
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 300  # Short for demo
    cfg.SOLVER.STEPS = (200,)
    
    # Output directory
    cfg.OUTPUT_DIR = "./demo_output"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    return cfg


def train_model(cfg):
    """Train the model."""
    print("\n🏋️ Training model...")
    
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    
    print("✓ Training complete")
    return trainer


def evaluate_model(cfg, trainer):
    """Evaluate the trained model."""
    print("\n📈 Evaluating model...")
    
    evaluator = COCOEvaluator("cylinders_val", output_dir=cfg.OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, "cylinders_val")
    results = inference_on_dataset(trainer.model, val_loader, evaluator)
    
    print(f"✓ mAP @ IoU=0.50: {results.get('segm', {}).get('AP50', 0):.1f}%")
    return results


def run_inference(cfg):
    """Run inference on a sample image."""
    print("\n🔍 Running inference...")
    
    # Create predictor
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    predictor = DefaultPredictor(cfg)
    
    # Find a sample image
    sample_images = list(Path("data/test").glob("*.jpg"))
    if not sample_images:
        print("⚠️ No test images found")
        return
    
    # Run prediction
    image_path = sample_images[0]
    im = cv2.imread(str(image_path))
    outputs = predictor(im)
    
    # Visualize
    v = Visualizer(im[:, :, ::-1], scale=1.0)
    vis = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    
    # Save result
    output_path = Path(cfg.OUTPUT_DIR) / "inference_result.jpg"
    cv2.imwrite(str(output_path), vis.get_image()[:, :, ::-1])
    
    print(f"✓ Result saved to {output_path}")
    print(f"  Found {len(outputs['instances'])} instances")


def main():
    """Main function demonstrating complete workflow."""
    print("="*60)
    print("MASK R-CNN COMPLETE WORKFLOW EXAMPLE")
    print("="*60)
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ No GPU detected")
    
    # 1. Register datasets
    register_datasets()
    
    # 2. Setup configuration
    cfg = setup_config()
    print(f"✓ Configuration ready")
    
    # 3. Train model
    trainer = train_model(cfg)
    
    # 4. Evaluate model
    results = evaluate_model(cfg, trainer)
    
    # 5. Run inference
    run_inference(cfg)
    
    print("\n" + "="*60)
    print("✅ WORKFLOW COMPLETE!")
    print("="*60)
    print(f"\nResults saved to: {cfg.OUTPUT_DIR}")
    print("\nNext steps:")
    print("  • Adjust training iterations for better results")
    print("  • Try different backbone architectures")
    print("  • Fine-tune hyperparameters")


if __name__ == "__main__":
    main()