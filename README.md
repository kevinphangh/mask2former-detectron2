# Mask2Former Training - Simplified

Simplified Mask2Former training pipeline for custom COCO-format datasets.

## Quick Start

### 1. Setup Environment
```bash
# Recommended: Create a virtual environment first
conda create -n mask2former python=3.10
conda activate mask2former

# Run setup script
bash scripts/setup.sh
```

### 2. Prepare Your Dataset
Place your COCO-format dataset in:
- `data/train/` - Training images and `_annotations.coco.json`
- `data/valid/` - Validation images and `_annotations.coco.json`
- `data/test/` - Test images and `_annotations.coco.json`

### 3. Train the Model
```bash
python train.py
```

## Configuration

Edit `train.py` to configure training:

```python
# Model selection (line 40)
MODEL_NAME = "swin_tiny"  # Options: swin_tiny, swin_small, swin_base

# Dataset classes (line 196)
cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 2  # Your number of classes

# Training parameters (lines 199-201)
cfg.SOLVER.IMS_PER_BATCH = 1      # Batch size
cfg.SOLVER.BASE_LR = 0.00005      # Learning rate
cfg.SOLVER.MAX_ITER = 3000        # Training iterations
```

### Model Requirements

| Model | GPU Memory | Performance |
|-------|------------|-------------|
| `swin_tiny` | 6GB | Good baseline |
| `swin_small` | 8GB | Better accuracy |
| `swin_base` | 16GB | Best accuracy |

## Training Output

Results are saved to `outputs/{model_name}/`:
- `model_*.pth` - Checkpoints every 500 iterations
- `model_final.pth` - Final trained model
- `metrics.json` - Training metrics
- `log.txt` - Training log

## Resume Training

To resume from the last checkpoint:
```bash
python train.py --resume
```

## Memory Optimization

For limited GPU memory, the script automatically uses:
- Batch size of 1
- Reduced image sizes (384-768px)
- Mixed precision training

## Evaluation Metrics

The model is evaluated every 500 iterations with COCO metrics:
- **AP**: Average Precision (main metric)
- **AP50**: AP at IoU=0.5
- **AP75**: AP at IoU=0.75

## Requirements

- GPU with 6GB+ VRAM
- CUDA 11.8 or 12.1
- Python 3.8+
- PyTorch 2.0+
- Detectron2

## Troubleshooting

### CUDA Out of Memory
- Switch to `swin_tiny` model
- Reduce batch size to 1
- Reduce image sizes in configuration

### Deprecation Warnings
The script automatically suppresses known deprecation warnings from PyTorch AMP and timm.

### CUDA Kernel Compilation Failed
Training will still work but may be slower. To fix:
```bash
# Install CUDA toolkit (Ubuntu/WSL)
sudo apt install nvidia-cuda-toolkit

# Re-run setup
bash scripts/setup.sh
```

## References

- [Mask2Former Paper](https://arxiv.org/abs/2112.01527)
- [Official Repository](https://github.com/facebookresearch/Mask2Former)