# Mask2Former for Instance Segmentation

Production-ready Mask2Former training pipeline using Detectron2. Successfully trains transformer-based universal segmentation models with custom COCO datasets.

## Requirements

- Linux/WSL2 with NVIDIA GPU
- Python 3.8+, CUDA 11.1+
- gcc-11 (for CUDA 12 compatibility)

## Quick Start

### 1. Setup Environment

```bash
# Clone and setup
git clone https://github.com/kevinphangh/mask2former-detectron2.git
cd mask2former-detectron2
chmod +x setup.sh
./setup.sh
```

If manual setup needed:
```bash
# Create conda environment
conda create -n mask2former python=3.10 -y
conda activate mask2former

# Install PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install Detectron2
pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Compile CUDA kernels (critical)
cd Mask2Former/mask2former/modeling/pixel_decoder/ops
export CC=/usr/bin/gcc-11 CXX=/usr/bin/g++-11
python setup.py build install
```

### 2. Train

```bash
python train.py
```

Default configuration (1000 iterations, batch size 2) works well for small datasets. Model saves to `outputs/training_results/`.

## Dataset Format

COCO format with polygon segmentation masks:
```
data/
├── train/
│   ├── *.jpg
│   └── _annotations.coco.json
└── valid/
    ├── *.jpg
    └── _annotations.coco.json
```

## Configuration

Key parameters in `train.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NUM_CLASSES` | 2 | Number of object classes |
| `IMS_PER_BATCH` | 2 | Batch size (GPU memory dependent) |
| `BASE_LR` | 0.00025 | Learning rate |
| `MAX_ITER` | 1000 | Training iterations |

## Architecture

Mask2Former uses:
- **Backbone**: ResNet-50 or Swin Transformer
- **Pixel Decoder**: Multi-scale deformable attention
- **Transformer Decoder**: Masked attention with object queries
- **Output**: Per-query class and mask predictions

## Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA kernel compilation fails | Install gcc-11, set `CC=/usr/bin/gcc-11` |
| Gradient clipping error | Use `CLIP_TYPE: "norm"` not `"full_model"` |
| Missing masks error | Ensure polygon masks in COCO annotations |
| Out of memory | Reduce batch size to 1 |

## Inference

```python
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

cfg = get_cfg()
cfg.merge_from_file("outputs/training_results/config.yaml")
cfg.MODEL.WEIGHTS = "outputs/training_results/model_final.pth"

predictor = DefaultPredictor(cfg)
outputs = predictor(image)
```

## Project Structure

```
mask2former-detectron2/
├── train.py                # Training script
├── setup.sh                # Setup script
├── Mask2Former/            # Mask2Former implementation
├── configs/                # Model configs
├── data/                   # Dataset
└── outputs/                # Training results
```

## References

- [Mask2Former Paper](https://arxiv.org/abs/2112.01527)
- [Official Repository](https://github.com/facebookresearch/Mask2Former)

## License

Apache 2.0 (Detectron2) and MIT (Mask2Former)