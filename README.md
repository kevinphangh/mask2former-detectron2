# Mask2Former for Instance Segmentation

Production-ready Mask2Former training pipeline using Detectron2 for custom COCO datasets.

## 📁 Project Structure

```
mask2former-detectron2/
├── src/                     # Source code
│   ├── training/           # Training modules
│   └── utils/              # Utility functions
├── scripts/                 # Executable scripts
│   ├── train.py            # Basic training
│   ├── train_swin.py       # Transfer learning
│   └── setup.sh            # Environment setup
├── configs/                 # Configuration files
├── data/                    # Dataset directory
├── outputs/                 # Training outputs
└── Mask2Former/            # Mask2Former submodule
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
git clone https://github.com/kevinphangh/mask2former-detectron2.git
cd mask2former-detectron2
chmod +x scripts/setup.sh
./scripts/setup.sh
```

### 2. Prepare Dataset

COCO format required:
```
data/
├── train/
│   ├── *.jpg
│   └── _annotations.coco.json
└── valid/
    ├── *.jpg
    └── _annotations.coco.json
```

### 3. Train Model

**From scratch:**
```bash
python scripts/train.py
```

**Transfer learning (recommended):**
```bash
python scripts/train_swin.py
```

## 🎯 Transfer Learning

### Available Pre-trained Models

| Model | Parameters | Config File |
|-------|------------|-------------|
| **Swin-Tiny** | 28M | `maskformer2_swin_tiny_bs16_50ep.yaml` |
| **Swin-Small** | 50M | `maskformer2_swin_small_bs16_50ep.yaml` |
| **Swin-Base** | 88M | `maskformer2_swin_base_IN21k_384_bs16_50ep.yaml` |
| **ResNet-50** | 44M | `maskformer2_R50_bs16_50ep.yaml` |
| **ResNet-101** | 63M | `maskformer2_R101_bs16_50ep.yaml` |

### Pre-trained Weights URLs

```python
# Swin models
SWIN_TINY = "https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/instance/maskformer2_swin_tiny_bs16_50ep/model_final_86143f.pkl"
SWIN_SMALL = "https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/instance/maskformer2_swin_small_bs16_50ep/model_final_1e7f22.pkl"
SWIN_BASE = "https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/instance/maskformer2_swin_base_IN21k_384_bs16_50ep/model_final_83d103.pkl"

# ResNet models
RESNET50 = "https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/instance/maskformer2_R50_bs16_50ep/model_final_3c8ec9.pkl"
RESNET101 = "https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/instance/maskformer2_R101_bs16_50ep/model_final_eba159.pkl"
```

### Transfer Learning Configuration

```python
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from mask2former import add_maskformer2_config

cfg = get_cfg()
add_deeplab_config(cfg)
add_maskformer2_config(cfg)

# Load config
cfg.merge_from_file("Mask2Former/configs/coco/instance-segmentation/swin/maskformer2_swin_small_bs16_50ep.yaml")

# Pre-trained weights
cfg.MODEL.WEIGHTS = SWIN_SMALL  # URL from above

# Set your classes
cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 1  # Your number of classes

# Fine-tuning parameters
cfg.SOLVER.BASE_LR = 0.0001
cfg.SOLVER.MAX_ITER = 3000
cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1
```

### Key Parameters

| Parameter | From Scratch | Transfer Learning |
|-----------|--------------|-------------------|
| `BASE_LR` | 0.00025 | 0.0001 |
| `BACKBONE_MULTIPLIER` | 1.0 | 0.1 |
| `MAX_ITER` | 10000+ | 1000-3000 |
| `WARMUP_ITERS` | 1000 | 100 |

## ⚙️ Training Configuration

Edit parameters in `scripts/train.py` or `scripts/train_swin.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NUM_CLASSES` | 1 | Number of object classes |
| `IMS_PER_BATCH` | 2 | Batch size |
| `BASE_LR` | 0.0001 | Learning rate |
| `MAX_ITER` | 3000 | Training iterations |

## 📊 Output Structure

```
outputs/experiments/
├── model_final.pth         # Final model
├── model_*.pth             # Checkpoints
├── metrics.json            # Training metrics
└── events.out.tfevents.*  # TensorBoard logs
```

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA kernel compilation fails | Install gcc-11: `sudo apt install gcc-11 g++-11` |
| Out of memory | Reduce batch size or use smaller model |
| Missing masks | Ensure COCO annotations include segmentation polygons |
| High initial loss | Normal with transfer learning, decreases after warmup |

## 📚 Inference

```python
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

cfg = get_cfg()
cfg.merge_from_file("outputs/experiments/swin_small_transfer/config.yaml")
cfg.MODEL.WEIGHTS = "outputs/experiments/swin_small_transfer/model_final.pth"

predictor = DefaultPredictor(cfg)
outputs = predictor(image)
```

## 📝 License

Apache 2.0 (Detectron2) and MIT (Mask2Former)

## 🔗 References

- [Mask2Former Paper](https://arxiv.org/abs/2112.01527)
- [Official Repository](https://github.com/facebookresearch/Mask2Former)
- [Detectron2 Documentation](https://detectron2.readthedocs.io/)