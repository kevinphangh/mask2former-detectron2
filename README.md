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

**Note:** The setup script will attempt to compile CUDA kernels for MSDeformAttn, which requires the CUDA toolkit (not just drivers). If compilation fails, the training will still work but may be slower. To install CUDA toolkit:

```bash
# Ubuntu/WSL
sudo apt update
sudo apt install nvidia-cuda-toolkit

# Then re-run setup
./scripts/setup.sh
```

### 2. Download Pre-trained Models (Optional)

The training scripts automatically download official Mask2Former pre-trained models during first run. For offline training, you can manually download them using the URLs from the Transfer Learning section below:

```bash
# Create models directory
mkdir -p models

# Download using wget with URLs from the Transfer Learning section
# Example for Swin-Small:
wget -O models/maskformer2_swin_small.pkl $SWIN_SMALL

# Then update scripts/train_swin.py line 41 to use local file:
cfg.MODEL.WEIGHTS = "models/maskformer2_swin_small.pkl"
```

### 3. Prepare Dataset

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

### 4. Train Model

**Quick Start (Recommended - Handles All Issues):**
```bash
# Use the wrapper script that fixes all environment issues
./train.sh

# This script automatically:
# - Fixes library compatibility issues
# - Uses PyTorch fallback if CUDA kernel fails
# - Handles all environment setup
```

**Direct Training (if environment is properly configured):**
```bash
# Uses official Facebook Research implementation
python scripts/train_mask2former.py

# Select model by editing MODEL_NAME in the script (line 88)
# Options: swin_tiny, swin_small, swin_base, resnet50, resnet101
```

**Alternative training scripts:**
```bash
python scripts/train.py         # Basic training
python scripts/train_swin.py     # Simplified transfer learning
```

**Note on CUDA Kernel:** The MSDeformAttn CUDA kernel provides 3-5x speedup. If compilation fails, the training automatically uses a PyTorch fallback that still runs on GPU but is slower. The `train.sh` script handles this automatically.

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

## ⚙️ Hyperparameter Configuration

### Where to Configure

1. **Training Scripts** (`scripts/train.py` or `scripts/train_swin.py`):
   - Primary location for hyperparameter tuning
   - Edit the `setup_cfg()` function
   - Changes take effect immediately

2. **Config Files** (`configs/mask2former/default.yaml`):
   - For persistent configuration across runs
   - Load with: `cfg.merge_from_file("configs/mask2former/default.yaml")`

3. **Command Line** (override any parameter):
   ```bash
   python scripts/train_swin.py --opts SOLVER.BASE_LR 0.0002 SOLVER.MAX_ITER 5000
   ```

### Key Hyperparameters

#### Model Configuration
```python
# In scripts/train_swin.py, lines 46-62
cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 2  # Number of classes in your dataset
cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES = 100  # Max objects per image
cfg.MODEL.MASK_FORMER.DEC_LAYERS = 9  # Decoder layers (9 for Swin, 6 for ResNet)
```

#### Training Parameters
```python
# In scripts/train_swin.py, lines 68-77
cfg.SOLVER.IMS_PER_BATCH = 2  # Batch size (reduce if OOM)
cfg.SOLVER.BASE_LR = 0.0001  # Learning rate
cfg.SOLVER.MAX_ITER = 3000  # Total training iterations
cfg.SOLVER.STEPS = (2000, 2700)  # LR decay milestones
cfg.SOLVER.CHECKPOINT_PERIOD = 500  # Save checkpoint every N iterations
```

#### Transfer Learning
```python
# In scripts/train_swin.py, line 87
cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1  # Backbone LR = BASE_LR * 0.1
```

#### Data Augmentation
```python
# In scripts/train_swin.py, lines 104-111
cfg.INPUT.MIN_SIZE_TRAIN = (480, 512, 544, 576, 608, 640)  # Random resize
cfg.INPUT.MAX_SIZE_TRAIN = 1333
cfg.INPUT.RANDOM_FLIP = "horizontal"  # or "none", "vertical"
```

### Recommended Settings by GPU Memory

| GPU Memory | Batch Size | Model | Image Size | Mixed Precision |
|------------|------------|-------|------------|-----------------|
| 8GB | 1-2 | Swin-Tiny | 640 | Required |
| 12GB | 2-4 | Swin-Small | 800 | Recommended |
| 24GB | 4-8 | Swin-Base | 1024 | Optional |

### Quick Tuning Guide

```python
# For small datasets (<1000 images)
cfg.SOLVER.MAX_ITER = 1000-2000
cfg.SOLVER.BASE_LR = 0.00005

# For medium datasets (1000-5000 images)
cfg.SOLVER.MAX_ITER = 3000-5000
cfg.SOLVER.BASE_LR = 0.0001

# For large datasets (>5000 images)
cfg.SOLVER.MAX_ITER = 10000+
cfg.SOLVER.BASE_LR = 0.00025
```

## 📊 Output Structure

```
outputs/experiments/
├── model_final.pth         # Final model
├── model_*.pth             # Checkpoints
├── metrics.json            # Training metrics
└── events.out.tfevents.*  # TensorBoard logs
```


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