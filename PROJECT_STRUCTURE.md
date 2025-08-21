# Mask2Former Project Structure

## Clean and Production-Ready Codebase

### Core Files
```
mask2former-detectron2/
├── train_mask2former.py      # Main Mask2Former training script
├── setup_mask2former.sh      # Automated setup script
├── requirements.txt          # Python dependencies
├── README.md                # Main documentation
├── README_MASK2FORMER.md    # Mask2Former specific guide
└── LICENSE                  # Project license
```

### Configuration
```
├── configs/
│   └── cylinder_swin_tiny.yaml  # Mask2Former config with Swin-Tiny
```

### Mask2Former Implementation
```
├── Mask2Former/             # Official Mask2Former repository (submodule)
│   ├── mask2former/         # Core implementation
│   ├── configs/             # Pre-configured model configs
│   ├── tools/               # Utility scripts
│   └── demo/                # Demo inference scripts
```

### Dataset
```
├── data/                    # COCO format dataset
│   ├── train/               # 76 training images with annotations
│   ├── valid/               # 11 validation images with annotations
│   └── test/                # 5 test images with annotations
```

### Training Outputs
```
├── outputs/
│   ├── mask2former/         # Initial training results
│   └── mask2former_v2/      # Latest training with model checkpoints
│       ├── model_final.pth  # Final trained model
│       └── metrics.json     # Training metrics
```

### Utilities
```
└── tools/
    └── dataset.py           # Dataset registration utilities
```

## Quick Start

1. **Setup Environment:**
   ```bash
   ./setup_mask2former.sh
   ```

2. **Train Mask2Former:**
   ```bash
   python train_mask2former.py
   ```

3. **Inference:**
   ```bash
   cd Mask2Former/demo
   python demo.py --config-file ../../configs/cylinder_swin_tiny.yaml \
                  --input path/to/image.jpg \
                  --opts MODEL.WEIGHTS ../../outputs/mask2former_v2/model_final.pth
   ```

## Key Features

- ✅ **Clean Architecture**: Only essential Mask2Former components
- ✅ **Working Pipeline**: Tested and verified training process
- ✅ **CUDA Optimized**: MSDeformAttn kernels compiled with gcc-11
- ✅ **Documentation**: Comprehensive setup and troubleshooting guides
- ✅ **Production Ready**: Clean codebase with no legacy files

## Removed Files

All Mask R-CNN related files have been removed:
- ❌ Old training scripts (train.py, train_simple.py)
- ❌ Mask R-CNN evaluation/inference scripts
- ❌ Detectron2 source directory
- ❌ Old training outputs
- ❌ Redundant setup files

The repository is now focused exclusively on Mask2Former training.