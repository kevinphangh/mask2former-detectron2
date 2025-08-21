# Mask2Former for Instance Segmentation

Production-ready Mask2Former training pipeline using Detectron2. Successfully trains transformer-based universal segmentation models with custom COCO datasets.

## 📁 Project Structure

```
mask2former-detectron2/
├── src/                     # Source code
│   ├── training/           # Training modules
│   │   ├── trainer.py      # Mask2Former trainer
│   │   └── data_loader.py  # Data loading utilities
│   └── utils/              # Utility functions
│       └── dataset.py      # Dataset registration
├── scripts/                 # Executable scripts
│   ├── train.py            # Training script
│   └── setup.sh            # Environment setup
├── configs/                 # Configuration files
│   └── mask2former/
│       └── default.yaml    # Default config
├── data/                    # Dataset directory
│   ├── train/              # Training images & annotations
│   ├── valid/              # Validation images & annotations
│   └── test/               # Test images & annotations
├── models/                  # Pre-trained models
├── outputs/                 # Training outputs
│   └── experiments/        # Experiment results
├── Mask2Former/            # Mask2Former submodule
└── requirements.txt        # Python dependencies
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

Place your COCO-format dataset in the `data/` directory:
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

```bash
python scripts/train.py
```

Training outputs will be saved to `outputs/experiments/latest/`.

## ⚙️ Configuration

Modify training parameters in `scripts/train.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NUM_CLASSES` | 2 | Number of object classes |
| `IMS_PER_BATCH` | 2 | Batch size |
| `BASE_LR` | 0.00025 | Learning rate |
| `MAX_ITER` | 1000 | Training iterations |

For advanced configuration, edit `configs/mask2former/default.yaml`.

## 🏗️ Architecture

The project follows a modular architecture:

- **`src/training/`**: Core training components
  - `trainer.py`: Custom Mask2Former trainer
  - `data_loader.py`: Data loading with mask conversion
  
- **`src/utils/`**: Utility functions
  - `dataset.py`: Dataset registration utilities
  
- **`scripts/`**: Executable scripts
  - `train.py`: Main training entry point
  - `setup.sh`: Environment setup script

## 📊 Training Output

```
outputs/experiments/latest/
├── model_final.pth         # Final trained model
├── model_*.pth             # Checkpoints
├── metrics.json            # Training metrics
└── events.out.tfevents.*  # TensorBoard logs
```

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA kernel compilation fails | Install gcc-11: `sudo apt install gcc-11 g++-11` |
| Out of memory | Reduce batch size in `scripts/train.py` |
| Missing masks | Ensure COCO annotations include segmentation polygons |

## 📚 API Reference

### Training
```python
from src.training import Mask2FormerTrainer
from src.utils import register_datasets

# Register datasets
register_datasets(data_dir="data")

# Train model
trainer = Mask2FormerTrainer(cfg)
trainer.train()
```

### Inference
```python
from detectron2.engine import DefaultPredictor

predictor = DefaultPredictor(cfg)
outputs = predictor(image)
```

## 🛠️ Development

### Running Tests
```bash
python -m pytest tests/
```

### Code Style
```bash
black src/ scripts/
flake8 src/ scripts/
```

## 📝 License

Apache 2.0 (Detectron2) and MIT (Mask2Former)

## 🔗 References

- [Mask2Former Paper](https://arxiv.org/abs/2112.01527)
- [Official Repository](https://github.com/facebookresearch/Mask2Former)
- [Detectron2 Documentation](https://detectron2.readthedocs.io/)