# Mask2Former Training

Train Mask2Former models on custom COCO-format datasets with a simple config file.

## Setup

```bash
# Create environment
conda create -n mask2former python=3.10
conda activate mask2former

# Install dependencies
bash scripts/setup.sh
```

## Dataset

Put your COCO-format data in:
- `data/train/` with `_annotations.coco.json`
- `data/valid/` with `_annotations.coco.json`

## Training

1. Edit `configs/custom_training_config.yaml`:
```yaml
MODEL:
  NUM_CLASSES: 2    # Your number of classes

TRAINING:
  BATCH_SIZE: 2      # Reduce if GPU memory issues
  MAX_EPOCHS: 50     # How long to train
```

2. Run training:
```bash
./train.sh
```

## Key Config Options

| Setting | Description | Default |
|---------|-------------|---------|
| `MODEL.NAME` | Model size (swin_tiny/small/base) | swin_tiny |
| `MODEL.NUM_CLASSES` | Number of object classes | 2 |
| `TRAINING.MODE` | "epochs" or "iterations" | iterations |
| `TRAINING.BATCH_SIZE` | Batch size | 2 |
| `OPTIMIZER.BASE_LR` | Learning rate | 0.00005 |

## Commands

```bash
# Train with default config
./train.sh

# Train with custom config
./train.sh configs/my_config.yaml

# Resume training
./train.sh --resume
```

## Output

Models saved to `outputs/{model_name}_{timestamp}/`

## GPU Requirements

- **swin_tiny**: 6GB VRAM
- **swin_small**: 8GB VRAM  
- **swin_base**: 16GB VRAM

## Troubleshooting

**Out of memory**: Reduce `BATCH_SIZE` to 1 or use smaller model

**Library error**: Use `./train.sh` (it handles library paths automatically)