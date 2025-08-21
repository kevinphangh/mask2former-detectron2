# Mask2Former Training for Cylinder Detection

This project is set up to train **Mask2Former** for cylinder instance segmentation using a custom COCO-format dataset.

## Setup

### 1. Install Dependencies

Run the setup script to install all required dependencies:

```bash
chmod +x setup_mask2former.sh
./setup_mask2former.sh
```

Or manually install:

```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install Detectron2
pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Install Mask2Former requirements
cd Mask2Former
pip install -r requirements.txt

# Compile CUDA kernel for MSDeformAttn
cd mask2former/modeling/pixel_decoder/ops
sh make.sh
```

## Training

### Basic Training

Start training with default settings:

```bash
python train_mask2former.py
```

### Custom Training Options

```bash
# Train with custom batch size and iterations
python train_mask2former.py --batch-size 4 --max-iter 5000

# Resume from checkpoint
python train_mask2former.py --resume

# Train with specific GPU
CUDA_VISIBLE_DEVICES=0 python train_mask2former.py

# Disable mixed precision training
python train_mask2former.py --no-amp
```

### Configuration

The configuration file is located at: `configs/cylinder_mask2former.yaml`

Key parameters:
- **Model**: Mask2Former with ResNet-50 backbone
- **Classes**: 1 (cylinder)
- **Batch Size**: 2 (adjustable via --batch-size)
- **Learning Rate**: 0.00025
- **Max Iterations**: 3000
- **Optimizer**: AdamW
- **Mixed Precision**: Enabled by default

## Dataset Structure

The training expects data in COCO format:

```
data/
├── train/
│   ├── *.jpg (training images)
│   └── _annotations.coco.json
├── valid/
│   ├── *.jpg (validation images)
│   └── _annotations.coco.json
└── test/
    ├── *.jpg (test images)
    └── _annotations.coco.json
```

## Model Architecture

Mask2Former is a universal image segmentation model that:
- Uses a transformer-based architecture
- Employs masked attention mechanism
- Can handle instance, semantic, and panoptic segmentation
- Uses queries to predict masks and classes

### Key Components:
1. **Backbone**: ResNet-50 (can be changed to Swin Transformer)
2. **Pixel Decoder**: Multi-scale deformable attention
3. **Transformer Decoder**: Masked attention with object queries
4. **Head**: Predicts class and mask for each query

## Outputs

Training outputs are saved to `outputs/mask2former/` including:
- Model checkpoints
- Training metrics
- Evaluation results
- Tensorboard logs

## Evaluation

To evaluate a trained model:

```bash
python train_mask2former.py --eval-only --weights outputs/mask2former/model_final.pth
```

## Inference

For inference on new images:

```bash
cd Mask2Former/demo
python demo.py \
  --config-file ../../configs/cylinder_mask2former.yaml \
  --input path/to/image.jpg \
  --output path/to/output \
  --opts MODEL.WEIGHTS ../../outputs/mask2former/model_final.pth
```

## Troubleshooting

### CUDA Kernel Compilation Error
If you encounter errors compiling the CUDA kernel:
1. Ensure CUDA_HOME is set: `export CUDA_HOME=/usr/local/cuda`
2. Check CUDA version compatibility with PyTorch
3. Try manual compilation:
   ```bash
   cd Mask2Former/mask2former/modeling/pixel_decoder/ops
   python setup.py build install
   ```

### Out of Memory Error
- Reduce batch size: `--batch-size 1`
- Reduce image size in config
- Enable gradient checkpointing

### Import Errors
Ensure the Mask2Former directory is in your Python path:
```python
import sys
sys.path.append('path/to/Mask2Former')
```

## References

- [Mask2Former Paper](https://arxiv.org/abs/2112.01527)
- [Official Repository](https://github.com/facebookresearch/Mask2Former)
- [Detectron2 Documentation](https://detectron2.readthedocs.io/)