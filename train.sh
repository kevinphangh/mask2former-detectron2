#!/bin/bash
# ============================================================
# Mask2Former Training Script with Automatic Environment Fix
# ============================================================

echo "=================================================="
echo "MASK2FORMER TRAINING LAUNCHER"
echo "=================================================="

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mask2former

# Fix library compatibility issues and enable CUDA kernel
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
export LD_LIBRARY_PATH=/home/kevinphangh/miniconda3/envs/mask2former/lib/python3.9/site-packages/torch/lib:$LD_LIBRARY_PATH
export CUDA_HOME=/usr

# Navigate to project directory
cd /home/kevinphangh/projects/mask2former-detectron2

# Check CUDA status
echo ""
echo "🔍 Checking environment..."
python -c "
import torch
import warnings
warnings.filterwarnings('ignore')
print(f'  ✓ PyTorch version: {torch.__version__}')
print(f'  ✓ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  ✓ GPU: {torch.cuda.get_device_name(0)}')
    print(f'  ✓ CUDA version: {torch.version.cuda}')
"

# Test imports
echo ""
echo "🔍 Testing Mask2Former imports..."
python -c "
import sys
import warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
sys.path.insert(0, './Mask2Former')
try:
    from mask2former import add_maskformer2_config
    print('  ✓ Mask2Former loaded successfully')
    # Test CUDA kernel
    try:
        sys.path.insert(0, './Mask2Former/mask2former/modeling/pixel_decoder/ops')
        import MultiScaleDeformableAttention
        print('  ✓ CUDA kernel for MSDeformAttn loaded successfully (fast performance)')
    except:
        print('  ℹ Using PyTorch fallback for MSDeformAttn (slower but works)')
except Exception as e:
    print(f'  ✗ Error: {e}')
    sys.exit(1)
"

echo ""
echo "=================================================="
echo "Starting training..."
echo "=================================================="
echo ""

# Run the training script with all arguments passed through
python scripts/train_mask2former.py "$@"