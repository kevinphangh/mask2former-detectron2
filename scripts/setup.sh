#!/bin/bash

echo "=================================================="
echo "MASK2FORMER SETUP SCRIPT"
echo "=================================================="

# Check if CUDA is available
if command -v nvidia-smi &> /dev/null; then
    echo "✓ CUDA detected"
    nvidia-smi --query-gpu=name,memory.total --format=csv
else
    echo "⚠️ No CUDA detected. Training will be slow."
fi

echo ""
echo "📦 Installing dependencies..."

# Install PyTorch with CUDA support
echo "Installing PyTorch with CUDA 12.1..."
pip install --break-system-packages torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install Detectron2
echo ""
echo "Installing Detectron2..."
python -m pip install --break-system-packages 'git+https://github.com/facebookresearch/detectron2.git'

# Install Mask2Former requirements
echo ""
echo "Installing Mask2Former requirements..."
cd Mask2Former
pip install --break-system-packages -r requirements.txt

# Install additional dependencies
echo ""
echo "Installing additional dependencies..."
pip install --break-system-packages opencv-python

# Compile CUDA kernel for MSDeformAttn
echo ""
echo "Compiling CUDA kernel for MSDeformAttn..."
cd mask2former/modeling/pixel_decoder/ops

# Check if CUDA_HOME is set
if [ -z "$CUDA_HOME" ]; then
    echo "⚠️ CUDA_HOME not set. Trying to detect CUDA installation..."
    
    # Try common CUDA locations
    if [ -d "/usr/local/cuda" ]; then
        export CUDA_HOME="/usr/local/cuda"
    elif [ -d "/usr/local/cuda-12.1" ]; then
        export CUDA_HOME="/usr/local/cuda-12.1"
    elif [ -d "/usr/local/cuda-11.8" ]; then
        export CUDA_HOME="/usr/local/cuda-11.8"
    else
        echo "❌ Could not find CUDA installation. Please set CUDA_HOME manually."
        echo "   Example: export CUDA_HOME=/usr/local/cuda"
        exit 1
    fi
    
    echo "Using CUDA_HOME=$CUDA_HOME"
fi

# Compile the kernel
sh make.sh

if [ $? -eq 0 ]; then
    echo "✓ CUDA kernel compiled successfully"
else
    echo "❌ Failed to compile CUDA kernel"
    echo "   You may need to install the kernel manually"
fi

# Go back to project root
cd ../../../..

echo ""
echo "=================================================="
echo "✓ Setup complete!"
echo "=================================================="
echo ""
echo "To train Mask2Former, run:"
echo "  python scripts/train.py"
echo ""
echo "Configuration can be modified in:"
echo "  scripts/train.py  (training parameters)"
echo "  configs/mask2former/default.yaml  (model config)"
echo ""