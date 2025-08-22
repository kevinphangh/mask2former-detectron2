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

# Detect if we're in a virtual environment or need system packages flag
PIP_FLAGS=""
if [[ "$VIRTUAL_ENV" == "" ]]; then
    # Check if the system requires --break-system-packages (Debian/Ubuntu)
    if pip install --help 2>&1 | grep -q "break-system-packages"; then
        PIP_FLAGS="--break-system-packages"
        echo "Note: Installing with --break-system-packages flag"
    fi
fi

# Install PyTorch with CUDA support
echo "Installing PyTorch with CUDA 12.1..."
pip install $PIP_FLAGS torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install Detectron2
echo ""
echo "Installing Detectron2..."
python -m pip install $PIP_FLAGS 'git+https://github.com/facebookresearch/detectron2.git'

# Get the project root directory (parent of scripts folder)
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

# Verify Mask2Former is present
echo ""
if [ ! -d "$PROJECT_ROOT/Mask2Former" ]; then
    echo "❌ Mask2Former directory not found!"
    echo "   Please ensure the Mask2Former folder exists in: $PROJECT_ROOT"
    exit 1
else
    echo "✓ Mask2Former directory found"
fi

# Install Mask2Former requirements
echo ""
echo "Installing Mask2Former requirements..."
cd "$PROJECT_ROOT/Mask2Former"
pip install $PIP_FLAGS -r requirements.txt

# Install additional dependencies
echo ""
echo "Installing additional dependencies..."
pip install $PIP_FLAGS opencv-python

# Compile CUDA kernel for MSDeformAttn
echo ""
echo "Compiling CUDA kernel for MSDeformAttn..."
cd mask2former/modeling/pixel_decoder/ops

# Check if CUDA_HOME is set
if [ -z "$CUDA_HOME" ]; then
    echo "⚠️ CUDA_HOME not set. Trying to detect CUDA installation..."
    
    # Check if nvcc exists (Ubuntu apt style installation)
    if command -v nvcc &> /dev/null; then
        # nvcc is in /usr/bin, so CUDA_HOME should be /usr
        export CUDA_HOME="/usr"
        echo "Found nvcc at $(which nvcc)"
    # Try common CUDA locations
    elif [ -d "/usr/local/cuda" ]; then
        export CUDA_HOME="/usr/local/cuda"
    elif [ -d "/usr/lib/cuda" ]; then
        # Ubuntu/Debian style installation from apt (alternate location)
        export CUDA_HOME="/usr/lib/cuda"
    elif [ -d "/usr/local/cuda-12.1" ]; then
        export CUDA_HOME="/usr/local/cuda-12.1"
    elif [ -d "/usr/local/cuda-11.8" ]; then
        export CUDA_HOME="/usr/local/cuda-11.8"
    else
        echo "❌ Could not find CUDA installation. Please set CUDA_HOME manually."
        echo "   For Ubuntu/WSL with apt-installed CUDA: export CUDA_HOME=/usr"
        echo "   For standard CUDA installation: export CUDA_HOME=/usr/local/cuda"
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
cd "$PROJECT_ROOT"

echo ""
echo "=================================================="
echo "✓ Setup complete!"
echo "=================================================="
echo ""
echo "To train Mask2Former, run:"
echo "  python train.py"
echo ""
echo "Configuration can be modified in:"
echo "  train.py  (training parameters)"
echo ""