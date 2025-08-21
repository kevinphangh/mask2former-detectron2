#!/bin/bash

# Setup script for Mask2Former with Detectron2
# This script installs all necessary dependencies

echo "========================================="
echo "Mask2Former Detectron2 Setup"
echo "========================================="

# Check Python version
python_version=$(python3 --version 2>&1 | grep -Po '(?<=Python )\d+\.\d+')
echo "Python version: $python_version"

# Create virtual environment (optional but recommended)
read -p "Create virtual environment? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python3 -m venv venv
    source venv/bin/activate
    echo "Virtual environment created and activated"
fi

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch (adjust CUDA version as needed)
echo "Installing PyTorch with CUDA 11.8..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install Detectron2
echo "Installing Detectron2..."
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Clone and install Mask2Former
echo "Cloning Mask2Former repository..."
if [ ! -d "Mask2Former" ]; then
    git clone https://github.com/facebookresearch/Mask2Former.git
    cd Mask2Former
    pip install -r requirements.txt
    pip install -e .
    cd ..
else
    echo "Mask2Former already cloned"
fi

# Install additional dependencies
echo "Installing additional dependencies..."
pip install opencv-python pillow matplotlib tqdm tensorboard pycocotools

# Download pre-trained model (optional)
read -p "Download pre-trained Mask2Former model? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    mkdir -p models
    cd models
    echo "Downloading Swin-T Mask2Former model..."
    wget https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/instance/maskformer2_swin_tiny_bs16_50ep/model_final_86143f.pkl
    cd ..
fi

echo "========================================="
echo "Setup complete!"
echo "========================================="
echo ""
echo "To train on your dataset:"
echo "1. Place your COCO format data in the 'data' directory"
echo "2. Run: python scripts/train.py --config configs/cylinder_swin_tiny.yaml"
echo ""
echo "For more options, see README.md"