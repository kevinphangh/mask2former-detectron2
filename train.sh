#!/bin/bash
# Training wrapper script with library fix

export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6

# Default config file
CONFIG_FILE="${1:-configs/custom_training_config.yaml}"

echo "Starting Mask2Former Training"
echo "Config: $CONFIG_FILE"
echo ""

python train.py --custom-config "$CONFIG_FILE" "${@:2}"