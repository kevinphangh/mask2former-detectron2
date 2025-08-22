#!/usr/bin/env python3
"""
Verification script to test if Mask2Former is properly set up.
"""

import sys
import warnings
warnings.filterwarnings('ignore')

# Add paths
sys.path.insert(0, '.')
sys.path.insert(0, './Mask2Former')

def check_cuda():
    """Check CUDA availability."""
    print("🔍 Checking CUDA...")
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        print(f"  ✓ PyTorch version: {torch.__version__}")
        print(f"  ✓ CUDA available: {cuda_available}")
        if cuda_available:
            print(f"  ✓ GPU: {torch.cuda.get_device_name(0)}")
            print(f"  ✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def check_detectron2():
    """Check Detectron2 installation."""
    print("\n🔍 Checking Detectron2...")
    try:
        from detectron2.config import get_cfg
        from detectron2.projects.deeplab import add_deeplab_config
        print("  ✓ Detectron2 imported successfully")
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def check_mask2former():
    """Check Mask2Former installation."""
    print("\n🔍 Checking Mask2Former...")
    try:
        from mask2former import add_maskformer2_config
        print("  ✓ Mask2Former imported successfully")
        
        # Check CUDA kernel status
        from mask2former.modeling.pixel_decoder.ops.functions import ms_deform_attn_func
        if hasattr(ms_deform_attn_func, 'CUDA_KERNEL_AVAILABLE'):
            if ms_deform_attn_func.CUDA_KERNEL_AVAILABLE:
                print("  ✓ CUDA kernel: Compiled and ready (fast mode)")
            else:
                print("  ⚠ CUDA kernel: Using PyTorch fallback (slower but works)")
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def check_dataset():
    """Check dataset availability."""
    print("\n🔍 Checking dataset...")
    from pathlib import Path
    
    train_path = Path("data/train/_annotations.coco.json")
    valid_path = Path("data/valid/_annotations.coco.json")
    
    if train_path.exists():
        print("  ✓ Training dataset found")
    else:
        print("  ✗ Training dataset not found at data/train/_annotations.coco.json")
    
    if valid_path.exists():
        print("  ✓ Validation dataset found")
    else:
        print("  ⚠ Validation dataset not found (optional)")
    
    return train_path.exists()

def main():
    print("="*60)
    print("MASK2FORMER SETUP VERIFICATION")
    print("="*60)
    
    checks = [
        ("CUDA", check_cuda()),
        ("Detectron2", check_detectron2()),
        ("Mask2Former", check_mask2former()),
        ("Dataset", check_dataset())
    ]
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    all_passed = True
    for name, passed in checks:
        status = "✅" if passed else "❌"
        print(f"  {status} {name}")
        if not passed:
            all_passed = False
    
    print("="*60)
    
    if all_passed:
        print("\n✅ All checks passed! You're ready to train.")
        print("\nTo start training, run:")
        print("  ./train.sh")
    else:
        print("\n⚠ Some checks failed. Please fix the issues above.")
        print("\nFor help, check the README.md or run:")
        print("  ./scripts/setup.sh")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())