"""
Utility functions for GPU-accelerated image processing and data handling
"""
import torch
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional


def setup_gpu():
    """
    Setup and verify GPU availability for CUDA operations
    Returns device object and prints GPU information
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("=" * 60)
        print("GPU SETUP INFORMATION")
        print("=" * 60)
        print(f"✓ CUDA is available")
        print(f"✓ GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"✓ CUDA Version: {torch.version.cuda}")
        print(f"✓ Number of GPUs: {torch.cuda.device_count()}")
        print(f"✓ Current GPU: {torch.cuda.current_device()}")
        
        # Get GPU memory info
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✓ GPU Memory: {gpu_memory:.2f} GB")
        print(f"✓ Allocated Memory: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"✓ Cached Memory: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
        print("=" * 60)
        
        # Enable cuDNN benchmarking for faster training
        torch.backends.cudnn.benchmark = True
        print("✓ cuDNN benchmark enabled for optimized performance")
        print("=" * 60)
        
    else:
        device = torch.device('cpu')
        print("⚠ WARNING: CUDA is not available. Running on CPU.")
        print("Please check your PyTorch installation and NVIDIA drivers.")
    
    return device


def verify_image_pair(raw_path: Path, ref_path: Path) -> bool:
    """
    Verify that an image pair is valid and readable
    
    Args:
        raw_path: Path to raw/degraded image
        ref_path: Path to reference/clean image
        
    Returns:
        bool: True if both images are valid
    """
    try:
        # Try to open both images
        raw_img = Image.open(raw_path)
        ref_img = Image.open(ref_path)
        
        # Verify they have content
        if raw_img.size[0] == 0 or raw_img.size[1] == 0:
            return False
        if ref_img.size[0] == 0 or ref_img.size[1] == 0:
            return False
            
        return True
    except Exception as e:
        print(f"Error verifying pair {raw_path.name}: {str(e)}")
        return False


def get_image_statistics(image_path: Path, device: torch.device) -> Dict:
    """
    Calculate image statistics using GPU acceleration
    
    Args:
        image_path: Path to image
        device: torch device (cuda or cpu)
        
    Returns:
        Dictionary containing image statistics
    """
    try:
        # Read image
        img = cv2.imread(str(image_path))
        if img is None:
            return None
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Convert to tensor and move to GPU
        img_tensor = torch.from_numpy(img).float().to(device) / 255.0
        
        # Calculate statistics on GPU
        stats = {
            'height': img.shape[0],
            'width': img.shape[1],
            'channels': img.shape[2],
            'mean': img_tensor.mean(dim=(0, 1)).cpu().numpy().tolist(),
            'std': img_tensor.std(dim=(0, 1)).cpu().numpy().tolist(),
            'min': img_tensor.min().item(),
            'max': img_tensor.max().item(),
            'size_mb': image_path.stat().st_size / (1024 * 1024)
        }
        
        return stats
    except Exception as e:
        print(f"Error calculating statistics for {image_path}: {str(e)}")
        return None


def match_image_pairs(raw_dir: Path, ref_dir: Path, 
                      raw_suffix: str = '', ref_suffix: str = '') -> List[Tuple[Path, Path]]:
    """
    Match raw and reference image pairs from two directories
    
    Args:
        raw_dir: Directory containing raw/degraded images
        ref_dir: Directory containing reference/clean images
        raw_suffix: Optional suffix to remove from raw filenames for matching
        ref_suffix: Optional suffix to remove from reference filenames for matching
        
    Returns:
        List of tuples (raw_path, ref_path)
    """
    # Get all images from both directories
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.JPG', '.PNG'}
    
    raw_images = {f.stem: f for f in raw_dir.iterdir() 
                  if f.suffix in image_extensions}
    ref_images = {f.stem: f for f in ref_dir.iterdir() 
                  if f.suffix in image_extensions}
    
    pairs = []
    unmatched_raw = []
    unmatched_ref = []
    
    # Try to match images
    for raw_name, raw_path in raw_images.items():
        # Try exact match first
        if raw_name in ref_images:
            pairs.append((raw_path, ref_images[raw_name]))
        else:
            # Try with suffix removal
            clean_name = raw_name.replace(raw_suffix, '')
            matching_ref = None
            
            for ref_name, ref_path in ref_images.items():
                clean_ref_name = ref_name.replace(ref_suffix, '')
                if clean_name == clean_ref_name:
                    matching_ref = ref_path
                    break
            
            if matching_ref:
                pairs.append((raw_path, matching_ref))
            else:
                unmatched_raw.append(raw_name)
    
    # Find unmatched reference images
    matched_refs = {pair[1].stem for pair in pairs}
    unmatched_ref = [name for name in ref_images.keys() if name not in matched_refs]
    
    # Report matching results
    print(f"  Matched pairs: {len(pairs)}")
    if unmatched_raw:
        print(f"  ⚠ Unmatched raw images: {len(unmatched_raw)}")
    if unmatched_ref:
        print(f"  ⚠ Unmatched reference images: {len(unmatched_ref)}")
    
    return pairs


def save_json(data: Dict, filepath: Path):
    """Save dictionary to JSON file"""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"✓ Saved: {filepath}")


def load_json(filepath: Path) -> Dict:
    """Load dictionary from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


def print_dataset_summary(registry: Dict):
    """Print a formatted summary of the dataset registry"""
    print("\n" + "=" * 60)
    print("DATASET REGISTRY SUMMARY")
    print("=" * 60)
    
    total_pairs = 0
    for dataset_name, dataset_info in registry['datasets'].items():
        print(f"\n{dataset_name}:")
        print(f"  Total pairs: {dataset_info['total_pairs']}")
        print(f"  Train: {dataset_info['splits']['train']} ({dataset_info['splits']['train']/dataset_info['total_pairs']*100:.1f}%)")
        print(f"  Val: {dataset_info['splits']['val']} ({dataset_info['splits']['val']/dataset_info['total_pairs']*100:.1f}%)")
        print(f"  Test: {dataset_info['splits']['test']} ({dataset_info['splits']['test']/dataset_info['total_pairs']*100:.1f}%)")
        total_pairs += dataset_info['total_pairs']
    
    print(f"\n{'─' * 60}")
    print(f"TOTAL PAIRS ACROSS ALL DATASETS: {total_pairs}")
    print(f"Total Train: {sum(d['splits']['train'] for d in registry['datasets'].values())}")
    print(f"Total Val: {sum(d['splits']['val'] for d in registry['datasets'].values())}")
    print(f"Total Test: {sum(d['splits']['test'] for d in registry['datasets'].values())}")
    print("=" * 60)