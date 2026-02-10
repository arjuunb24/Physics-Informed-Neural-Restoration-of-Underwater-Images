"""
GPU-Accelerated Dataset Loader for Underwater Image Restoration
Supports UIEB, SUIM-E, and EUVP datasets
"""
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from PIL import Image


class UnderwaterImageDataset(Dataset):
    """
    PyTorch Dataset for underwater image restoration
    Supports multiple datasets with unified interface
    """
    
    def __init__(self, 
                 registry_path: Path,
                 split: str = 'train',
                 transform: Optional[transforms.Compose] = None,
                 target_size: Tuple[int, int] = (256, 256),
                 datasets_to_use: Optional[List[str]] = None):
        """
        Args:
            registry_path: Path to dataset_registry.json
            split: 'train', 'val', or 'test'
            transform: Optional torchvision transforms
            target_size: Target image size (height, width)
            datasets_to_use: List of dataset names to use (None = use all)
        """
        self.split = split
        self.target_size = target_size
        self.transform = transform
        
        # Load registry
        with open(registry_path, 'r') as f:
            self.registry = json.load(f)
        
        # Get image pairs for this split
        self.pairs = self.registry['splits'][split]
        
        # Filter by dataset if specified
        if datasets_to_use is not None:
            self.pairs = [p for p in self.pairs if p['dataset'] in datasets_to_use]
        
        print(f"Loaded {len(self.pairs)} image pairs for {split} split")
        
        # Default transform if none provided
        if self.transform is None:
            self.transform = self.get_default_transform()
    
    def get_default_transform(self):
        """Get default transformation pipeline"""
        return transforms.Compose([
            transforms.ToTensor(),  # Converts to [0, 1] and changes to CHW
        ])
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        """
        Get a single image pair
        
        Returns:
            dict with keys:
                'raw': degraded image tensor (C, H, W)
                'reference': clean image tensor (C, H, W)
                'dataset': dataset name
                'index': image index
        """
        pair_info = self.pairs[idx]
        
        # Load images using OpenCV (BGR format)
        raw_img = cv2.imread(pair_info['raw'])
        ref_img = cv2.imread(pair_info['reference'])
        
        # Convert BGR to RGB
        raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
        ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
        
        # Resize to target size
        raw_img = cv2.resize(raw_img, (self.target_size[1], self.target_size[0]), 
                            interpolation=cv2.INTER_LINEAR)
        ref_img = cv2.resize(ref_img, (self.target_size[1], self.target_size[0]), 
                            interpolation=cv2.INTER_LINEAR)
        
        # Convert to PIL for transforms
        raw_img = Image.fromarray(raw_img)
        ref_img = Image.fromarray(ref_img)
        
        # Apply transforms
        if self.transform:
            raw_img = self.transform(raw_img)
            ref_img = self.transform(ref_img)
        
        return {
            'raw': raw_img,
            'reference': ref_img,
            'dataset': pair_info['dataset'],
            'index': idx
        }
    
    def get_dataset_distribution(self):
        """Get distribution of images across datasets"""
        distribution = {}
        for pair in self.pairs:
            dataset = pair['dataset']
            distribution[dataset] = distribution.get(dataset, 0) + 1
        return distribution


class UnderwaterDataLoader:
    """
    Wrapper class for creating GPU-optimized data loaders
    """
    
    def __init__(self, 
                 registry_path: Path,
                 batch_size: int = 16,
                 num_workers: int = 4,
                 target_size: Tuple[int, int] = (256, 256),
                 pin_memory: bool = True):
        """
        Args:
            registry_path: Path to dataset_registry.json
            batch_size: Batch size for data loader
            num_workers: Number of worker processes for data loading
            target_size: Target image size (height, width)
            pin_memory: Pin memory for faster GPU transfer
        """
        self.registry_path = registry_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.target_size = target_size
        self.pin_memory = pin_memory
        
    def get_train_transform(self):
        """
        Get training data augmentation transforms
        You can customize this based on your needs
        """
        return transforms.Compose([
            transforms.ToTensor(),
            # Add more augmentations here if needed
            # transforms.RandomHorizontalFlip(p=0.5),
            # transforms.ColorJitter(brightness=0.1, contrast=0.1),
        ])
    
    def get_val_test_transform(self):
        """Get validation/test transforms (no augmentation)"""
        return transforms.Compose([
            transforms.ToTensor(),
        ])
    
    def get_loader(self, split: str, datasets_to_use: Optional[List[str]] = None):
        """
        Create a DataLoader for specified split
        
        Args:
            split: 'train', 'val', or 'test'
            datasets_to_use: List of dataset names to use (None = use all)
            
        Returns:
            PyTorch DataLoader optimized for GPU
        """
        # Choose transform based on split
        if split == 'train':
            transform = self.get_train_transform()
            shuffle = True
        else:
            transform = self.get_val_test_transform()
            shuffle = False
        
        # Create dataset
        dataset = UnderwaterImageDataset(
            registry_path=self.registry_path,
            split=split,
            transform=transform,
            target_size=self.target_size,
            datasets_to_use=datasets_to_use
        )
        
        # Create DataLoader with GPU optimization
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,  # Faster GPU transfer
            persistent_workers=True if self.num_workers > 0 else False,
            prefetch_factor=2 if self.num_workers > 0 else None,
        )
        
        # Print dataset distribution
        print(f"\n{split.upper()} SPLIT - Dataset Distribution:")
        distribution = dataset.get_dataset_distribution()
        for dataset_name, count in distribution.items():
            print(f"  {dataset_name}: {count} pairs ({count/len(dataset)*100:.1f}%)")
        
        return loader
    
    def get_all_loaders(self, datasets_to_use: Optional[List[str]] = None):
        """
        Get train, val, and test loaders
        
        Returns:
            Dictionary with 'train', 'val', 'test' loaders
        """
        return {
            'train': self.get_loader('train', datasets_to_use),
            'val': self.get_loader('val', datasets_to_use),
            'test': self.get_loader('test', datasets_to_use)
        }


def test_dataloader(device: torch.device):
    """
    Test function to verify dataloader works correctly with GPU
    """
    print("\n" + "=" * 60)
    print("TESTING DATALOADER")
    print("=" * 60)
    
    # Paths
    registry_path = Path("data/registry/dataset_registry.json")
    
    if not registry_path.exists():
        print(f"❌ Registry file not found at {registry_path}")
        print("Please run data_organization.py first!")
        return
    
    # Create data loader
    data_loader = UnderwaterDataLoader(
        registry_path=registry_path,
        batch_size=8,
        num_workers=4,
        target_size=(256, 256)
    )
    
    # Get train loader
    train_loader = data_loader.get_loader('train')
    
    # Test loading a batch
    print("\nLoading first batch...")
    batch = next(iter(train_loader))
    
    # Move to GPU
    raw_imgs = batch['raw'].to(device)
    ref_imgs = batch['reference'].to(device)
    
    print(f"\n✓ Successfully loaded batch to GPU:")
    print(f"  Raw images shape: {raw_imgs.shape}")
    print(f"  Reference images shape: {ref_imgs.shape}")
    print(f"  Device: {raw_imgs.device}")
    print(f"  Dtype: {raw_imgs.dtype}")
    print(f"  Value range: [{raw_imgs.min().item():.3f}, {raw_imgs.max().item():.3f}]")
    
    # Print dataset sources in this batch
    datasets_in_batch = batch['dataset']
    print(f"\nDatasets in this batch:")
    for dataset in set(datasets_in_batch):
        count = datasets_in_batch.count(dataset)
        print(f"  {dataset}: {count} images")
    
    print("\n✓ Dataloader test completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    # Test the dataloader
    from utils import setup_gpu
    device = setup_gpu()
    test_dataloader(device)