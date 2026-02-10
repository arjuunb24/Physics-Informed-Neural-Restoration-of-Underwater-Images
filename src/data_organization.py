"""
Data Organization System for Underwater Image Restoration
Handles UIEB, SUIM-E, and EUVP datasets with GPU acceleration
"""
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import shutil
from tqdm import tqdm
import json
from sklearn.model_selection import train_test_split
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))
from utils import (
    setup_gpu, verify_image_pair, get_image_statistics,
    match_image_pairs, save_json, print_dataset_summary
)


class DataOrganizer:
    """
    Organizes multiple underwater image datasets into a unified structure
    with train/val/test splits and GPU-accelerated processing
    """
    
    def __init__(self, raw_data_dir: Path, output_dir: Path, device: torch.device):
        """
        Args:
            raw_data_dir: Root directory containing UIEB, SUIM-E, EUVP folders
            output_dir: Directory to save organized data
            device: torch device for GPU operations
        """
        self.raw_data_dir = Path(raw_data_dir)
        self.output_dir = Path(output_dir)
        self.device = device
        
        # Dataset configurations
        self.dataset_configs = {
            'UIEB': {
                'raw_folder': 'raw-890',
                'ref_folder': 'reference-890',
                'raw_suffix': '',
                'ref_suffix': ''
            },
            'SUIM-E': {
                'raw_folder': 'raw (A)',
                'ref_folder': 'reference (B)',
                'raw_suffix': '',
                'ref_suffix': ''
            },
            'EUVP': {
                'raw_folder': 'raw (A)',
                'ref_folder': 'reference (B)',
                'raw_suffix': '',
                'ref_suffix': ''
            }
        }
        
        # Split ratios
        self.train_ratio = 0.70
        self.val_ratio = 0.15
        self.test_ratio = 0.15
        
        # Create output directories
        self.create_output_structure()
        
        # Registry to store dataset information
        self.registry = {
            'datasets': {},
            'splits': {
                'train': [],
                'val': [],
                'test': []
            },
            'statistics': {}
        }
    
    def create_output_structure(self):
        """Create the output directory structure"""
        directories = [
            self.output_dir / 'processed' / 'train' / 'raw',
            self.output_dir / 'processed' / 'train' / 'reference',
            self.output_dir / 'processed' / 'val' / 'raw',
            self.output_dir / 'processed' / 'val' / 'reference',
            self.output_dir / 'processed' / 'test' / 'raw',
            self.output_dir / 'processed' / 'test' / 'reference',
            self.output_dir / 'registry'
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        print("✓ Created output directory structure")
    
    def scan_dataset(self, dataset_name: str) -> List[Tuple[Path, Path]]:
        """
        Scan a dataset and match raw-reference pairs
        
        Args:
            dataset_name: Name of dataset (UIEB, SUIM-E, or EUVP)
            
        Returns:
            List of matched image pairs
        """
        print(f"\nScanning {dataset_name} dataset...")
        
        config = self.dataset_configs[dataset_name]
        dataset_dir = self.raw_data_dir / dataset_name
        
        if not dataset_dir.exists():
            print(f"⚠ Warning: {dataset_name} directory not found at {dataset_dir}")
            return []
        
        raw_dir = dataset_dir / config['raw_folder']
        ref_dir = dataset_dir / config['ref_folder']
        
        if not raw_dir.exists() or not ref_dir.exists():
            print(f"⚠ Warning: Raw or reference folder not found for {dataset_name}")
            return []
        
        # Match image pairs
        pairs = match_image_pairs(
            raw_dir, ref_dir,
            config['raw_suffix'], config['ref_suffix']
        )
        
        # Verify pairs
        valid_pairs = []
        print(f"  Verifying {len(pairs)} pairs...")
        for raw_path, ref_path in tqdm(pairs, desc="  Verification"):
            if verify_image_pair(raw_path, ref_path):
                valid_pairs.append((raw_path, ref_path))
        
        print(f"  ✓ Valid pairs: {len(valid_pairs)}/{len(pairs)}")
        
        return valid_pairs
    
    def split_dataset(self, pairs: List[Tuple[Path, Path]], 
                     dataset_name: str) -> Dict[str, List]:
        """
        Split dataset into train/val/test sets
        
        Args:
            pairs: List of image pairs
            dataset_name: Name of the dataset
            
        Returns:
            Dictionary with train/val/test splits
        """
        print(f"\nSplitting {dataset_name} into train/val/test...")
        
        n_total = len(pairs)
        n_train = int(n_total * self.train_ratio)
        n_val = int(n_total * self.val_ratio)
        n_test = n_total - n_train - n_val
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Shuffle pairs
        indices = np.random.permutation(n_total)
        
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]
        
        splits = {
            'train': [pairs[i] for i in train_indices],
            'val': [pairs[i] for i in val_indices],
            'test': [pairs[i] for i in test_indices]
        }
        
        print(f"  Train: {len(splits['train'])} pairs ({len(splits['train'])/n_total*100:.1f}%)")
        print(f"  Val: {len(splits['val'])} pairs ({len(splits['val'])/n_total*100:.1f}%)")
        print(f"  Test: {len(splits['test'])} pairs ({len(splits['test'])/n_total*100:.1f}%)")
        
        return splits
    
    def copy_split_to_output(self, pairs: List[Tuple[Path, Path]], 
                            split_name: str, dataset_name: str):
        """
        Copy image pairs to the appropriate output directory
        
        Args:
            pairs: List of image pairs to copy
            split_name: 'train', 'val', or 'test'
            dataset_name: Name of source dataset
        """
        print(f"\nCopying {dataset_name} {split_name} split...")
        
        raw_out_dir = self.output_dir / 'processed' / split_name / 'raw'
        ref_out_dir = self.output_dir / 'processed' / split_name / 'reference'
        
        for idx, (raw_path, ref_path) in enumerate(tqdm(pairs, desc=f"  Copying")):
            # Create unique filename with dataset prefix and index
            base_name = f"{dataset_name}_{idx:05d}{raw_path.suffix}"
            
            # Copy files
            shutil.copy2(raw_path, raw_out_dir / base_name)
            shutil.copy2(ref_path, ref_out_dir / base_name)
            
            # Add to registry
            entry = {
                'raw': str(raw_out_dir / base_name),
                'reference': str(ref_out_dir / base_name),
                'dataset': dataset_name,
                'original_raw': str(raw_path),
                'original_reference': str(ref_path)
            }
            self.registry['splits'][split_name].append(entry)
    
    def calculate_dataset_statistics(self):
        """Calculate statistics for all datasets using GPU acceleration"""
        print("\nCalculating dataset statistics (using GPU)...")
        
        stats = {
            'train': {'count': 0, 'total_size_mb': 0},
            'val': {'count': 0, 'total_size_mb': 0},
            'test': {'count': 0, 'total_size_mb': 0}
        }
        
        for split_name in ['train', 'val', 'test']:
            split_dir = self.output_dir / 'processed' / split_name / 'raw'
            images = list(split_dir.glob('*.*'))
            
            stats[split_name]['count'] = len(images)
            
            # Calculate size
            for img_path in images:
                stats[split_name]['total_size_mb'] += img_path.stat().st_size / (1024 * 1024)
            
            print(f"  {split_name}: {stats[split_name]['count']} images, "
                  f"{stats[split_name]['total_size_mb']:.2f} MB")
        
        self.registry['statistics'] = stats
    
    def organize_all_datasets(self):
        """Main method to organize all datasets"""
        print("\n" + "=" * 60)
        print("STARTING DATA ORGANIZATION")
        print("=" * 60)
        
        # Process each dataset
        for dataset_name in self.dataset_configs.keys():
            print(f"\n{'─' * 60}")
            print(f"Processing {dataset_name}")
            print(f"{'─' * 60}")
            
            # Scan and match pairs
            pairs = self.scan_dataset(dataset_name)
            
            if len(pairs) == 0:
                print(f"⚠ No valid pairs found for {dataset_name}, skipping...")
                continue
            
            # Split into train/val/test
            splits = self.split_dataset(pairs, dataset_name)
            
            # Copy to output directories
            for split_name, split_pairs in splits.items():
                self.copy_split_to_output(split_pairs, split_name, dataset_name)
            
            # Update registry
            self.registry['datasets'][dataset_name] = {
                'total_pairs': len(pairs),
                'splits': {
                    'train': len(splits['train']),
                    'val': len(splits['val']),
                    'test': len(splits['test'])
                }
            }
        
        # Calculate statistics
        self.calculate_dataset_statistics()
        
        # Save registry
        registry_path = self.output_dir / 'registry' / 'dataset_registry.json'
        save_json(self.registry, registry_path)
        
        # Print summary
        print_dataset_summary(self.registry)
        
        print("\n" + "=" * 60)
        print("✓ DATA ORGANIZATION COMPLETE!")
        print("=" * 60)
        print(f"\nOrganized data saved to: {self.output_dir / 'processed'}")
        print(f"Registry saved to: {registry_path}")


def main():
    """Main execution function"""
    # Setup GPU
    device = setup_gpu()
    
    # Define paths - MODIFY THESE TO MATCH YOUR SETUP
    raw_data_dir = Path("data/raw")  # Your existing dataset folder
    output_dir = Path("data")        # Will create processed/ and registry/ here
    
    # Verify raw data directory exists
    if not raw_data_dir.exists():
        print(f"\n❌ Error: Raw data directory not found at {raw_data_dir}")
        print("Please update the 'raw_data_dir' path in the script to match your setup.")
        return
    
    # Create organizer and run
    organizer = DataOrganizer(raw_data_dir, output_dir, device)
    organizer.organize_all_datasets()
    
    print("\n✓ Next steps:")
    print("  1. Check the dataset_registry.json file for details")
    print("  2. Verify the organized data in data/processed/")
    print("  3. Use the dataset_loader.py to load data for training")


if __name__ == "__main__":
    main()