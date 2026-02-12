"""
Data Table Construction Module for Implicit Neural Representation Training
Integrates coordinate sampling with image data loading and Fourier feature encoding
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
import sys

# Import coordinate grid generator and Fourier encoding
from coordinate_grid_generator import CoordinateGridGenerator
from fourier_encoding import FourierFeatureEncoding, create_encoding


class ImplicitRepresentationDataset(Dataset):
    """
    PyTorch Dataset for implicit neural representation training.
    Combines image pairs with coordinate sampling and Fourier encoding to create data tables.
    """
    
    def __init__(
        self,
        registry_path: Path,
        split: str = 'train',
        target_size: int = 256,
        samples_per_image: int = 2048,
        sampling_strategy: str = 'random',
        edge_bias: float = 0.7,
        datasets_to_use: Optional[List[str]] = None,
        use_fourier_encoding: bool = True,
        fourier_mapping_size: int = 128,
        fourier_scale: float = 10.0,
        include_original_coords: bool = False
    ):
        """
        Args:
            registry_path: Path to dataset_registry.json
            split: 'train', 'val', or 'test'
            target_size: Image size (assumes square images)
            samples_per_image: Number of coordinate samples per image
            sampling_strategy: 'random', 'edge_weighted', or 'uniform'
            edge_bias: Probability weight for edge regions (for edge_weighted strategy)
            datasets_to_use: List of dataset names to use (None = use all)
            use_fourier_encoding: Whether to apply Fourier feature encoding
            fourier_mapping_size: Number of Fourier frequency components (output: 2*mapping_size)
            fourier_scale: Frequency scale (sigma) for Fourier encoding
            include_original_coords: Concatenate original coordinates with Fourier features
        """
        self.split = split
        self.target_size = target_size
        self.samples_per_image = samples_per_image
        self.sampling_strategy = sampling_strategy
        self.edge_bias = edge_bias
        self.use_fourier_encoding = use_fourier_encoding
        self.include_original_coords = include_original_coords
        
        # Load registry
        with open(registry_path, 'r') as f:
            self.registry = json.load(f)
        
        # Get image pairs for this split
        self.pairs = self.registry['splits'][split]
        
        # Filter by dataset if specified
        if datasets_to_use is not None:
            self.pairs = [p for p in self.pairs if p['dataset'] in datasets_to_use]
        
        # Initialize coordinate grid generator (CPU for now, will move to GPU in collate)
        self.coord_generator = CoordinateGridGenerator(
            image_size=target_size,
            device=torch.device('cpu'),
            samples_per_image=samples_per_image
        )
        
        # Initialize Fourier encoder (CPU - will be moved to GPU in collate function)
        if use_fourier_encoding:
            self.fourier_encoder = FourierFeatureEncoding(
                input_dim=2,
                mapping_size=fourier_mapping_size,
                scale=fourier_scale,
                device=torch.device('cpu')
            )
            self.encoded_dim = self.fourier_encoder.output_dim
            if include_original_coords:
                self.encoded_dim += 2  # Add 2 for original (x, y)
        else:
            self.fourier_encoder = None
            self.encoded_dim = 2  # Just (x, y)
        
        print(f"ImplicitRepresentationDataset initialized:")
        print(f"  Split: {split}")
        print(f"  Total pairs: {len(self.pairs)}")
        print(f"  Sampling strategy: {sampling_strategy}")
        print(f"  Samples per image: {samples_per_image}")
        print(f"  Fourier encoding: {use_fourier_encoding}")
        if use_fourier_encoding:
            print(f"  Fourier mapping size: {fourier_mapping_size}")
            print(f"  Fourier scale: {fourier_scale}")
            print(f"  Include original coords: {include_original_coords}")
            print(f"  Encoded feature dimension: {self.encoded_dim}")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        """
        Get a single data table entry.
        
        Returns:
            dict with keys:
                'coordinates': Normalized (x, y) coordinates [N, 2]
                'encoded_features': Fourier-encoded features [N, 256] or [N, 258] (if enabled)
                'degraded_rgb': RGB values from degraded image [N, 3]
                'reference_rgb': RGB values from reference image [N, 3]
                'dataset': Dataset name
                'pair_id': Image pair identifier
        """
        pair_info = self.pairs[idx]
        
        # Load preprocessed images (should be 256x256, but verify)
        raw_path = Path(pair_info['raw'])
        ref_path = Path(pair_info['reference'])
        
        # Load images
        raw_img = Image.open(raw_path).convert('RGB')
        ref_img = Image.open(ref_path).convert('RGB')
        
        # Ensure images are exactly target_size x target_size
        if raw_img.size != (self.target_size, self.target_size):
            raw_img = raw_img.resize((self.target_size, self.target_size), Image.BILINEAR)
        if ref_img.size != (self.target_size, self.target_size):
            ref_img = ref_img.resize((self.target_size, self.target_size), Image.BILINEAR)
        
        # Convert to tensors [C, H, W] with values in [0, 1]
        raw_tensor = torch.from_numpy(np.array(raw_img).astype(np.float32) / 255.0).permute(2, 0, 1)
        ref_tensor = torch.from_numpy(np.array(ref_img).astype(np.float32) / 255.0).permute(2, 0, 1)
        
        # Sample coordinates based on strategy (on CPU)
        if self.sampling_strategy == 'random':
            coordinates, indices = self.coord_generator.sample_random(self.samples_per_image)
            
        elif self.sampling_strategy == 'edge_weighted':
            # Compute edge map
            edge_map = self.coord_generator.compute_edge_map(raw_tensor, method='sobel')
            coordinates, indices = self.coord_generator.sample_edge_weighted(
                edge_map,
                self.samples_per_image,
                self.edge_bias
            )
            
        elif self.sampling_strategy == 'uniform':
            # Calculate stride to get approximately samples_per_image points
            total_pixels = self.target_size ** 2
            stride = max(1, int(np.sqrt(total_pixels / self.samples_per_image)))
            coordinates, indices = self.coord_generator.sample_uniform_grid(stride)
            
        else:
            raise ValueError(f"Unknown sampling strategy: {self.sampling_strategy}")
        
        # Create data table (on CPU)
        data_table = self.coord_generator.create_data_table(
            coordinates,
            indices,
            raw_tensor,
            ref_tensor
        )
        
        # Apply Fourier encoding if enabled
        if self.use_fourier_encoding:
            encoded_features = self.fourier_encoder.encode_with_coordinates(
                data_table['coordinates'],
                include_original=self.include_original_coords
            )
        else:
            encoded_features = data_table['coordinates']
        
        return {
            'coordinates': data_table['coordinates'],      # Original coords [N, 2]
            'encoded_features': encoded_features,          # Encoded features [N, 256/258]
            'degraded_rgb': data_table['degraded_rgb'],
            'reference_rgb': data_table['reference_rgb'],
            'dataset': pair_info['dataset'],
            'pair_id': raw_path.stem
        }
    
    def get_dataset_distribution(self):
        """Get distribution of images across datasets"""
        distribution = {}
        for pair in self.pairs:
            dataset = pair['dataset']
            distribution[dataset] = distribution.get(dataset, 0) + 1
        return distribution


def custom_collate_fn(batch):
    """
    Custom collate function to properly stack batched data tables with Fourier encoding.
    
    Args:
        batch: List of dictionaries from __getitem__
        
    Returns:
        Batched dictionary with proper tensor shapes
    """
    # Stack tensors
    coordinates = torch.stack([item['coordinates'] for item in batch])
    encoded_features = torch.stack([item['encoded_features'] for item in batch])
    degraded_rgb = torch.stack([item['degraded_rgb'] for item in batch])
    reference_rgb = torch.stack([item['reference_rgb'] for item in batch])
    
    # Keep metadata as lists
    datasets = [item['dataset'] for item in batch]
    pair_ids = [item['pair_id'] for item in batch]
    
    return {
        'coordinates': coordinates,            # [B, N, 2] - original coordinates
        'encoded_features': encoded_features,  # [B, N, 256/258] - Fourier encoded
        'degraded_rgb': degraded_rgb,          # [B, N, 3]
        'reference_rgb': reference_rgb,        # [B, N, 3]
        'dataset': datasets,                   # List[str]
        'pair_id': pair_ids                   # List[str]
    }


class ImplicitRepresentationDataLoader:
    """
    Wrapper class for creating GPU-optimized data loaders with data table construction
    and Fourier feature encoding.
    """
    
    def __init__(
        self,
        registry_path: Path,
        batch_size: int = 16,
        num_workers: int = 4,
        target_size: int = 256,
        samples_per_image: int = 2048,
        pin_memory: bool = True,
        use_fourier_encoding: bool = True,
        fourier_mapping_size: int = 128,
        fourier_scale: float = 10.0,
        include_original_coords: bool = False
    ):
        """
        Args:
            registry_path: Path to dataset_registry.json
            batch_size: Batch size for data loader
            num_workers: Number of worker processes for data loading
            target_size: Image size (assumes square images)
            samples_per_image: Number of coordinate samples per image
            pin_memory: Pin memory for faster GPU transfer
            use_fourier_encoding: Whether to apply Fourier feature encoding
            fourier_mapping_size: Number of Fourier frequency components
            fourier_scale: Frequency scale (sigma) for Fourier encoding
            include_original_coords: Concatenate original coordinates with Fourier features
        """
        self.registry_path = registry_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.target_size = target_size
        self.samples_per_image = samples_per_image
        self.pin_memory = pin_memory
        self.use_fourier_encoding = use_fourier_encoding
        self.fourier_mapping_size = fourier_mapping_size
        self.fourier_scale = fourier_scale
        self.include_original_coords = include_original_coords
    
    def get_loader(
        self,
        split: str,
        sampling_strategy: str = 'random',
        edge_bias: float = 0.7,
        datasets_to_use: Optional[List[str]] = None,
        shuffle: Optional[bool] = None
    ):
        """
        Create a DataLoader for specified split with data table construction.
        
        Args:
            split: 'train', 'val', or 'test'
            sampling_strategy: 'random', 'edge_weighted', or 'uniform'
            edge_bias: For edge_weighted strategy, probability weight for edges
            datasets_to_use: List of dataset names to use (None = use all)
            shuffle: Whether to shuffle (None = auto based on split)
            
        Returns:
            PyTorch DataLoader with batched data tables and Fourier encoding
        """
        # Auto-determine shuffle if not specified
        if shuffle is None:
            shuffle = (split == 'train')
        
        # Adjust sampling strategy for validation/test
        if split in ['val', 'test'] and sampling_strategy == 'random':
            # Use uniform sampling for evaluation for consistent results
            sampling_strategy = 'uniform'
            print(f"  Using uniform sampling for {split} split (consistent evaluation)")
        
        # Create dataset
        dataset = ImplicitRepresentationDataset(
            registry_path=self.registry_path,
            split=split,
            target_size=self.target_size,
            samples_per_image=self.samples_per_image,
            sampling_strategy=sampling_strategy,
            edge_bias=edge_bias,
            datasets_to_use=datasets_to_use,
            use_fourier_encoding=self.use_fourier_encoding,
            fourier_mapping_size=self.fourier_mapping_size,
            fourier_scale=self.fourier_scale,
            include_original_coords=self.include_original_coords
        )
        
        # Create DataLoader with GPU optimization
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False,
            prefetch_factor=2 if self.num_workers > 0 else None,
            collate_fn=custom_collate_fn
        )
        
        # Print dataset distribution
        print(f"\n{split.upper()} SPLIT - Dataset Distribution:")
        distribution = dataset.get_dataset_distribution()
        for dataset_name, count in distribution.items():
            print(f"  {dataset_name}: {count} pairs ({count/len(dataset)*100:.1f}%)")
        
        return loader
    
    def get_all_loaders(
        self,
        train_strategy: str = 'random',
        val_test_strategy: str = 'uniform',
        edge_bias: float = 0.7,
        datasets_to_use: Optional[List[str]] = None
    ):
        """
        Get train, val, and test loaders with appropriate sampling strategies.
        
        Args:
            train_strategy: Sampling strategy for training ('random' or 'edge_weighted')
            val_test_strategy: Sampling strategy for val/test (typically 'uniform')
            edge_bias: For edge_weighted strategy
            datasets_to_use: List of dataset names to use (None = use all)
            
        Returns:
            Dictionary with 'train', 'val', 'test' loaders
        """
        print("\n" + "="*70)
        print("CREATING DATA LOADERS WITH FOURIER FEATURE ENCODING")
        print("="*70)
        print(f"Fourier encoding: {self.use_fourier_encoding}")
        if self.use_fourier_encoding:
            print(f"  Mapping size: {self.fourier_mapping_size}")
            print(f"  Scale: {self.fourier_scale}")
            print(f"  Include original coords: {self.include_original_coords}")
        
        loaders = {
            'train': self.get_loader(
                'train',
                sampling_strategy=train_strategy,
                edge_bias=edge_bias,
                datasets_to_use=datasets_to_use
            ),
            'val': self.get_loader(
                'val',
                sampling_strategy=val_test_strategy,
                datasets_to_use=datasets_to_use
            ),
            'test': self.get_loader(
                'test',
                sampling_strategy=val_test_strategy,
                datasets_to_use=datasets_to_use
            )
        }
        
        return loaders


if __name__ == "__main__":
    # Test the data table construction with Fourier encoding
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    from pathlib import Path
    
    print("\n" + "=" * 70)
    print("TESTING DATA TABLE CONSTRUCTION WITH FOURIER ENCODING")
    print("=" * 70)
    
    # Paths
    registry_path = Path("data/registry/dataset_registry.json")
    
    if not registry_path.exists():
        print(f"❌ Registry file not found at {registry_path}")
        print("Please run organize_data.py first!")
        sys.exit(1)
    
    # Create data loader with Fourier encoding
    data_loader = ImplicitRepresentationDataLoader(
        registry_path=registry_path,
        batch_size=4,
        num_workers=0,
        target_size=256,
        samples_per_image=2048,
        use_fourier_encoding=True,
        fourier_mapping_size=128,
        fourier_scale=10.0,
        include_original_coords=False
    )
    
    # Get train loader
    print("\nLoading train data with Fourier encoding...")
    train_loader = data_loader.get_loader('train', sampling_strategy='random')
    
    # Load first batch
    print("\nLoading first batch...")
    batch = next(iter(train_loader))
    
    # Move to GPU
    batch['coordinates'] = batch['coordinates'].to(device)
    batch['encoded_features'] = batch['encoded_features'].to(device)
    batch['degraded_rgb'] = batch['degraded_rgb'].to(device)
    batch['reference_rgb'] = batch['reference_rgb'].to(device)
    
    # Print batch structure
    print(f"\n✓ Batch loaded successfully:")
    print(f"  Original coordinates shape: {batch['coordinates'].shape}")
    print(f"  Encoded features shape: {batch['encoded_features'].shape}")
    print(f"  Degraded RGB shape: {batch['degraded_rgb'].shape}")
    print(f"  Reference RGB shape: {batch['reference_rgb'].shape}")
    print(f"  Device: {batch['coordinates'].device}")
    
    print(f"\n  Coordinate range: [{batch['coordinates'].min().item():.3f}, {batch['coordinates'].max().item():.3f}]")
    print(f"  Encoded features range: [{batch['encoded_features'].min().item():.3f}, {batch['encoded_features'].max().item():.3f}]")
    print(f"  Degraded RGB range: [{batch['degraded_rgb'].min().item():.3f}, {batch['degraded_rgb'].max().item():.3f}]")
    
    print("\n" + "=" * 70)
    print("✓ FOURIER ENCODING INTEGRATION TEST COMPLETED!")
    print("=" * 70)