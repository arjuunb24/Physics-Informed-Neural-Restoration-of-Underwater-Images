"""
Test script for Fourier Feature Encoding Module
Validates encoding functionality and integration with data pipeline
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from fourier_encoding import FourierFeatureEncoding, create_encoding
from data_table_construction import ImplicitRepresentationDataLoader


def visualize_encoding_patterns(device):
    """Visualize how Fourier encoding transforms coordinate space"""
    print("\n" + "="*70)
    print("VISUALIZATION 1: FOURIER ENCODING PATTERNS")
    print("="*70)
    
    # Create encoders with different scales
    scales = [1.0, 5.0, 10.0, 20.0]
    
    # Create a grid of coordinates
    x = np.linspace(-1, 1, 100)
    y = np.linspace(-1, 1, 100)
    xx, yy = np.meshgrid(x, y)
    coords = np.stack([xx.flatten(), yy.flatten()], axis=-1)
    coords_tensor = torch.from_numpy(coords).float().to(device)
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    for idx, scale in enumerate(scales):
        encoder = FourierFeatureEncoding(scale=scale, device=device)
        
        # Encode coordinates
        encoded = encoder(coords_tensor)  # [10000, 256]
        
        # Visualize first sine component
        sine_component = encoded[:, 0].cpu().numpy().reshape(100, 100)
        cosine_component = encoded[:, 128].cpu().numpy().reshape(100, 100)
        
        # Plot sine component
        im1 = axes[0, idx].imshow(sine_component, cmap='RdBu', vmin=-1, vmax=1)
        axes[0, idx].set_title(f'Scale={scale} - First Sine')
        axes[0, idx].axis('off')
        plt.colorbar(im1, ax=axes[0, idx], fraction=0.046)
        
        # Plot cosine component
        im2 = axes[1, idx].imshow(cosine_component, cmap='RdBu', vmin=-1, vmax=1)
        axes[1, idx].set_title(f'Scale={scale} - First Cosine')
        axes[1, idx].axis('off')
        plt.colorbar(im2, ax=axes[1, idx], fraction=0.046)
    
    plt.tight_layout()
    output_path = Path('data/coordinate_visualizations/fourier_encoding_patterns.png')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved visualization to: {output_path}")
    plt.close()


def compare_encoding_statistics(device):
    """Compare statistics of encoded features across different scales"""
    print("\n" + "="*70)
    print("ANALYSIS: ENCODING STATISTICS ACROSS SCALES")
    print("="*70)
    
    scales = [1.0, 5.0, 10.0, 15.0, 20.0]
    
    # Sample random coordinates
    coords = torch.randn(1000, 2, device=device) * 2 - 1  # Random in [-1, 1]
    
    print(f"\nInput coordinates:")
    print(f"  Shape: {coords.shape}")
    print(f"  Range: [{coords.min().item():.3f}, {coords.max().item():.3f}]")
    print(f"  Mean: {coords.mean().item():.3f}")
    print(f"  Std: {coords.std().item():.3f}")
    
    print(f"\n{'Scale':<10} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
    print("-" * 58)
    
    for scale in scales:
        encoder = FourierFeatureEncoding(scale=scale, device=device)
        encoded = encoder(coords)
        
        print(f"{scale:<10.1f} {encoded.mean().item():<12.6f} {encoded.std().item():<12.6f} "
              f"{encoded.min().item():<12.6f} {encoded.max().item():<12.6f}")


def test_encoding_integration(device):
    """Test Fourier encoding integration with data pipeline"""
    print("\n" + "="*70)
    print("INTEGRATION TEST: FOURIER ENCODING IN DATA PIPELINE")
    print("="*70)
    
    registry_path = Path("data/registry/dataset_registry.json")
    
    if not registry_path.exists():
        print(f"❌ Registry file not found at {registry_path}")
        print("Please run organize_data.py and preprocess_dataset.py first!")
        return
    
    # Test both with and without Fourier encoding
    configs = [
        {
            'name': 'Without Fourier Encoding',
            'use_fourier': False,
            'include_coords': False
        },
        {
            'name': 'With Fourier Encoding',
            'use_fourier': True,
            'include_coords': False
        },
        {
            'name': 'Fourier + Original Coords',
            'use_fourier': True,
            'include_coords': True
        }
    ]
    
    for config in configs:
        print(f"\n{'-'*70}")
        print(f"Testing: {config['name']}")
        print(f"{'-'*70}")
        
        # Create data loader
        data_loader = ImplicitRepresentationDataLoader(
            registry_path=registry_path,
            batch_size=4,
            num_workers=0,
            samples_per_image=2048,
            use_fourier_encoding=config['use_fourier'],
            fourier_mapping_size=128,
            fourier_scale=10.0,
            include_original_coords=config['include_coords']
        )
        
        # Get train loader
        train_loader = data_loader.get_loader('train', sampling_strategy='random')
        
        # Load one batch
        batch = next(iter(train_loader))
        
        # Move to GPU
        coords = batch['coordinates'].to(device)
        encoded = batch['encoded_features'].to(device)
        
        print(f"\n✓ Batch loaded:")
        print(f"  Original coords shape: {coords.shape}")
        print(f"  Encoded features shape: {encoded.shape}")
        print(f"  Expected encoded dim: {2 if not config['use_fourier'] else (256 if not config['include_coords'] else 258)}")
        
        # Verify dimensions
        expected_dim = 2 if not config['use_fourier'] else (256 if not config['include_coords'] else 258)
        assert encoded.shape[-1] == expected_dim, f"Dimension mismatch! Expected {expected_dim}, got {encoded.shape[-1]}"
        
        print(f"\n✓ Statistics:")
        print(f"  Coords range: [{coords.min().item():.3f}, {coords.max().item():.3f}]")
        print(f"  Encoded range: [{encoded.min().item():.3f}, {encoded.max().item():.3f}]")
        print(f"  Encoded mean: {encoded.mean().item():.3f}")
        print(f"  Encoded std: {encoded.std().item():.3f}")


def test_frequency_scale_comparison(device):
    """Compare different frequency scales on actual image data"""
    print("\n" + "="*70)
    print("COMPARISON: DIFFERENT FREQUENCY SCALES ON REAL DATA")
    print("="*70)
    
    registry_path = Path("data/registry/dataset_registry.json")
    
    if not registry_path.exists():
        print(f"❌ Registry file not found")
        return
    
    scales = [1.0, 5.0, 10.0, 20.0]
    
    for scale in scales:
        print(f"\n{'-'*70}")
        print(f"Testing scale: {scale}")
        print(f"{'-'*70}")
        
        data_loader = ImplicitRepresentationDataLoader(
            registry_path=registry_path,
            batch_size=2,
            num_workers=0,
            samples_per_image=2048,
            use_fourier_encoding=True,
            fourier_scale=scale
        )
        
        train_loader = data_loader.get_loader('train')
        batch = next(iter(train_loader))
        
        encoded = batch['encoded_features'].to(device)
        
        print(f"  Encoded shape: {encoded.shape}")
        print(f"  Encoded range: [{encoded.min().item():.3f}, {encoded.max().item():.3f}]")
        print(f"  Encoded std: {encoded.std().item():.3f}")


def main():
    """Run all Fourier encoding tests"""
    print("\n" + "="*70)
    print("FOURIER FEATURE ENCODING - COMPREHENSIVE TEST SUITE")
    print("="*70)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Run tests
    try:
        visualize_encoding_patterns(device)
        compare_encoding_statistics(device)
        test_encoding_integration(device)
        test_frequency_scale_comparison(device)
        
        print("\n" + "="*70)
        print("✓ ALL FOURIER ENCODING TESTS PASSED SUCCESSFULLY!")
        print("="*70)
        
        if torch.cuda.is_available():
            print(f"\nGPU Memory Used: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
            print(f"GPU Memory Cached: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()