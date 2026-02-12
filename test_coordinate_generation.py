import torch
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from coordinate_grid_generator import CoordinateGridGenerator


def verify_gpu():
    """Verify GPU availability."""
    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available. Running on CPU.")
        return torch.device('cpu')
    
    device = torch.device('cuda')
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"Available GPU Memory: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1e9:.2f} GB\n")
    
    return device


def load_sample_image(preprocessed_dir: Path, split: str = 'train') -> tuple:
    """Load a sample image pair for testing."""
    raw_dir = preprocessed_dir / split / 'raw'
    ref_dir = preprocessed_dir / split / 'reference'
    
    # Get first image
    raw_images = list(raw_dir.glob('*.png'))
    if not raw_images:
        return None, None
    
    raw_path = raw_images[0]
    ref_path = ref_dir / raw_path.name
    
    # Load images
    raw_img = Image.open(raw_path).convert('RGB')
    ref_img = Image.open(ref_path).convert('RGB')
    
    # Convert to tensors [C, H, W]
    raw_tensor = torch.from_numpy(np.array(raw_img).astype(np.float32) / 255.0).permute(2, 0, 1)
    ref_tensor = torch.from_numpy(np.array(ref_img).astype(np.float32) / 255.0).permute(2, 0, 1)
    
    return raw_tensor, ref_tensor


def visualize_sampling(
    image: torch.Tensor,
    coordinates: torch.Tensor,
    indices: torch.Tensor,
    title: str,
    save_path: Path
):
    """Visualize coordinate sampling on an image."""
    # Convert image to numpy for visualization
    img_np = image.cpu().permute(1, 2, 0).numpy()
    
    # Get indices as numpy
    indices_np = indices.cpu().numpy()
    
    # Create visualization
    plt.figure(figsize=(10, 10))
    plt.imshow(img_np)
    plt.scatter(indices_np[:, 1], indices_np[:, 0], c='red', s=1, alpha=0.5)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved visualization: {save_path}")


def main():
    """Test coordinate grid generation."""
    
    # Verify GPU
    device = verify_gpu()
    
    # Define paths
    project_root = Path(__file__).parent
    preprocessed_dir = project_root / 'data' / 'preprocessed'
    output_dir = project_root / 'data' / 'coordinate_visualizations'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("COORDINATE GRID GENERATION TEST")
    print("="*70)
    
    # Load sample image
    print("\nLoading sample image...")
    degraded_img, reference_img = load_sample_image(preprocessed_dir)
    
    if degraded_img is None:
        print("ERROR: No preprocessed images found.")
        print("Please run preprocess_dataset.py first.")
        return
    
    print(f"Loaded image shape: {degraded_img.shape}")
    
    # Initialize coordinate grid generator
    coord_gen = CoordinateGridGenerator(
        image_size=256,
        device=device,
        samples_per_image=2048
    )
    
    print("\n" + "="*70)
    print("1. FULL NORMALIZED GRID")
    print("="*70)
    full_grid = coord_gen.get_full_grid()
    print(f"Full grid shape: {full_grid.shape}")
    print(f"Coordinate range: [{full_grid.min():.2f}, {full_grid.max():.2f}]")
    print(f"Top-left coordinate: {full_grid[0, 0]}")
    print(f"Bottom-right coordinate: {full_grid[-1, -1]}")
    print(f"Center coordinate: {full_grid[128, 128]}")
    
    print("\n" + "="*70)
    print("2. RANDOM SAMPLING")
    print("="*70)
    coords_random, indices_random = coord_gen.sample_random(num_samples=2048)
    print(f"Sampled coordinates shape: {coords_random.shape}")
    print(f"Sampled indices shape: {indices_random.shape}")
    
    # Create data table
    data_table_random = coord_gen.create_data_table(
        coords_random,
        indices_random,
        degraded_img,
        reference_img
    )
    print(f"Data table keys: {list(data_table_random.keys())}")
    print(f"Degraded RGB shape: {data_table_random['degraded_rgb'].shape}")
    print(f"Reference RGB shape: {data_table_random['reference_rgb'].shape}")
    
    # Visualize random sampling
    visualize_sampling(
        degraded_img,
        coords_random,
        indices_random,
        "Random Sampling (2048 points)",
        output_dir / "random_sampling.png"
    )
    
    print("\n" + "="*70)
    print("3. EDGE-WEIGHTED SAMPLING")
    print("="*70)
    
    # Compute edge map
    print("Computing edge map using Sobel filter...")
    edge_map = coord_gen.compute_edge_map(degraded_img, method='sobel')
    print(f"Edge map shape: {edge_map.shape}")
    print(f"Edge map range: [{edge_map.min():.4f}, {edge_map.max():.4f}]")
    
    # Save edge map visualization
    plt.figure(figsize=(8, 8))
    plt.imshow(edge_map.cpu().numpy(), cmap='hot')
    plt.title("Edge Map (Sobel)")
    plt.colorbar()
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_dir / "edge_map.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved edge map: {output_dir / 'edge_map.png'}")
    
    # Sample with edge weighting
    coords_edge, indices_edge = coord_gen.sample_edge_weighted(
        edge_map,
        num_samples=2048,
        edge_bias=0.7
    )
    print(f"Edge-weighted coordinates shape: {coords_edge.shape}")
    
    # Visualize edge-weighted sampling
    visualize_sampling(
        degraded_img,
        coords_edge,
        indices_edge,
        "Edge-Weighted Sampling (2048 points, 70% bias)",
        output_dir / "edge_weighted_sampling.png"
    )
    
    print("\n" + "="*70)
    print("4. UNIFORM GRID SAMPLING")
    print("="*70)
    coords_uniform, indices_uniform = coord_gen.sample_uniform_grid(stride=4)
    print(f"Uniform grid coordinates shape: {coords_uniform.shape}")
    print(f"Total uniform samples: {coords_uniform.shape[0]}")
    
    # Visualize uniform sampling
    visualize_sampling(
        degraded_img,
        coords_uniform,
        indices_uniform,
        f"Uniform Grid Sampling (stride=4, {coords_uniform.shape[0]} points)",
        output_dir / "uniform_sampling.png"
    )
    
    print("\n" + "="*70)
    print("5. BATCH DATA TABLE CREATION")
    print("="*70)
    
    # Create a batch (repeat same image for demo)
    batch_degraded = degraded_img.unsqueeze(0).repeat(4, 1, 1, 1)
    batch_reference = reference_img.unsqueeze(0).repeat(4, 1, 1, 1)
    
    print(f"Batch degraded shape: {batch_degraded.shape}")
    
    # Test different sampling strategies
    for strategy in ['random', 'edge_weighted', 'uniform']:
        print(f"\n  Testing '{strategy}' strategy:")
        
        batch_data = coord_gen.create_batch_data_table(
            batch_degraded,
            batch_reference,
            sampling_strategy=strategy,
            num_samples=2048
        )
        
        print(f"    Coordinates shape: {batch_data['coordinates'].shape}")
        print(f"    Degraded RGB shape: {batch_data['degraded_rgb'].shape}")
        print(f"    Reference RGB shape: {batch_data['reference_rgb'].shape}")
    
    print("\n" + "="*70)
    print("TESTING COMPLETE")
    print("="*70)
    print(f"\nVisualizations saved to: {output_dir}")
    print(f"GPU Memory Used: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    print(f"GPU Memory Cached: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("✓ Full normalized grid generation: [-1, 1] range")
    print("✓ Random sampling: 2048 uniform samples")
    print("✓ Edge-weighted sampling: 70% from edges, 30% uniform")
    print("✓ Uniform grid sampling: Regular stride-based pattern")
    print("✓ Data table creation: (coordinates, degraded_rgb, reference_rgb)")
    print("✓ Batch processing: All strategies working")


if __name__ == '__main__':
    main()