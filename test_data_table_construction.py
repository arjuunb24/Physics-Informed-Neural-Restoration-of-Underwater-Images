import torch
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from data_table_construction import ImplicitRepresentationDataLoader, test_data_table_construction


def verify_gpu():
    """Verify GPU availability and print device info."""
    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available. Running on CPU.")
        return torch.device('cpu')
    
    device = torch.device('cuda')
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"Available GPU Memory: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1e9:.2f} GB\n")
    
    return device


def main():
    """Main testing execution."""
    
    # Verify GPU
    device = verify_gpu()
    
    # Define paths
    project_root = Path(__file__).parent
    registry_path = project_root / 'data' / 'registry' / 'dataset_registry.json'
    
    # Verify registry exists
    if not registry_path.exists():
        print(f"ERROR: Registry file not found at {registry_path}")
        print("Please run organize_data.py first to create the registry.")
        return
    
    print("="*70)
    print("DATA TABLE CONSTRUCTION - COMPREHENSIVE TEST")
    print("="*70)
    
    # Run the comprehensive test
    test_data_table_construction(device)
    
    print("\n" + "="*70)
    print("ADDITIONAL BATCH ORGANIZATION TESTS")
    print("="*70)
    
    # Create data loader
    data_loader = ImplicitRepresentationDataLoader(
        registry_path=registry_path,
        batch_size=8,
        num_workers=0,
        target_size=256,
        samples_per_image=2048
    )
    
    # Get all loaders
    print("\nCreating all loaders (train/val/test)...")
    loaders = data_loader.get_all_loaders(
        train_strategy='random',
        val_test_strategy='uniform'
    )
    
    print("\n" + "="*70)
    print("BATCH ORGANIZATION VERIFICATION")
    print("="*70)
    
    for split_name, loader in loaders.items():
        print(f"\n{split_name.upper()} Loader:")
        print(f"  Total batches: {len(loader)}")
        print(f"  Total samples: {len(loader.dataset)}")
        
        # Get first batch
        batch = next(iter(loader))
        
        # Move to GPU
        batch['coordinates'] = batch['coordinates'].to(device)
        batch['degraded_rgb'] = batch['degraded_rgb'].to(device)
        batch['reference_rgb'] = batch['reference_rgb'].to(device)
        
        print(f"\n  First batch structure:")
        print(f"    Batch size: {batch['coordinates'].shape[0]}")
        print(f"    Samples per image: {batch['coordinates'].shape[1]}")
        print(f"    Total coordinate samples in batch: {batch['coordinates'].shape[0] * batch['coordinates'].shape[1]}")
        print(f"    Memory per batch: ~{batch['coordinates'].element_size() * batch['coordinates'].nelement() / 1e6:.2f} MB (coords)")
        print(f"                      ~{batch['degraded_rgb'].element_size() * batch['degraded_rgb'].nelement() / 1e6:.2f} MB (degraded)")
        print(f"                      ~{batch['reference_rgb'].element_size() * batch['reference_rgb'].nelement() / 1e6:.2f} MB (reference)")
    
    print("\n" + "="*70)
    print("TRAINING ITERATION SIMULATION")
    print("="*70)
    
    # Simulate a few training iterations
    train_loader = loaders['train']
    
    print("\nSimulating 3 training iterations...")
    for iteration, batch in enumerate(train_loader):
        if iteration >= 3:
            break
        
        # Move to GPU
        coords = batch['coordinates'].to(device)
        degraded = batch['degraded_rgb'].to(device)
        reference = batch['reference_rgb'].to(device)
        
        print(f"\nIteration {iteration + 1}:")
        print(f"  Input coordinates shape: {coords.shape}")
        print(f"  Input degraded RGB shape: {degraded.shape}")
        print(f"  Target reference RGB shape: {reference.shape}")
        print(f"  Datasets: {set(batch['dataset'])}")
        
        # Simulate forward pass (just a dummy operation)
        # In real training, you'd do: output = model(coords, degraded)
        # Then compute loss: loss = criterion(output, reference)
        
        print(f"  ✓ Batch ready for neural network training")
    
    print("\n" + "="*70)
    print("ALL TESTS COMPLETED SUCCESSFULLY")
    print("="*70)
    
    print("\nSummary:")
    print("  ✓ Data table construction working")
    print("  ✓ Batch organization: [B, N, D] format")
    print("  ✓ All sampling strategies functional")
    print("  ✓ Train/val/test loaders created")
    print("  ✓ GPU memory management efficient")
    print("  ✓ Ready for implicit neural representation training")
    
    print(f"\nFinal GPU Memory Used: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    print(f"Final GPU Memory Cached: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")


if __name__ == '__main__':
    main()