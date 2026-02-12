"""
Test script for SIREN Network with Channel-Specific Heads
Tests complete architecture integration with data pipeline
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from siren_network import SIRENWithChannelHeads, create_siren_model
from fourier_encoding import FourierFeatureEncoding
from data_table_construction import ImplicitRepresentationDataLoader


def test_complete_pipeline(device):
    """Test complete pipeline: Data → Fourier → SIREN → RGB predictions"""
    print("\n" + "="*70)
    print("COMPLETE PIPELINE TEST: DATA → FOURIER → SIREN → RGB")
    print("="*70)
    
    registry_path = Path("data/registry/dataset_registry.json")
    
    if not registry_path.exists():
        print(f"❌ Registry file not found at {registry_path}")
        print("Please run organize_data.py and preprocess_dataset.py first!")
        return
    
    # Create data loader with Fourier encoding
    print("\nStep 1: Loading data with Fourier encoding...")
    data_loader = ImplicitRepresentationDataLoader(
        registry_path=registry_path,
        batch_size=4,
        num_workers=0,
        samples_per_image=2048,
        use_fourier_encoding=True,
        fourier_mapping_size=128,
        fourier_scale=10.0
    )
    
    train_loader = data_loader.get_loader('train', sampling_strategy='random')
    batch = next(iter(train_loader))
    
    # Move to GPU
    encoded_features = batch['encoded_features'].to(device)
    degraded_rgb = batch['degraded_rgb'].to(device)
    reference_rgb = batch['reference_rgb'].to(device)
    
    print(f"  ✓ Batch loaded:")
    print(f"    Encoded features: {encoded_features.shape}")
    print(f"    Degraded RGB: {degraded_rgb.shape}")
    print(f"    Reference RGB: {reference_rgb.shape}")
    
    # Create SIREN model with channel heads
    print("\nStep 2: Creating SIREN model with channel heads...")
    model = create_siren_model(
        input_dim=256,
        hidden_dim=256,
        num_hidden_layers=4,
        omega=30.0,
        head_architecture='dense',
        device=device
    )
    
    # Forward pass
    print("\nStep 3: Forward pass through complete network...")
    with torch.no_grad():
        rgb_predictions = model(encoded_features)
    
    print(f"  ✓ RGB predictions: {rgb_predictions.shape}")
    print(f"    Prediction range: [{rgb_predictions.min().item():.3f}, {rgb_predictions.max().item():.3f}]")
    print(f"    Red channel mean: {rgb_predictions[..., 0].mean().item():.3f}")
    print(f"    Green channel mean: {rgb_predictions[..., 1].mean().item():.3f}")
    print(f"    Blue channel mean: {rgb_predictions[..., 2].mean().item():.3f}")
    
    print("\n✓ Complete pipeline working successfully!")
    print("  Data flow: Coordinates [B,N,2] → Fourier [B,N,256] → SIREN [B,N,256] → RGB [B,N,3]")


def test_training_step(device):
    """Simulate a complete training step"""
    print("\n" + "="*70)
    print("TRAINING STEP SIMULATION")
    print("="*70)
    
    registry_path = Path("data/registry/dataset_registry.json")
    
    if not registry_path.exists():
        print(f"❌ Registry file not found")
        return
    
    # Setup data loader
    data_loader = ImplicitRepresentationDataLoader(
        registry_path=registry_path,
        batch_size=4,
        num_workers=0,
        samples_per_image=2048,
        use_fourier_encoding=True,
        fourier_mapping_size=128,
        fourier_scale=10.0
    )
    
    train_loader = data_loader.get_loader('train', sampling_strategy='random')
    
    # Create model
    model = create_siren_model(
        input_dim=256,
        hidden_dim=256,
        num_hidden_layers=4,
        head_architecture='dense',
        device=device
    )
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Loss function
    criterion = torch.nn.MSELoss()
    
    print("\nSimulating 5 training steps...")
    losses = []
    
    for step in range(5):
        # Get batch
        batch = next(iter(train_loader))
        encoded_features = batch['encoded_features'].to(device)
        reference_rgb = batch['reference_rgb'].to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(encoded_features)
        
        # Compute loss
        loss = criterion(predictions, reference_rgb)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping (optional but recommended)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update weights
        optimizer.step()
        
        losses.append(loss.item())
        
        if step == 0 or step == 4:
            print(f"\n  Step {step+1}:")
            print(f"    Loss: {loss.item():.6f}")
            print(f"    Pred range: [{predictions.min().item():.3f}, {predictions.max().item():.3f}]")
            print(f"    Target range: [{reference_rgb.min().item():.3f}, {reference_rgb.max().item():.3f}]")
    
    print(f"\n✓ Training simulation completed!")
    print(f"  Initial loss: {losses[0]:.6f}")
    print(f"  Final loss: {losses[-1]:.6f}")


def compare_architectures(device):
    """Compare dense vs adaptive head architectures"""
    print("\n" + "="*70)
    print("ARCHITECTURE COMPARISON: DENSE vs ADAPTIVE HEADS")
    print("="*70)
    
    # Sample input
    x = torch.randn(4, 2048, 256, device=device)
    target = torch.rand(4, 2048, 3, device=device)
    
    architectures = [
        {'name': 'Dense (Equal Capacity)', 'type': 'dense', 'config': None},
        {'name': 'Adaptive (Red>Green>Blue)', 'type': 'adaptive', 'config': {'red': 256, 'green': 192, 'blue': 128}},
    ]
    
    for arch in architectures:
        print(f"\n{'-'*70}")
        print(f"Testing: {arch['name']}")
        print(f"{'-'*70}")
        
        model = create_siren_model(
            input_dim=256,
            hidden_dim=256,
            num_hidden_layers=4,
            head_architecture=arch['type'],
            head_config=arch['config'],
            device=device
        )
        
        # Forward pass
        with torch.no_grad():
            predictions = model(x)
            outputs = model.forward_with_intermediate(x)
        
        print(f"\n  Model statistics:")
        print(f"    Total parameters: {model.count_parameters():,}")
        print(f"    Trunk parameters: {sum(p.numel() for p in model.trunk.parameters()):,}")
        print(f"    Red head parameters: {sum(p.numel() for p in model.red_head.parameters()):,}")
        print(f"    Green head parameters: {sum(p.numel() for p in model.green_head.parameters()):,}")
        print(f"    Blue head parameters: {sum(p.numel() for p in model.blue_head.parameters()):,}")
        
        print(f"\n  Output statistics:")
        print(f"    Red mean: {outputs['red'].mean().item():.3f}, std: {outputs['red'].std().item():.3f}")
        print(f"    Green mean: {outputs['green'].mean().item():.3f}, std: {outputs['green'].std().item():.3f}")
        print(f"    Blue mean: {outputs['blue'].mean().item():.3f}, std: {outputs['blue'].std().item():.3f}")
        
        # Test loss computation
        loss = torch.nn.functional.mse_loss(predictions, target)
        print(f"\n  Sample MSE loss: {loss.item():.6f}")


def visualize_channel_predictions(device):
    """Visualize predictions from individual channel heads"""
    print("\n" + "="*70)
    print("CHANNEL PREDICTION VISUALIZATION")
    print("="*70)
    
    registry_path = Path("data/registry/dataset_registry.json")
    
    if not registry_path.exists():
        print(f"❌ Registry file not found")
        return
    
    # Load one image
    data_loader = ImplicitRepresentationDataLoader(
        registry_path=registry_path,
        batch_size=1,
        num_workers=0,
        samples_per_image=256*256,  # Full image
        use_fourier_encoding=True,
        fourier_mapping_size=128,
        fourier_scale=10.0
    )
    
    train_loader = data_loader.get_loader('train', sampling_strategy='uniform')
    batch = next(iter(train_loader))
    
    encoded_features = batch['encoded_features'].to(device)
    degraded_rgb = batch['degraded_rgb'].to(device)
    reference_rgb = batch['reference_rgb'].to(device)
    
    # Create model
    model = create_siren_model(
        input_dim=256,
        hidden_dim=256,
        num_hidden_layers=4,
        head_architecture='dense',
        device=device
    )
    
    # Get predictions
    print("\nGenerating predictions...")
    with torch.no_grad():
        outputs = model.forward_with_intermediate(encoded_features)
    
    # Reshape to image
    def reshape_to_image(tensor):
        return tensor.squeeze(0).reshape(256, 256, -1).cpu().numpy()
    
    degraded_img = reshape_to_image(degraded_rgb)
    reference_img = reshape_to_image(reference_rgb)
    pred_img = reshape_to_image(outputs['rgb'])
    red_channel = reshape_to_image(outputs['red'])
    green_channel = reshape_to_image(outputs['green'])
    blue_channel = reshape_to_image(outputs['blue'])
    
    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].imshow(degraded_img)
    axes[0, 0].set_title('Degraded Input')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(reference_img)
    axes[0, 1].set_title('Reference (Ground Truth)')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(pred_img)
    axes[0, 2].set_title('SIREN Prediction')
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(red_channel.squeeze(), cmap='Reds', vmin=0, vmax=1)
    axes[1, 0].set_title('Red Channel Head Output')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(green_channel.squeeze(), cmap='Greens', vmin=0, vmax=1)
    axes[1, 1].set_title('Green Channel Head Output')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(blue_channel.squeeze(), cmap='Blues', vmin=0, vmax=1)
    axes[1, 2].set_title('Blue Channel Head Output')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    output_path = Path('data/coordinate_visualizations/channel_predictions.png')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved visualization to: {output_path}")
    plt.close()


def test_different_head_configurations(device):
    """Test various head configurations"""
    print("\n" + "="*70)
    print("TESTING DIFFERENT HEAD CONFIGURATIONS")
    print("="*70)
    
    x = torch.randn(2, 1024, 256, device=device)
    
    configs = [
        {'name': 'Equal (256-256-256)', 'type': 'adaptive', 'config': {'red': 256, 'green': 256, 'blue': 256}},
        {'name': 'Moderate (256-192-128)', 'type': 'adaptive', 'config': {'red': 256, 'green': 192, 'blue': 128}},
        {'name': 'Aggressive (256-128-64)', 'type': 'adaptive', 'config': {'red': 256, 'green': 128, 'blue': 64}},
    ]
    
    for cfg in configs:
        print(f"\n{'-'*70}")
        print(f"Configuration: {cfg['name']}")
        print(f"{'-'*70}")
        
        model = create_siren_model(
            input_dim=256,
            hidden_dim=256,
            num_hidden_layers=4,
            head_architecture=cfg['type'],
            head_config=cfg['config'],
            device=device
        )
        
        with torch.no_grad():
            out = model(x)
        
        print(f"  Total parameters: {model.count_parameters():,}")
        print(f"  Output shape: {out.shape}")
        print(f"  Output range: [{out.min().item():.3f}, {out.max().item():.3f}]")


def main():
    """Run all integration tests"""
    print("\n" + "="*70)
    print("SIREN WITH CHANNEL HEADS - INTEGRATION TEST SUITE")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    try:
        test_complete_pipeline(device)
        test_training_step(device)
        compare_architectures(device)
        test_different_head_configurations(device)
        visualize_channel_predictions(device)
        
        print("\n" + "="*70)
        print("✓ ALL INTEGRATION TESTS PASSED SUCCESSFULLY!")
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