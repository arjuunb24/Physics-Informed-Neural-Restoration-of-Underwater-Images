"""
Fourier Feature Encoding Module for Implicit Neural Representations
Transforms low-dimensional coordinates to high-dimensional frequency space
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple


class FourierFeatureEncoding(nn.Module):
    """
    Fourier Feature Mapping for coordinate encoding.
    
    Transforms 2D coordinates (x, y) to high-dimensional frequency space
    using random Fourier features to help networks learn high-frequency details.
    
    Based on: "Fourier Features Let Networks Learn High Frequency Functions"
    (Tancik et al., NeurIPS 2020)
    """
    
    def __init__(
        self,
        input_dim: int = 2,
        mapping_size: int = 128,
        scale: float = 10.0,
        device: torch.device = torch.device('cpu')
    ):
        """
        Args:
            input_dim: Input coordinate dimension (2 for x,y)
            mapping_size: Number of frequency components (output will be 2*mapping_size)
            scale: Frequency scale (sigma in Gaussian distribution)
            device: Device to place the frequency matrix
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.mapping_size = mapping_size
        self.scale = scale
        self.output_dim = 2 * mapping_size  # sin and cos for each frequency
        
        # Sample frequency matrix B from Gaussian distribution
        # Shape: [input_dim, mapping_size] = [2, 128]
        # B is fixed during training (not learnable)
        B = torch.randn(input_dim, mapping_size, device=device) * scale
        self.register_buffer('B', B)
        
        print(f"FourierFeatureEncoding initialized:")
        print(f"  Input dim: {input_dim}")
        print(f"  Mapping size: {mapping_size}")
        print(f"  Output dim: {self.output_dim}")
        print(f"  Frequency scale (σ): {scale}")
        print(f"  Frequency matrix B shape: {self.B.shape}")
        print(f"  Device: {device}")
    
    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        """
        Apply Fourier feature encoding to coordinates.
        
        Args:
            coordinates: Input coordinates of shape [..., input_dim]
                        Typically [B, N, 2] for batched samples
        
        Returns:
            Encoded features of shape [..., 2*mapping_size]
            Typically [B, N, 256] for mapping_size=128
        """
        # coordinates shape: [B, N, 2] or [N, 2]
        # B (frequency matrix) shape: [2, 128]
        
        # Matrix multiply: [..., 2] @ [2, 128] -> [..., 128]
        x_proj = torch.matmul(coordinates, self.B)
        
        # Apply sin and cos to get Fourier features
        # Concatenate: [..., 128] + [..., 128] -> [..., 256]
        encoded = torch.cat([torch.sin(2 * np.pi * x_proj), 
                            torch.cos(2 * np.pi * x_proj)], dim=-1)
        
        return encoded
    
    def encode_with_coordinates(
        self, 
        coordinates: torch.Tensor,
        include_original: bool = False
    ) -> torch.Tensor:
        """
        Encode coordinates and optionally concatenate with original coordinates.
        
        Args:
            coordinates: Input coordinates [..., input_dim]
            include_original: If True, concatenate original coords with encoding
        
        Returns:
            Encoded features [..., 2*mapping_size] or [..., 2*mapping_size + input_dim]
        """
        encoded = self.forward(coordinates)
        
        if include_original:
            # Concatenate original coordinates: [..., 256] + [..., 2] -> [..., 258]
            encoded = torch.cat([encoded, coordinates], dim=-1)
        
        return encoded
    
    def get_frequency_statistics(self) -> dict:
        """
        Get statistics about the frequency matrix.
        
        Returns:
            Dictionary with frequency matrix statistics
        """
        B_np = self.B.cpu().numpy()
        
        return {
            'shape': self.B.shape,
            'mean': float(B_np.mean()),
            'std': float(B_np.std()),
            'min': float(B_np.min()),
            'max': float(B_np.max()),
            'scale': self.scale,
            'output_dim': self.output_dim
        }


class IdentityEncoding(nn.Module):
    """
    Identity encoding (no transformation) for baseline comparison.
    """
    
    def __init__(self, input_dim: int = 2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = input_dim
        print(f"IdentityEncoding initialized (no transformation)")
    
    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        """Return coordinates unchanged."""
        return coordinates
    
    def encode_with_coordinates(
        self, 
        coordinates: torch.Tensor,
        include_original: bool = False
    ) -> torch.Tensor:
        """Return coordinates unchanged (include_original has no effect)."""
        return coordinates


def create_encoding(
    encoding_type: str = 'fourier',
    input_dim: int = 2,
    mapping_size: int = 128,
    scale: float = 10.0,
    device: torch.device = torch.device('cpu')
) -> nn.Module:
    """
    Factory function to create coordinate encoding module.
    
    Args:
        encoding_type: 'fourier' or 'identity'
        input_dim: Input coordinate dimension
        mapping_size: Number of frequency components (for Fourier)
        scale: Frequency scale (for Fourier)
        device: Device for frequency matrix
    
    Returns:
        Encoding module (FourierFeatureEncoding or IdentityEncoding)
    """
    if encoding_type == 'fourier':
        return FourierFeatureEncoding(
            input_dim=input_dim,
            mapping_size=mapping_size,
            scale=scale,
            device=device
        )
    elif encoding_type == 'identity':
        return IdentityEncoding(input_dim=input_dim)
    else:
        raise ValueError(f"Unknown encoding type: {encoding_type}")


if __name__ == "__main__":
    # Test Fourier encoding
    print("\n" + "="*70)
    print("TESTING FOURIER FEATURE ENCODING")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Create encoding module
    encoder = FourierFeatureEncoding(
        input_dim=2,
        mapping_size=128,
        scale=10.0,
        device=device
    )
    
    # Test with sample coordinates
    print("\n" + "-"*70)
    print("Test 1: Single batch encoding")
    print("-"*70)
    
    # Simulate batch of coordinates [B=4, N=2048, 2]
    batch_size = 4
    samples_per_image = 2048
    coords = torch.randn(batch_size, samples_per_image, 2, device=device)
    coords = coords * 2 - 1  # Normalize to [-1, 1]
    
    print(f"Input coordinates shape: {coords.shape}")
    print(f"Input range: [{coords.min().item():.3f}, {coords.max().item():.3f}]")
    
    # Encode
    encoded = encoder(coords)
    
    print(f"\nEncoded features shape: {encoded.shape}")
    print(f"Encoded range: [{encoded.min().item():.3f}, {encoded.max().item():.3f}]")
    print(f"Encoded mean: {encoded.mean().item():.3f}")
    print(f"Encoded std: {encoded.std().item():.3f}")
    
    # Test with original coordinates concatenated
    print("\n" + "-"*70)
    print("Test 2: Encoding with original coordinates concatenated")
    print("-"*70)
    
    encoded_with_coords = encoder.encode_with_coordinates(coords, include_original=True)
    print(f"Encoded + original shape: {encoded_with_coords.shape}")
    
    # Frequency statistics
    print("\n" + "-"*70)
    print("Test 3: Frequency matrix statistics")
    print("-"*70)
    
    stats = encoder.get_frequency_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test different scales
    print("\n" + "-"*70)
    print("Test 4: Different frequency scales")
    print("-"*70)
    
    scales = [1.0, 5.0, 10.0, 20.0]
    sample_coord = torch.tensor([[0.5, 0.5]], device=device)
    
    for scale in scales:
        encoder_temp = FourierFeatureEncoding(scale=scale, device=device)
        encoded_temp = encoder_temp(sample_coord)
        print(f"\nScale {scale:5.1f}: range=[{encoded_temp.min().item():6.3f}, {encoded_temp.max().item():6.3f}], "
              f"std={encoded_temp.std().item():.3f}")
    
    print("\n" + "="*70)
    print("✓ FOURIER ENCODING TEST COMPLETED SUCCESSFULLY!")
    print("="*70)