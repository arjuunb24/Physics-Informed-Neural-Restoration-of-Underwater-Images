"""
SIREN (Sinusoidal Representation Networks) Architecture
Implements sinusoidal activation networks for implicit neural representations
"""
import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional


class SineLayer(nn.Module):
    """
    Single layer with sine activation and SIREN-specific initialization.
    
    Implements: sin(ω * (Wx + b)) where ω is the frequency parameter
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        is_first: bool = False,
        omega: float = 30.0
    ):
        """
        Args:
            in_features: Input dimension
            out_features: Output dimension
            bias: Whether to use bias
            is_first: Whether this is the first layer (uses different initialization)
            omega: Frequency parameter for sine activation
        """
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.omega = omega
        self.is_first = is_first
        
        # Linear transformation
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        # SIREN-specific initialization
        self.init_weights()
    
    def init_weights(self):
        """
        SIREN-specific weight initialization.
        
        First layer: Uniform[-1/n_in, 1/n_in]
        Other layers: Uniform[-sqrt(6/n_in)/omega, sqrt(6/n_in)/omega]
        Bias: Zero initialization
        """
        with torch.no_grad():
            if self.is_first:
                # First layer initialization
                # U(-1/n_in, 1/n_in)
                bound = 1 / self.in_features
                self.linear.weight.uniform_(-bound, bound)
            else:
                # Subsequent layers initialization
                # U(-sqrt(6/n_in)/omega, sqrt(6/n_in)/omega)
                bound = np.sqrt(6 / self.in_features) / self.omega
                self.linear.weight.uniform_(-bound, bound)
            
            # Bias initialization: zero
            if self.linear.bias is not None:
                self.linear.bias.uniform_(-0.0, 0.0)  # Explicitly zero
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with sine activation.
        
        Args:
            x: Input tensor [..., in_features]
        
        Returns:
            Output tensor [..., out_features] after sin(omega * linear(x))
        """
        return torch.sin(self.omega * self.linear(x))
    
    def forward_with_intermediate(self, x: torch.Tensor):
        """
        Forward pass that returns both pre-activation and post-activation values.
        Useful for analysis and debugging.
        """
        linear_out = self.linear(x)
        activated = torch.sin(self.omega * linear_out)
        return activated, linear_out


class SIRENTrunk(nn.Module):
    """
    SIREN Trunk Network - Main feature extraction component.
    
    Implements a multi-layer fully connected network with sinusoidal activations
    and SIREN-specific initialization.
    """
    
    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 256,
        num_hidden_layers: int = 4,
        omega: float = 30.0,
        omega_first: Optional[float] = None
    ):
        """
        Args:
            input_dim: Input feature dimension (from Fourier encoding)
            hidden_dim: Hidden layer dimension
            num_hidden_layers: Number of hidden layers (excluding input layer)
            omega: Frequency parameter for sine activation
            omega_first: Frequency parameter for first layer (if None, uses omega)
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.omega = omega
        self.omega_first = omega_first if omega_first is not None else omega
        
        # Build network layers
        layers = []
        
        # First layer (special initialization)
        layers.append(
            SineLayer(
                in_features=input_dim,
                out_features=hidden_dim,
                is_first=True,
                omega=self.omega_first
            )
        )
        
        # Hidden layers
        for _ in range(num_hidden_layers):
            layers.append(
                SineLayer(
                    in_features=hidden_dim,
                    out_features=hidden_dim,
                    is_first=False,
                    omega=omega
                )
            )
        
        self.layers = nn.ModuleList(layers)
        
        print(f"SIRENTrunk initialized:")
        print(f"  Input dim: {input_dim}")
        print(f"  Hidden dim: {hidden_dim}")
        print(f"  Number of hidden layers: {num_hidden_layers}")
        print(f"  Total layers: {len(layers)}")
        print(f"  Omega (first layer): {self.omega_first}")
        print(f"  Omega (other layers): {omega}")
        print(f"  Output dim: {hidden_dim}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the SIREN trunk.
        
        Args:
            x: Input tensor [Batch, N, input_dim] - Fourier encoded features
        
        Returns:
            Shared features [Batch, N, hidden_dim]
        """
        # Pass through all layers
        for layer in self.layers:
            x = layer(x)
        
        return x
    
    def forward_with_activations(self, x: torch.Tensor) -> tuple:
        """
        Forward pass that returns intermediate activations for analysis.
        
        Returns:
            (final_output, list_of_activations)
        """
        activations = []
        
        for layer in self.layers:
            x = layer(x)
            activations.append(x)
        
        return x, activations
    
    def get_layer_statistics(self, x: torch.Tensor) -> dict:
        """
        Get statistics of activations at each layer for debugging.
        
        Args:
            x: Input tensor [Batch, N, input_dim]
        
        Returns:
            Dictionary with statistics for each layer
        """
        stats = {}
        
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            
            stats[f'layer_{idx}'] = {
                'mean': x.mean().item(),
                'std': x.std().item(),
                'min': x.min().item(),
                'max': x.max().item(),
                'shape': list(x.shape)
            }
        
        return stats


class SIRENNetwork(nn.Module):
    """
    Complete SIREN Network with trunk architecture.
    This is the main component before adding channel-specific heads.
    """
    
    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 256,
        num_hidden_layers: int = 4,
        omega: float = 30.0,
        omega_first: Optional[float] = None
    ):
        """
        Args:
            input_dim: Input feature dimension (from Fourier encoding)
            hidden_dim: Hidden layer dimension
            num_hidden_layers: Number of hidden layers
            omega: Frequency parameter for sine activation
            omega_first: Frequency parameter for first layer
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # SIREN trunk for feature extraction
        self.trunk = SIRENTrunk(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
            omega=omega,
            omega_first=omega_first
        )
        
        print(f"\nSIRENNetwork initialized:")
        print(f"  Total parameters: {self.count_parameters():,}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through SIREN network.
        
        Args:
            x: Input tensor [Batch, N, input_dim]
        
        Returns:
            Shared features [Batch, N, hidden_dim]
        """
        return self.trunk(x)
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_parameter_groups(self) -> List[dict]:
        """Get parameter groups for optimizer (useful for different learning rates)."""
        return [
            {
                'params': self.trunk.layers[0].parameters(),
                'name': 'first_layer'
            },
            {
                'params': [p for layer in self.trunk.layers[1:] for p in layer.parameters()],
                'name': 'hidden_layers'
            }
        ]

class ChannelHead(nn.Module):
    """
    Channel-specific prediction head for individual RGB channels.
    Uses sine activations for intermediate layers.
    """
    
    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 1,
        omega: float = 30.0,
        channel_name: str = 'channel'
    ):
        """
        Args:
            input_dim: Input dimension from SIREN trunk
            hidden_dim: Hidden dimension for the head
            num_layers: Number of hidden layers (1 or 2)
            omega: Frequency parameter for sine activation
            channel_name: Name of the channel (for logging)
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.channel_name = channel_name
        
        layers = []
        
        if num_layers == 1:
            # Single hidden layer: input_dim → hidden_dim → 1
            layers.append(
                SineLayer(
                    in_features=input_dim,
                    out_features=hidden_dim,
                    is_first=False,
                    omega=omega
                )
            )
            layers.append(nn.Linear(hidden_dim, 1))
            
        elif num_layers == 2:
            # Two hidden layers: input_dim → hidden_dim → hidden_dim → 1
            layers.append(
                SineLayer(
                    in_features=input_dim,
                    out_features=hidden_dim,
                    is_first=False,
                    omega=omega
                )
            )
            layers.append(
                SineLayer(
                    in_features=hidden_dim,
                    out_features=hidden_dim,
                    is_first=False,
                    omega=omega
                )
            )
            layers.append(nn.Linear(hidden_dim, 1))
        else:
            raise ValueError(f"num_layers must be 1 or 2, got {num_layers}")
        
        self.layers = nn.ModuleList(layers)
        
        # Initialize final linear layer with smaller weights
        with torch.no_grad():
            self.layers[-1].weight.uniform_(-1e-4, 1e-4)
            if self.layers[-1].bias is not None:
                self.layers[-1].bias.uniform_(-1e-4, 1e-4)
        
        print(f"  {channel_name} Head: {input_dim} → {hidden_dim}" + 
              (f" → {hidden_dim}" if num_layers == 2 else "") + " → 1")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through channel head.
        
        Args:
            x: Input features [Batch, N, input_dim]
        
        Returns:
            Channel predictions [Batch, N, 1]
        """
        for layer in self.layers:
            x = layer(x)
        
        return x


class SIRENWithChannelHeads(nn.Module):
    """
    Complete SIREN network with channel-specific prediction heads.
    Implements the full architecture for underwater image restoration.
    
    Architecture:
        Fourier Features [B, N, 256] 
        → SIREN Trunk [B, N, 256]
        → Channel-specific heads:
            - Red Head: [B, N, 256] → [B, N, 1]
            - Green Head: [B, N, 256] → [B, N, 1]
            - Blue Head: [B, N, 256] → [B, N, 1]
        → Concatenate → [B, N, 3]
    """
    
    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 256,
        num_hidden_layers: int = 4,
        omega: float = 30.0,
        omega_first: Optional[float] = None,
        head_architecture: str = 'dense',  # 'dense' or 'adaptive'
        head_config: Optional[dict] = None
    ):
        """
        Args:
            input_dim: Input feature dimension (from Fourier encoding)
            hidden_dim: Hidden layer dimension for trunk
            num_hidden_layers: Number of hidden layers in trunk
            omega: Frequency parameter for sine activation
            omega_first: Frequency parameter for first layer
            head_architecture: 'dense' (equal capacity) or 'adaptive' (varying capacity)
            head_config: Optional dict specifying head configurations
                        Format: {'red': 256, 'green': 192, 'blue': 128}
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.head_architecture = head_architecture
        
        # SIREN trunk for shared feature extraction
        self.trunk = SIRENTrunk(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
            omega=omega,
            omega_first=omega_first
        )
        
        # Channel-specific prediction heads
        print(f"\nChannel-Specific Prediction Heads ({head_architecture}):")
        
        if head_architecture == 'dense':
            # Dense architecture: All heads have equal capacity (256 hidden units)
            self.red_head = ChannelHead(
                input_dim=hidden_dim,
                hidden_dim=256,
                num_layers=1,
                omega=omega,
                channel_name='Red'
            )
            self.green_head = ChannelHead(
                input_dim=hidden_dim,
                hidden_dim=256,
                num_layers=1,
                omega=omega,
                channel_name='Green'
            )
            self.blue_head = ChannelHead(
                input_dim=hidden_dim,
                hidden_dim=256,
                num_layers=1,
                omega=omega,
                channel_name='Blue'
            )
            
        elif head_architecture == 'adaptive':
            # Adaptive architecture: Varying capacity based on degradation severity
            # Red (most degraded) > Green > Blue (least degraded)
            if head_config is None:
                head_config = {'red': 256, 'green': 192, 'blue': 128}
            
            self.red_head = ChannelHead(
                input_dim=hidden_dim,
                hidden_dim=head_config['red'],
                num_layers=1,
                omega=omega,
                channel_name=f"Red (adaptive-{head_config['red']})"
            )
            self.green_head = ChannelHead(
                input_dim=hidden_dim,
                hidden_dim=head_config['green'],
                num_layers=1,
                omega=omega,
                channel_name=f"Green (adaptive-{head_config['green']})"
            )
            self.blue_head = ChannelHead(
                input_dim=hidden_dim,
                hidden_dim=head_config['blue'],
                num_layers=1,
                omega=omega,
                channel_name=f"Blue (adaptive-{head_config['blue']})"
            )
        else:
            raise ValueError(f"Unknown head_architecture: {head_architecture}")
        
        print(f"\nSIRENWithChannelHeads initialized:")
        print(f"  Total parameters: {self.count_parameters():,}")
        print(f"  Trunk parameters: {sum(p.numel() for p in self.trunk.parameters()):,}")
        print(f"  Head parameters: {sum(p.numel() for p in self.red_head.parameters()) + sum(p.numel() for p in self.green_head.parameters()) + sum(p.numel() for p in self.blue_head.parameters()):,}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through complete network.
        
        Args:
            x: Input Fourier-encoded features [Batch, N, input_dim]
        
        Returns:
            RGB predictions [Batch, N, 3]
        """
        # Shared feature extraction through trunk
        shared_features = self.trunk(x)  # [B, N, 256]
        
        # Channel-specific predictions
        red_pred = self.red_head(shared_features)    # [B, N, 1]
        green_pred = self.green_head(shared_features)  # [B, N, 1]
        blue_pred = self.blue_head(shared_features)   # [B, N, 1]
        
        # Concatenate channel predictions
        rgb_pred = torch.cat([red_pred, green_pred, blue_pred], dim=-1)  # [B, N, 3]
        
        # Apply sigmoid to ensure output is in [0, 1]
        rgb_pred = torch.sigmoid(rgb_pred)
        
        return rgb_pred
    
    def forward_with_intermediate(self, x: torch.Tensor) -> dict:
        """
        Forward pass that returns intermediate features for analysis.
        
        Returns:
            Dictionary with:
                - 'shared_features': Features from trunk [B, N, 256]
                - 'red': Red channel prediction [B, N, 1]
                - 'green': Green channel prediction [B, N, 1]
                - 'blue': Blue channel prediction [B, N, 1]
                - 'rgb': Final RGB prediction [B, N, 3]
        """
        # Shared features
        shared_features = self.trunk(x)
        
        # Individual channel predictions (before sigmoid)
        red_raw = self.red_head(shared_features)
        green_raw = self.green_head(shared_features)
        blue_raw = self.blue_head(shared_features)
        
        # Apply sigmoid
        red_pred = torch.sigmoid(red_raw)
        green_pred = torch.sigmoid(green_raw)
        blue_pred = torch.sigmoid(blue_raw)
        
        # Concatenate
        rgb_pred = torch.cat([red_pred, green_pred, blue_pred], dim=-1)
        
        return {
            'shared_features': shared_features,
            'red_raw': red_raw,
            'green_raw': green_raw,
            'blue_raw': blue_raw,
            'red': red_pred,
            'green': green_pred,
            'blue': blue_pred,
            'rgb': rgb_pred
        }
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_parameter_groups(self, trunk_lr: float = 1e-4, head_lr: float = 1e-4) -> List[dict]:
        """
        Get parameter groups for optimizer with different learning rates.
        
        Args:
            trunk_lr: Learning rate for trunk network
            head_lr: Learning rate for channel heads
        
        Returns:
            List of parameter group dictionaries
        """
        return [
            {
                'params': self.trunk.parameters(),
                'lr': trunk_lr,
                'name': 'trunk'
            },
            {
                'params': self.red_head.parameters(),
                'lr': head_lr,
                'name': 'red_head'
            },
            {
                'params': self.green_head.parameters(),
                'lr': head_lr,
                'name': 'green_head'
            },
            {
                'params': self.blue_head.parameters(),
                'lr': head_lr,
                'name': 'blue_head'
            }
        ]


def create_siren_model(
    input_dim: int = 256,
    hidden_dim: int = 256,
    num_hidden_layers: int = 4,
    omega: float = 30.0,
    head_architecture: str = 'dense',
    head_config: Optional[dict] = None,
    device: torch.device = torch.device('cpu')
) -> SIRENWithChannelHeads:
    """
    Factory function to create SIREN model with channel heads.
    
    Args:
        input_dim: Input dimension from Fourier encoding
        hidden_dim: Hidden dimension for trunk
        num_hidden_layers: Number of hidden layers in trunk
        omega: Frequency parameter
        head_architecture: 'dense' or 'adaptive'
        head_config: Optional head configuration for adaptive architecture
        device: Device to place model on
    
    Returns:
        SIRENWithChannelHeads model
    """
    model = SIRENWithChannelHeads(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_hidden_layers=num_hidden_layers,
        omega=omega,
        head_architecture=head_architecture,
        head_config=head_config
    ).to(device)
    
    return model

def test_channel_head():
    """Test individual channel head."""
    print("\n" + "="*70)
    print("TEST 1: CHANNEL HEAD")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test dense head
    print("\nTesting Dense Head (256 hidden units):")
    head = ChannelHead(
        input_dim=256,
        hidden_dim=256,
        num_layers=1,
        channel_name='Red'
    ).to(device)
    
    x = torch.randn(4, 2048, 256, device=device)
    out = head(x)
    
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {out.shape}")
    print(f"  Output range: [{out.min().item():.3f}, {out.max().item():.3f}]")
    print(f"  Parameters: {sum(p.numel() for p in head.parameters()):,}")
    
    # Test adaptive head
    print("\nTesting Adaptive Head (128 hidden units):")
    head_adaptive = ChannelHead(
        input_dim=256,
        hidden_dim=128,
        num_layers=1,
        channel_name='Blue'
    ).to(device)
    
    out_adaptive = head_adaptive(x)
    
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {out_adaptive.shape}")
    print(f"  Output range: [{out_adaptive.min().item():.3f}, {out_adaptive.max().item():.3f}]")
    print(f"  Parameters: {sum(p.numel() for p in head_adaptive.parameters()):,}")


def test_siren_with_dense_heads():
    """Test SIREN with dense (equal capacity) channel heads."""
    print("\n" + "="*70)
    print("TEST 2: SIREN WITH DENSE CHANNEL HEADS")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model with dense heads
    model = SIRENWithChannelHeads(
        input_dim=256,
        hidden_dim=256,
        num_hidden_layers=4,
        omega=30.0,
        head_architecture='dense'
    ).to(device)
    
    # Test forward pass
    print("\nForward pass test:")
    x = torch.randn(4, 2048, 256, device=device)
    
    print(f"  Input shape: {x.shape}")
    
    rgb_pred = model(x)
    
    print(f"  RGB prediction shape: {rgb_pred.shape}")
    print(f"  RGB range: [{rgb_pred.min().item():.3f}, {rgb_pred.max().item():.3f}]")
    print(f"  Red channel mean: {rgb_pred[..., 0].mean().item():.3f}")
    print(f"  Green channel mean: {rgb_pred[..., 1].mean().item():.3f}")
    print(f"  Blue channel mean: {rgb_pred[..., 2].mean().item():.3f}")
    
    # Test intermediate outputs
    print("\nIntermediate outputs test:")
    outputs = model.forward_with_intermediate(x)
    
    print(f"  Shared features shape: {outputs['shared_features'].shape}")
    print(f"  Red prediction shape: {outputs['red'].shape}")
    print(f"  Green prediction shape: {outputs['green'].shape}")
    print(f"  Blue prediction shape: {outputs['blue'].shape}")
    print(f"  Final RGB shape: {outputs['rgb'].shape}")


def test_siren_with_adaptive_heads():
    """Test SIREN with adaptive (varying capacity) channel heads."""
    print("\n" + "="*70)
    print("TEST 3: SIREN WITH ADAPTIVE CHANNEL HEADS")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model with adaptive heads
    head_config = {'red': 256, 'green': 192, 'blue': 128}
    
    model = SIRENWithChannelHeads(
        input_dim=256,
        hidden_dim=256,
        num_hidden_layers=4,
        omega=30.0,
        head_architecture='adaptive',
        head_config=head_config
    ).to(device)
    
    # Test forward pass
    print("\nForward pass test:")
    x = torch.randn(4, 2048, 256, device=device)
    
    rgb_pred = model(x)
    
    print(f"  Input shape: {x.shape}")
    print(f"  RGB prediction shape: {rgb_pred.shape}")
    print(f"  RGB range: [{rgb_pred.min().item():.3f}, {rgb_pred.max().item():.3f}]")
    
    # Compare head parameters
    print("\nHead parameter counts:")
    print(f"  Red head (256): {sum(p.numel() for p in model.red_head.parameters()):,}")
    print(f"  Green head (192): {sum(p.numel() for p in model.green_head.parameters()):,}")
    print(f"  Blue head (128): {sum(p.numel() for p in model.blue_head.parameters()):,}")


def test_parameter_groups():
    """Test parameter grouping for optimizer."""
    print("\n" + "="*70)
    print("TEST 4: PARAMETER GROUPS")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = SIRENWithChannelHeads(
        input_dim=256,
        hidden_dim=256,
        num_hidden_layers=4,
        head_architecture='dense'
    ).to(device)
    
    # Get parameter groups
    param_groups = model.get_parameter_groups(trunk_lr=1e-4, head_lr=5e-4)
    
    print("\nParameter groups:")
    for group in param_groups:
        num_params = sum(p.numel() for p in group['params'])
        print(f"  {group['name']}: {num_params:,} parameters, lr={group['lr']}")
    
    # Create optimizer with different learning rates
    optimizer = torch.optim.Adam(param_groups)
    
    print("\n✓ Optimizer created with parameter groups")


def test_gradient_flow():
    """Test gradient flow through complete network."""
    print("\n" + "="*70)
    print("TEST 5: GRADIENT FLOW")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = SIRENWithChannelHeads(
        input_dim=256,
        hidden_dim=256,
        num_hidden_layers=4,
        head_architecture='dense'
    ).to(device)
    
    # Forward pass
    x = torch.randn(4, 2048, 256, device=device)
    target = torch.rand(4, 2048, 3, device=device)
    
    prediction = model(x)
    loss = torch.nn.functional.mse_loss(prediction, target)
    
    # Backward pass
    loss.backward()
    
    # Check gradients in each component
    print("\nGradient statistics:")
    
    trunk_grads = [p.grad.norm().item() for p in model.trunk.parameters() if p.grad is not None]
    red_grads = [p.grad.norm().item() for p in model.red_head.parameters() if p.grad is not None]
    green_grads = [p.grad.norm().item() for p in model.green_head.parameters() if p.grad is not None]
    blue_grads = [p.grad.norm().item() for p in model.blue_head.parameters() if p.grad is not None]
    
    print(f"  Trunk: {len(trunk_grads)} params, grad norm range: [{min(trunk_grads):.6f}, {max(trunk_grads):.6f}]")
    print(f"  Red head: {len(red_grads)} params, grad norm range: [{min(red_grads):.6f}, {max(red_grads):.6f}]")
    print(f"  Green head: {len(green_grads)} params, grad norm range: [{min(green_grads):.6f}, {max(green_grads):.6f}]")
    print(f"  Blue head: {len(blue_grads)} params, grad norm range: [{min(blue_grads):.6f}, {max(blue_grads):.6f}]")
    
    print("\n✓ Gradients flow to all components")


def main():
    """Run all SIREN tests."""
    print("\n" + "="*70)
    print("SIREN NETWORK WITH CHANNEL HEADS - COMPREHENSIVE TEST SUITE")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    try:
        test_channel_head()
        test_siren_with_dense_heads()
        test_siren_with_adaptive_heads()
        test_parameter_groups()
        test_gradient_flow()
        
        print("\n" + "="*70)
        print("✓ ALL SIREN CHANNEL HEAD TESTS PASSED SUCCESSFULLY!")
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