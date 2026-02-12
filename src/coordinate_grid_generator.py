import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import cv2


class CoordinateGridGenerator:
    """
    GPU-accelerated coordinate grid generation for implicit neural representations.
    Generates normalized coordinates and implements various sampling strategies.
    """
    
    def __init__(
        self,
        image_size: int = 256,
        device: Optional[torch.device] = None,
        samples_per_image: int = 2048
    ):
        """
        Initialize the coordinate grid generator.
        
        Args:
            image_size: Size of the image (assumes square images)
            device: GPU device to use
            samples_per_image: Number of coordinate samples per image for training
        """
        self.image_size = image_size
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.samples_per_image = samples_per_image
        
        # Pre-generate full normalized coordinate grid
        self.full_grid = self._generate_normalized_grid()
        
        print(f"CoordinateGridGenerator initialized on device: {self.device}")
        print(f"Image size: {image_size}x{image_size}")
        print(f"Samples per image: {samples_per_image}")
    
    def _generate_normalized_grid(self) -> torch.Tensor:
        """
        Generate normalized coordinate grid for the entire image.
        Coordinates are normalized to range [-1, 1].
        
        Returns:
            Coordinate grid tensor of shape [H, W, 2] where last dim is (x, y)
        """
        # Create linearly spaced coordinates
        y_coords = torch.linspace(-1, 1, self.image_size, device=self.device)
        x_coords = torch.linspace(-1, 1, self.image_size, device=self.device)
        
        # Create meshgrid
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Stack to create [H, W, 2] grid
        grid = torch.stack([x_grid, y_grid], dim=-1)
        
        return grid
    
    def get_full_grid(self) -> torch.Tensor:
        """
        Get the full normalized coordinate grid.
        
        Returns:
            Grid tensor [H, W, 2]
        """
        return self.full_grid
    
    def sample_random(
        self,
        num_samples: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Random uniform sampling of coordinates from the full grid.
        
        Args:
            num_samples: Number of samples (default: self.samples_per_image)
            
        Returns:
            Tuple of (coordinates [N, 2], indices [N, 2])
            - coordinates: Normalized (x, y) coordinates in [-1, 1]
            - indices: (row, col) pixel indices in [0, image_size-1]
        """
        if num_samples is None:
            num_samples = self.samples_per_image
        
        # Random sampling of pixel indices
        row_indices = torch.randint(0, self.image_size, (num_samples,), device=self.device)
        col_indices = torch.randint(0, self.image_size, (num_samples,), device=self.device)
        
        # Get coordinates from full grid
        coordinates = self.full_grid[row_indices, col_indices]
        
        # Stack indices
        indices = torch.stack([row_indices, col_indices], dim=-1)
        
        return coordinates, indices
    
    def sample_edge_weighted(
        self,
        edge_map: torch.Tensor,
        num_samples: Optional[int] = None,
        edge_bias: float = 0.7
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Edge-weighted sampling - higher probability near edges.
        
        Args:
            edge_map: Edge detection map [H, W] with values in [0, 1]
            num_samples: Number of samples (default: self.samples_per_image)
            edge_bias: Probability weight for edge regions (0.7 = 70% from edges)
            
        Returns:
            Tuple of (coordinates [N, 2], indices [N, 2])
        """
        if num_samples is None:
            num_samples = self.samples_per_image
        
        edge_map = edge_map.to(self.device)
        
        # Flatten edge map to create probability distribution
        edge_flat = edge_map.flatten()
        
        # Add small epsilon to avoid zero probabilities
        edge_flat = edge_flat + 1e-6
        
        # Normalize to create probability distribution
        edge_probs = edge_flat / edge_flat.sum()
        
        # Number of samples from edge regions vs uniform
        num_edge_samples = int(num_samples * edge_bias)
        num_uniform_samples = num_samples - num_edge_samples
        
        # Sample from edge distribution
        edge_flat_indices = torch.multinomial(
            edge_probs,
            num_edge_samples,
            replacement=True
        )
        
        # Convert flat indices to 2D indices
        edge_row_indices = edge_flat_indices // self.image_size
        edge_col_indices = edge_flat_indices % self.image_size
        
        # Sample remaining uniformly
        uniform_row_indices = torch.randint(0, self.image_size, (num_uniform_samples,), device=self.device)
        uniform_col_indices = torch.randint(0, self.image_size, (num_uniform_samples,), device=self.device)
        
        # Combine edge and uniform samples
        row_indices = torch.cat([edge_row_indices, uniform_row_indices])
        col_indices = torch.cat([edge_col_indices, uniform_col_indices])
        
        # Shuffle to mix edge and uniform samples
        shuffle_idx = torch.randperm(num_samples, device=self.device)
        row_indices = row_indices[shuffle_idx]
        col_indices = col_indices[shuffle_idx]
        
        # Get coordinates from full grid
        coordinates = self.full_grid[row_indices, col_indices]
        
        # Stack indices
        indices = torch.stack([row_indices, col_indices], dim=-1)
        
        return coordinates, indices
    
    def sample_uniform_grid(
        self,
        stride: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Uniform/dense sampling in a regular grid pattern.
        Used for validation and testing.
        
        Args:
            stride: Stride for grid sampling (1 = every pixel, 2 = every other pixel, etc.)
            
        Returns:
            Tuple of (coordinates [N, 2], indices [N, 2])
        """
        # Create strided indices
        row_indices = torch.arange(0, self.image_size, stride, device=self.device)
        col_indices = torch.arange(0, self.image_size, stride, device=self.device)
        
        # Create meshgrid of indices
        row_grid, col_grid = torch.meshgrid(row_indices, col_indices, indexing='ij')
        
        # Flatten grids
        row_flat = row_grid.flatten()
        col_flat = col_grid.flatten()
        
        # Get coordinates from full grid
        coordinates = self.full_grid[row_flat, col_flat]
        
        # Stack indices
        indices = torch.stack([row_flat, col_flat], dim=-1)
        
        return coordinates, indices
    
    def compute_edge_map(
        self,
        image: torch.Tensor,
        method: str = 'sobel',
        threshold: float = 0.1
    ) -> torch.Tensor:
        """
        Compute edge map for edge-weighted sampling.
        
        Args:
            image: Input image tensor [C, H, W] or [H, W, C]
            method: Edge detection method ('sobel', 'canny', 'gradient')
            threshold: Threshold for edge detection
            
        Returns:
            Edge map [H, W] with values in [0, 1], guaranteed to match image size
        """
        # Ensure image is [C, H, W]
        if image.shape[-1] == 3:
            image = image.permute(2, 0, 1)
        
        image = image.to(self.device)
        
        # Store original dimensions
        _, H, W = image.shape
        
        # Convert to grayscale
        if image.shape[0] == 3:
            grayscale = 0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]
        else:
            grayscale = image[0]
        
        if method == 'sobel':
            edge_map = self._sobel_edge_detection(grayscale)
        elif method == 'canny':
            edge_map = self._canny_edge_detection(grayscale, threshold)
        elif method == 'gradient':
            edge_map = self._gradient_edge_detection(grayscale)
        else:
            raise ValueError(f"Unknown edge detection method: {method}")
        
        # Ensure edge map matches original image size
        if edge_map.shape != (H, W):
            edge_map = F.interpolate(
                edge_map.unsqueeze(0).unsqueeze(0),
                size=(H, W),
                mode='nearest'
            ).squeeze()
        
        # Normalize to [0, 1]
        edge_map = (edge_map - edge_map.min()) / (edge_map.max() - edge_map.min() + 1e-8)
        
        return edge_map
    
    def _sobel_edge_detection(self, grayscale: torch.Tensor) -> torch.Tensor:
        """
        Sobel edge detection using GPU convolution.
        
        Args:
            grayscale: Grayscale image [H, W]
            
        Returns:
            Edge magnitude [H, W]
        """
        # Sobel kernels
        sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32, device=self.device).view(1, 1, 3, 3)
        
        sobel_y = torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32, device=self.device).view(1, 1, 3, 3)
        
        # Add batch and channel dimensions
        img = grayscale.unsqueeze(0).unsqueeze(0)
        
        # Apply Sobel filters
        edges_x = F.conv2d(img, sobel_x, padding=1)
        edges_y = F.conv2d(img, sobel_y, padding=1)
        
        # Compute magnitude
        edge_magnitude = torch.sqrt(edges_x**2 + edges_y**2).squeeze()
        
        return edge_magnitude
    
    def _gradient_edge_detection(self, grayscale: torch.Tensor) -> torch.Tensor:
        """
        Simple gradient-based edge detection.
        
        Args:
            grayscale: Grayscale image [H, W]
            
        Returns:
            Edge magnitude [H, W]
        """
        # Add batch and channel dimensions
        img = grayscale.unsqueeze(0).unsqueeze(0)
        
        # Compute gradients using finite differences
        grad_x = img[:, :, :, 1:] - img[:, :, :, :-1]
        grad_y = img[:, :, 1:, :] - img[:, :, :-1, :]
        
        # Pad to match original size
        grad_x = F.pad(grad_x, (0, 1, 0, 0))
        grad_y = F.pad(grad_y, (0, 0, 0, 1))
        
        # Compute magnitude
        edge_magnitude = torch.sqrt(grad_x**2 + grad_y**2).squeeze()
        
        return edge_magnitude
    
    def _canny_edge_detection(
        self,
        grayscale: torch.Tensor,
        threshold: float = 0.1
    ) -> torch.Tensor:
        """
        Canny edge detection using OpenCV (on CPU).
        
        Args:
            grayscale: Grayscale image [H, W]
            threshold: Edge threshold
            
        Returns:
            Edge map [H, W]
        """
        # Convert to numpy and move to CPU
        img_np = grayscale.cpu().numpy()
        img_np = (img_np * 255).astype(np.uint8)
        
        # Apply Canny edge detection
        low_threshold = int(threshold * 255)
        high_threshold = int(threshold * 255 * 2)
        edges = cv2.Canny(img_np, low_threshold, high_threshold)
        
        # Convert back to tensor
        edge_map = torch.from_numpy(edges.astype(np.float32) / 255.0).to(self.device)
        
        return edge_map
    
    def create_data_table(
        self,
        coordinates: torch.Tensor,
        indices: torch.Tensor,
        degraded_image: torch.Tensor,
        reference_image: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Create a data table containing coordinates and corresponding RGB values.
        
        Args:
            coordinates: Normalized coordinates [N, 2]
            indices: Pixel indices [N, 2] (row, col)
            degraded_image: Degraded image [C, H, W]
            reference_image: Reference image [C, H, W]
            
        Returns:
            Dictionary containing:
                - 'coordinates': [N, 2] normalized (x, y) positions
                - 'degraded_rgb': [N, 3] RGB values from degraded image
                - 'reference_rgb': [N, 3] RGB values from reference image
        """
        # Ensure images are on device and [C, H, W]
        if degraded_image.shape[-1] == 3:
            degraded_image = degraded_image.permute(2, 0, 1)
        if reference_image.shape[-1] == 3:
            reference_image = reference_image.permute(2, 0, 1)
        
        degraded_image = degraded_image.to(self.device)
        reference_image = reference_image.to(self.device)
        
        # Get actual image dimensions
        _, H, W = degraded_image.shape
        
        # Extract RGB values at sampled indices
        row_idx = indices[:, 0]
        col_idx = indices[:, 1]
        
        # Clamp indices to valid range (safety check)
        row_idx = torch.clamp(row_idx, 0, H - 1)
        col_idx = torch.clamp(col_idx, 0, W - 1)
        
        # Get RGB values [N, 3]
        degraded_rgb = degraded_image[:, row_idx, col_idx].T
        reference_rgb = reference_image[:, row_idx, col_idx].T
        
        data_table = {
            'coordinates': coordinates,
            'degraded_rgb': degraded_rgb,
            'reference_rgb': reference_rgb
        }
        
        return data_table
    
    def create_batch_data_table(
        self,
        degraded_images: torch.Tensor,
        reference_images: torch.Tensor,
        sampling_strategy: str = 'random',
        edge_maps: Optional[torch.Tensor] = None,
        num_samples: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Create batched data table for multiple images.
        
        Args:
            degraded_images: Batch of degraded images [B, C, H, W]
            reference_images: Batch of reference images [B, C, H, W]
            sampling_strategy: 'random', 'edge_weighted', or 'uniform'
            edge_maps: Edge maps for edge-weighted sampling [B, H, W]
            num_samples: Number of samples per image
            
        Returns:
            Dictionary containing batched data:
                - 'coordinates': [B, N, 2]
                - 'degraded_rgb': [B, N, 3]
                - 'reference_rgb': [B, N, 3]
        """
        batch_size = degraded_images.shape[0]
        if num_samples is None:
            num_samples = self.samples_per_image
        
        # Initialize batch tensors
        batch_coordinates = []
        batch_degraded_rgb = []
        batch_reference_rgb = []
        
        for i in range(batch_size):
            # Sample coordinates based on strategy
            if sampling_strategy == 'random':
                coordinates, indices = self.sample_random(num_samples)
            elif sampling_strategy == 'edge_weighted':
                if edge_maps is None:
                    # Compute edge map on the fly
                    edge_map = self.compute_edge_map(degraded_images[i])
                else:
                    edge_map = edge_maps[i]
                coordinates, indices = self.sample_edge_weighted(edge_map, num_samples)
            elif sampling_strategy == 'uniform':
                # Calculate stride to get approximately num_samples points
                total_pixels = self.image_size ** 2
                stride = max(1, int(np.sqrt(total_pixels / num_samples)))
                coordinates, indices = self.sample_uniform_grid(stride)
            else:
                raise ValueError(f"Unknown sampling strategy: {sampling_strategy}")
            
            # Create data table for this image
            data_table = self.create_data_table(
                coordinates,
                indices,
                degraded_images[i],
                reference_images[i]
            )
            
            batch_coordinates.append(data_table['coordinates'])
            batch_degraded_rgb.append(data_table['degraded_rgb'])
            batch_reference_rgb.append(data_table['reference_rgb'])
        
        # Stack into batch tensors
        batched_data = {
            'coordinates': torch.stack(batch_coordinates),
            'degraded_rgb': torch.stack(batch_degraded_rgb),
            'reference_rgb': torch.stack(batch_reference_rgb)
        }
        
        return batched_data