import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import json
from typing import Dict, Tuple, Optional, List
from torchvision import transforms
from skimage.metrics import structural_similarity as ssim


class ImagePreprocessor:
    """
    GPU-accelerated image preprocessing with resolution standardization and degradation analysis.
    """
    
    def __init__(
        self,
        target_size: int = 256,
        device: Optional[torch.device] = None,
        maintain_aspect_ratio: bool = True
    ):
        """
        Initialize the preprocessor.
        
        Args:
            target_size: Target resolution (default 256x256)
            device: GPU device to use
            maintain_aspect_ratio: Whether to maintain aspect ratio during resize
        """
        self.target_size = target_size
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.maintain_aspect_ratio = maintain_aspect_ratio
        
        print(f"ImagePreprocessor initialized on device: {self.device}")
    
    def resize_image(
        self,
        image: torch.Tensor,
        method: str = 'smart'
    ) -> torch.Tensor:
        """
        Resize image to target size using specified method.
        
        Args:
            image: Input image tensor [C, H, W] or [H, W, C]
            method: 'smart' (crop/pad based on size), 'center_crop', 'pad', 'stretch'
            
        Returns:
            Resized image tensor [C, target_size, target_size]
        """
        # Ensure image is [C, H, W]
        if image.ndim == 3 and image.shape[2] == 3:
            image = image.permute(2, 0, 1)
        
        # Move to GPU
        image = image.to(self.device)
        
        C, H, W = image.shape
        
        if method == 'smart':
            # If larger than target, crop. If smaller, pad
            if H > self.target_size or W > self.target_size:
                image = self._center_crop(image)
            elif H < self.target_size or W < self.target_size:
                image = self._pad_image(image)
            else:
                return image
                
        elif method == 'center_crop':
            image = self._center_crop(image)
            
        elif method == 'pad':
            image = self._pad_image(image)
            
        elif method == 'stretch':
            # Resize without maintaining aspect ratio
            image = F.interpolate(
                image.unsqueeze(0),
                size=(self.target_size, self.target_size),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
        
        return image
    
    def _center_crop(self, image: torch.Tensor) -> torch.Tensor:
        """
        Center crop image to target size.
        
        Args:
            image: Input tensor [C, H, W]
            
        Returns:
            Cropped tensor [C, target_size, target_size]
        """
        C, H, W = image.shape
        
        if self.maintain_aspect_ratio:
            # Resize so smallest dimension matches target_size
            scale = self.target_size / min(H, W)
            new_H = int(H * scale)
            new_W = int(W * scale)
            
            image = F.interpolate(
                image.unsqueeze(0),
                size=(new_H, new_W),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
            
            H, W = new_H, new_W
        
        # Calculate crop coordinates
        top = (H - self.target_size) // 2
        left = (W - self.target_size) // 2
        
        # Crop
        cropped = image[:, top:top + self.target_size, left:left + self.target_size]
        
        return cropped
    
    def _pad_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Pad image to target size.
        
        Args:
            image: Input tensor [C, H, W]
            
        Returns:
            Padded tensor [C, target_size, target_size]
        """
        C, H, W = image.shape
        
        if self.maintain_aspect_ratio:
            # Resize so largest dimension matches target_size
            scale = self.target_size / max(H, W)
            new_H = int(H * scale)
            new_W = int(W * scale)
            
            image = F.interpolate(
                image.unsqueeze(0),
                size=(new_H, new_W),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
            
            H, W = new_H, new_W
        
        # Calculate padding
        pad_top = (self.target_size - H) // 2
        pad_bottom = self.target_size - H - pad_top
        pad_left = (self.target_size - W) // 2
        pad_right = self.target_size - W - pad_left
        
        # Pad with zeros (black)
        padded = F.pad(
            image,
            (pad_left, pad_right, pad_top, pad_bottom),
            mode='constant',
            value=0
        )
        
        return padded
    
    def analyze_degradation(
        self,
        degraded_image: torch.Tensor,
        reference_image: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Analyze degradation characteristics of an image.
        
        Args:
            degraded_image: Degraded image tensor [C, H, W], values in [0, 1]
            reference_image: Optional reference image for comparison
            
        Returns:
            Dictionary containing degradation metrics
        """
        metrics = {}
        
        # Ensure image is on GPU
        degraded_image = degraded_image.to(self.device)
        
        # 1. Color Cast Severity (RGB channel imbalance)
        metrics['color_cast'] = self._compute_color_cast(degraded_image)
        
        # 2. Contrast Level (histogram analysis)
        metrics['contrast'] = self._compute_contrast(degraded_image)
        
        # 3. Haze Density (dark channel prior)
        metrics['haze_density'] = self._compute_haze_density(degraded_image)
        
        # 4. Low-Frequency Content Dominance
        metrics['low_freq_dominance'] = self._compute_low_freq_dominance(degraded_image)
        
        # 5. Overall brightness
        metrics['brightness'] = degraded_image.mean().item()
        
        # 6. Saturation
        metrics['saturation'] = self._compute_saturation(degraded_image)
        
        # If reference image provided, compute comparative metrics
        if reference_image is not None:
            reference_image = reference_image.to(self.device)
            
            # PSNR
            metrics['psnr'] = self._compute_psnr(degraded_image, reference_image)
            
            # MSE
            metrics['mse'] = F.mse_loss(degraded_image, reference_image).item()
        
        return metrics
    
    def _compute_color_cast(self, image: torch.Tensor) -> float:
        """
        Compute color cast severity based on RGB channel imbalance.
        Higher values indicate stronger color cast.
        
        Args:
            image: Image tensor [C, H, W]
            
        Returns:
            Color cast severity (0 = no cast, higher = stronger cast)
        """
        # Compute mean of each channel
        r_mean = image[0].mean()
        g_mean = image[1].mean()
        b_mean = image[2].mean()
        
        # Compute standard deviation of channel means
        channel_means = torch.tensor([r_mean, g_mean, b_mean], device=self.device)
        color_cast = channel_means.std().item()
        
        return color_cast
    
    def _compute_contrast(self, image: torch.Tensor) -> float:
        """
        Compute contrast level using histogram analysis.
        
        Args:
            image: Image tensor [C, H, W]
            
        Returns:
            Contrast value (higher = more contrast)
        """
        # Convert to grayscale
        grayscale = 0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]
        
        # Compute standard deviation as a measure of contrast
        contrast = grayscale.std().item()
        
        return contrast
    
    def _compute_haze_density(self, image: torch.Tensor) -> float:
        """
        Estimate haze density using dark channel prior.
        Higher values indicate more haze.
        
        Args:
            image: Image tensor [C, H, W]
            
        Returns:
            Haze density estimate (0 = clear, 1 = heavy haze)
        """
        # Dark channel prior: minimum across RGB channels in local patches
        patch_size = 15
        
        # Compute dark channel using min pooling
        dark_channel = image.min(dim=0)[0]  # Min across RGB channels
        
        # Apply min pooling with patch_size
        dark_channel = dark_channel.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        dark_channel = -F.max_pool2d(
            -dark_channel,
            kernel_size=patch_size,
            stride=1,
            padding=patch_size // 2
        ).squeeze()
        
        # Haze density is the mean of dark channel
        haze_density = dark_channel.mean().item()
        
        return haze_density
    
    def _compute_low_freq_dominance(self, image: torch.Tensor) -> float:
        """
        Compute low-frequency content dominance using FFT.
        Higher values indicate more low-frequency content (blurry/hazy).
        
        Args:
            image: Image tensor [C, H, W]
            
        Returns:
            Low-frequency dominance ratio
        """
        # Convert to grayscale
        grayscale = 0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]
        
        # Compute 2D FFT
        fft = torch.fft.fft2(grayscale)
        fft_shift = torch.fft.fftshift(fft)
        magnitude = torch.abs(fft_shift)
        
        H, W = magnitude.shape
        center_h, center_w = H // 2, W // 2
        
        # Define low-frequency region (center 25% of spectrum)
        low_freq_size = min(H, W) // 4
        low_freq_region = magnitude[
            center_h - low_freq_size:center_h + low_freq_size,
            center_w - low_freq_size:center_w + low_freq_size
        ]
        
        # Compute energy ratio
        low_freq_energy = low_freq_region.sum()
        total_energy = magnitude.sum()
        
        low_freq_dominance = (low_freq_energy / total_energy).item()
        
        return low_freq_dominance
    
    def _compute_saturation(self, image: torch.Tensor) -> float:
        """
        Compute average saturation of the image.
        
        Args:
            image: Image tensor [C, H, W]
            
        Returns:
            Average saturation value
        """
        # Convert RGB to HSV (on CPU for cv2 compatibility)
        image_np = image.cpu().permute(1, 2, 0).numpy()
        image_np = (image_np * 255).astype(np.uint8)
        
        hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
        saturation = hsv[:, :, 1].mean() / 255.0
        
        return float(saturation)
    
    def _compute_psnr(
        self,
        degraded: torch.Tensor,
        reference: torch.Tensor
    ) -> float:
        """
        Compute Peak Signal-to-Noise Ratio.
        
        Args:
            degraded: Degraded image tensor
            reference: Reference image tensor
            
        Returns:
            PSNR value in dB
        """
        mse = F.mse_loss(degraded, reference)
        
        if mse == 0:
            return 100.0
        
        psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
        return psnr.item()
    
    def preprocess_image_pair(
        self,
        degraded_path: Path,
        reference_path: Path,
        resize_method: str = 'smart',
        analyze: bool = True
    ) -> Dict:
        """
        Preprocess a single image pair (degraded + reference).
        
        Args:
            degraded_path: Path to degraded image
            reference_path: Path to reference image
            resize_method: Resizing method to use
            analyze: Whether to perform degradation analysis
            
        Returns:
            Dictionary containing processed images and metrics
        """
        # Load images
        degraded_img = self._load_image(degraded_path)
        reference_img = self._load_image(reference_path)
        
        # Resize images
        degraded_resized = self.resize_image(degraded_img, method=resize_method)
        reference_resized = self.resize_image(reference_img, method=resize_method)
        
        result = {
            'degraded': degraded_resized,
            'reference': reference_resized,
            'original_size_degraded': degraded_img.shape[1:],
            'original_size_reference': reference_img.shape[1:],
        }
        
        # Perform degradation analysis if requested
        if analyze:
            metrics = self.analyze_degradation(degraded_resized, reference_resized)
            result['metrics'] = metrics
        
        return result
    
    def _load_image(self, path: Path) -> torch.Tensor:
        """
        Load image from disk and convert to tensor.
        
        Args:
            path: Path to image file
            
        Returns:
            Image tensor [C, H, W] with values in [0, 1]
        """
        img = Image.open(path).convert('RGB')
        img_array = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)  # [H, W, C] -> [C, H, W]
        
        return img_tensor
    
    def save_preprocessed_image(
        self,
        image: torch.Tensor,
        output_path: Path
    ):
        """
        Save preprocessed image to disk.
        
        Args:
            image: Image tensor [C, H, W]
            output_path: Path to save image
        """
        # Move to CPU and convert to numpy
        image_np = image.cpu().permute(1, 2, 0).numpy()
        image_np = (image_np * 255).clip(0, 255).astype(np.uint8)
        
        # Save using PIL
        img = Image.fromarray(image_np)
        img.save(output_path)
    
    def batch_preprocess_dataset(
        self,
        registry_path: Path,
        output_dir: Path,
        split: str = 'train',
        resize_method: str = 'smart',
        save_preprocessed: bool = True,
        save_metrics: bool = True
    ) -> Dict:
        """
        Preprocess entire dataset split in batches using registry.
        
        Args:
            registry_path: Path to dataset_registry.json
            output_dir: Output directory for preprocessed images
            split: Dataset split ('train', 'val', 'test')
            resize_method: Resizing method
            save_preprocessed: Whether to save preprocessed images
            save_metrics: Whether to save degradation metrics
            
        Returns:
            Dictionary containing aggregate statistics
        """
        # Load registry
        with open(registry_path, 'r') as f:
            registry = json.load(f)
        
        # Get pairs for this split
        pairs = registry['splits'][split]
        
        print(f"\nPreprocessing {len(pairs)} image pairs from {split} split...")
        
        # Create output directories
        if save_preprocessed:
            degraded_out = output_dir / split / 'raw'
            reference_out = output_dir / split / 'reference'
            degraded_out.mkdir(parents=True, exist_ok=True)
            reference_out.mkdir(parents=True, exist_ok=True)
        
        metrics_list = []
        processed_count = 0
        
        for idx, pair in enumerate(pairs):
            # Convert Windows paths to Path objects
            degraded_path = Path(pair['raw'])
            reference_path = Path(pair['reference'])
            
            if not degraded_path.exists() or not reference_path.exists():
                print(f"WARNING: Missing files for pair {idx}, skipping...")
                continue
            
            try:
                # Preprocess pair
                result = self.preprocess_image_pair(
                    degraded_path,
                    reference_path,
                    resize_method=resize_method,
                    analyze=True
                )
                
                # Save preprocessed images if requested
                if save_preprocessed:
                    self.save_preprocessed_image(
                        result['degraded'],
                        degraded_out / degraded_path.name
                    )
                    self.save_preprocessed_image(
                        result['reference'],
                        reference_out / reference_path.name
                    )
                
                # Store metrics with full metadata
                metrics_entry = {
                    'pair_id': degraded_path.stem,
                    'dataset': pair['dataset'],
                    'degraded_path': str(degraded_path),
                    'reference_path': str(reference_path),
                    'original_degraded': pair['original_raw'],
                    'original_reference': pair['original_reference'],
                    **result['metrics']
                }
                metrics_list.append(metrics_entry)
                
                processed_count += 1
                
                if processed_count % 100 == 0:
                    print(f"Processed {processed_count}/{len(pairs)} pairs...")
                    
            except Exception as e:
                print(f"ERROR processing pair {idx}: {str(e)}")
                continue
        
        print(f"Successfully processed {processed_count}/{len(pairs)} pairs")
        
        # Save metrics to JSON
        if save_metrics and metrics_list:
            metrics_path = output_dir / f'{split}_degradation_metrics.json'
            with open(metrics_path, 'w') as f:
                json.dump(metrics_list, f, indent=2)
            print(f"Saved degradation metrics to: {metrics_path}")
        
        # Compute aggregate statistics (overall + per-dataset)
        aggregate_stats = self._compute_aggregate_stats(metrics_list) if metrics_list else {}
        
        return {
            'num_pairs': processed_count,
            'metrics': metrics_list,
            'aggregate_stats': aggregate_stats
        }
    
    def _compute_aggregate_stats(self, metrics_list: List[Dict]) -> Dict:
        """
        Compute aggregate statistics across all images and per dataset.
        
        Args:
            metrics_list: List of metric dictionaries
            
        Returns:
            Aggregate statistics
        """
        if not metrics_list:
            return {}
            
        import numpy as np
        
        stats = {}
        
        # Metrics to aggregate
        metric_keys = [
            'color_cast', 'contrast', 'haze_density', 
            'low_freq_dominance', 'brightness', 'saturation'
        ]
        
        if 'psnr' in metrics_list[0]:
            metric_keys.extend(['psnr', 'mse'])
        
        # Overall statistics
        stats['overall'] = {}
        for key in metric_keys:
            values = [m[key] for m in metrics_list]
            stats['overall'][key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'median': float(np.median(values))
            }
        
        # Per-dataset statistics
        datasets = sorted(set(m['dataset'] for m in metrics_list))
        stats['per_dataset'] = {}
        
        for dataset in datasets:
            dataset_metrics = [m for m in metrics_list if m['dataset'] == dataset]
            stats['per_dataset'][dataset] = {
                'count': len(dataset_metrics)
            }
            
            for key in metric_keys:
                values = [m[key] for m in dataset_metrics]
                stats['per_dataset'][dataset][key] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values))
                }
        
        return stats