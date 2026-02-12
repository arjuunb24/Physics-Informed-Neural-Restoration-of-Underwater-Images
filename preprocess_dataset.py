import torch
from pathlib import Path
import json
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from image_preprocessing import ImagePreprocessor


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


def print_dataset_summary(registry_path: Path):
    """Print dataset summary from registry."""
    with open(registry_path, 'r') as f:
        registry = json.load(f)
    
    print("\nDataset Summary:")
    print("=" * 70)
    for dataset, info in registry['datasets'].items():
        print(f"\n{dataset}:")
        print(f"  Total pairs: {info['total_pairs']}")
        print(f"  Splits:")
        for split, count in info['splits'].items():
            print(f"    {split}: {count}")
    print("=" * 70)


def main():
    """Main preprocessing execution."""
    
    # Verify GPU
    device = verify_gpu()
    
    # Define paths
    project_root = Path(__file__).parent
    registry_path = project_root / 'data' / 'registry' / 'dataset_registry.json'
    preprocessed_dir = project_root / 'data' / 'preprocessed'
    
    # Verify registry exists
    if not registry_path.exists():
        print(f"ERROR: Registry file not found at {registry_path}")
        print("Please run organize_data.py first to create the registry.")
        return
    
    # Create preprocessed directory
    preprocessed_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("UNDERWATER IMAGE PREPROCESSING")
    print("="*70)
    
    # Print dataset summary
    print_dataset_summary(registry_path)
    
    # Initialize preprocessor
    preprocessor = ImagePreprocessor(
        target_size=256,
        device=device,
        maintain_aspect_ratio=True
    )
    
    # Preprocessing parameters
    resize_method = 'smart'  # Options: 'smart', 'center_crop', 'pad', 'stretch'
    
    # Process each split
    splits = ['train', 'val', 'test']
    all_stats = {}
    
    for split in splits:
        print(f"\n{'='*70}")
        print(f"Processing {split.upper()} split")
        print(f"{'='*70}")
        
        result = preprocessor.batch_preprocess_dataset(
            registry_path=registry_path,
            output_dir=preprocessed_dir,
            split=split,
            resize_method=resize_method,
            save_preprocessed=True,
            save_metrics=True
        )
        
        all_stats[split] = result['aggregate_stats']
        
        # Print summary statistics
        print(f"\n{split.upper()} Split Summary:")
        print(f"  Total pairs processed: {result['num_pairs']}")
        
        if result['aggregate_stats']:
            stats = result['aggregate_stats']
            
            # Overall statistics
            print(f"\n  Overall Degradation Metrics:")
            for metric in ['color_cast', 'contrast', 'haze_density', 'low_freq_dominance']:
                if metric in stats['overall']:
                    print(f"    {metric}:")
                    print(f"      Mean: {stats['overall'][metric]['mean']:.4f}")
                    print(f"      Std:  {stats['overall'][metric]['std']:.4f}")
                    print(f"      Range: [{stats['overall'][metric]['min']:.4f}, {stats['overall'][metric]['max']:.4f}]")
            
            if 'psnr' in stats['overall']:
                print(f"    PSNR (vs reference):")
                print(f"      Mean: {stats['overall']['psnr']['mean']:.2f} dB")
                print(f"      Range: [{stats['overall']['psnr']['min']:.2f}, {stats['overall']['psnr']['max']:.2f}] dB")
            
            # Per-dataset statistics
            print(f"\n  Per-Dataset Degradation Characteristics:")
            for dataset in sorted(stats['per_dataset'].keys()):
                ds_stats = stats['per_dataset'][dataset]
                print(f"\n    {dataset} ({ds_stats['count']} pairs):")
                print(f"      Color Cast: {ds_stats['color_cast']['mean']:.4f} ± {ds_stats['color_cast']['std']:.4f}")
                print(f"      Haze Density: {ds_stats['haze_density']['mean']:.4f} ± {ds_stats['haze_density']['std']:.4f}")
                print(f"      Contrast: {ds_stats['contrast']['mean']:.4f} ± {ds_stats['contrast']['std']:.4f}")
                if 'psnr' in ds_stats:
                    print(f"      PSNR: {ds_stats['psnr']['mean']:.2f} ± {ds_stats['psnr']['std']:.2f} dB")
    
    # Save combined statistics
    combined_stats_path = preprocessed_dir / 'preprocessing_statistics.json'
    with open(combined_stats_path, 'w') as f:
        json.dump(all_stats, f, indent=2)
    
    print(f"\n{'='*70}")
    print("PREPROCESSING COMPLETE")
    print(f"{'='*70}")
    print(f"\nPreprocessed images saved to: {preprocessed_dir}")
    print(f"Combined statistics saved to: {combined_stats_path}")
    print(f"\nGPU Memory Used: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    print(f"GPU Memory Cached: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")


if __name__ == '__main__':
    main()