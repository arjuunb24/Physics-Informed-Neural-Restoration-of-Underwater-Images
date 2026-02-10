"""
Main script to organize underwater image datasets
Run this script to organize UIEB, SUIM-E, and EUVP datasets
"""
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from data_organization import DataOrganizer
from utils import setup_gpu


def main():
    """Main execution function"""
    print("\n" + "=" * 60)
    print("UNDERWATER IMAGE DATASET ORGANIZATION")
    print("=" * 60)
    
    # Setup GPU
    device = setup_gpu()
    
    # Define paths - MODIFY THESE TO MATCH YOUR SETUP
    # Assuming your project structure is:
    # your_project/
    #   ├── data/
    #   │   └── raw/  (your existing Underwater Image Restoration dataset folder)
    #   ├── src/
    #   └── organize_data.py (this file)
    
    raw_data_dir = Path("data/raw")  # Your existing dataset folder with UIEB, SUIM-E, EUVP
    output_dir = Path("data")        # Will create processed/ and registry/ subdirectories here
    
    print(f"\nConfiguration:")
    print(f"  Raw data directory: {raw_data_dir.absolute()}")
    print(f"  Output directory: {output_dir.absolute()}")
    print(f"  GPU device: {device}")
    
    # Verify raw data directory exists
    if not raw_data_dir.exists():
        print(f"\n❌ Error: Raw data directory not found at {raw_data_dir.absolute()}")
        print("\nPlease ensure your folder structure is:")
        print("  your_project/")
        print("    ├── data/")
        print("    │   └── raw/")
        print("    │       ├── UIEB/")
        print("    │       │   ├── reference-890/")
        print("    │       │   └── raw-890/")
        print("    │       ├── SUIM-E/")
        print("    │       │   ├── reference (B)/")
        print("    │       │   └── raw (A)/")
        print("    │       └── EUVP/")
        print("    │           ├── reference (B)/")
        print("    │           └── raw (A)/")
        print("    ├── src/")
        print("    └── organize_data.py")
        print("\nIf your structure is different, update the paths in this script.")
        return
    
    # Check if datasets exist
    print(f"\nChecking for datasets in {raw_data_dir}...")
    datasets_found = []
    datasets_expected = ['UIEB', 'SUIM-E', 'EUVP']
    
    for dataset in datasets_expected:
        dataset_path = raw_data_dir / dataset
        if dataset_path.exists():
            print(f"  ✓ Found {dataset}")
            datasets_found.append(dataset)
        else:
            print(f"  ⚠ {dataset} not found at {dataset_path}")
    
    if len(datasets_found) == 0:
        print("\n❌ Error: No datasets found!")
        return
    
    print(f"\nProceeding with {len(datasets_found)} dataset(s): {', '.join(datasets_found)}")
    
    # Ask for confirmation
    print("\n" + "─" * 60)
    response = input("Continue with data organization? (yes/no): ").strip().lower()
    
    if response not in ['yes', 'y']:
        print("Organization cancelled.")
        return
    
    # Create organizer and run
    print("\n" + "─" * 60)
    organizer = DataOrganizer(raw_data_dir, output_dir, device)
    organizer.organize_all_datasets()
    
    print("\n" + "=" * 60)
    print("✓ ORGANIZATION COMPLETE!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Check the dataset registry:")
    print(f"     {output_dir / 'registry' / 'dataset_registry.json'}")
    print("  2. Verify organized data:")
    print(f"     {output_dir / 'processed'}")
    print("  3. Test the dataloader:")
    print("     python src/dataset_loader.py")


if __name__ == "__main__":
    main()