import torch
import argparse
import sys
from pathlib import Path

# Add src to path to ensure imports work if run from different directories
sys.path.append(str(Path(__file__).parent))

from src.data_loader.dataset import get_dataloaders
from src.data_loader.transforms import get_transforms
from src.utils.config import load_config

def parse_args():
    parser = argparse.ArgumentParser(
        description="""
        FetoGuard Data Pipeline Verification Script.
        
        This script checks if the data loading and processing pipeline is functioning correctly.
        It loads the dataset using parameters from a config file (default: configs/config.yaml)
        and verifies that:
        1. Only image files are loaded.
        2. Splits are created correctly (Train/Val/Test).
        3. Images and Masks are resized and transformed to the correct shapes.
        
        You can override key configuration parameters directly from the command line.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--config', 
        type=str, 
        default='configs/config.yaml', 
        help='Path to the YAML configuration file. Default: configs/config.yaml'
    )
    
    parser.add_argument(
        '--data-dir', 
        type=str, 
        help='Override the data directory path defined in config.'
    )
    
    parser.add_argument(
        '--batch-size', 
        type=int, 
        help='Override the batch size defined in config.'
    )
    
    parser.add_argument(
        '--num-workers', 
        type=int, 
        help='Override the number of workers for data loading.'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed information about the loaded configuration.'
    )
    
    return parser.parse_args()

def test_pipeline():
    args = parse_args()
    
    # Load Config
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"Error: Config file not found at {args.config}")
        sys.exit(1)

    data_cfg = config['data']
    
    # Override config with args if provided
    if args.data_dir:
        data_cfg['data_dir'] = args.data_dir
    if args.batch_size:
        data_cfg['batch_size'] = args.batch_size
    if args.num_workers is not None:
        data_cfg['num_workers'] = args.num_workers
        
    if args.verbose:
        print("-" * 50)
        print(f"Configuration (Active):")
        print(f"  Config File: {args.config}")
        print(f"  Data Directory: {data_cfg['data_dir']}")
        print(f"  Batch Size: {data_cfg['batch_size']}")
        print(f"  Input Size: {data_cfg['input_size']}")
        print(f"  Num Workers: {data_cfg['num_workers']}")
        print("-" * 50)
    
    # Transforms
    img_size = tuple(data_cfg['input_size'])
    train_transform = get_transforms(mode='train', img_size=img_size)
    val_transform = get_transforms(mode='val', img_size=img_size)
    
    print(f"Initializing DataLoaders from: {data_cfg['data_dir']}...")
    
    # Loaders
    try:
        loaders = get_dataloaders(
            data_dir=data_cfg['data_dir'],
            batch_size=data_cfg['batch_size'],
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=val_transform,
            val_split=data_cfg['val_split'],
            num_workers=data_cfg['num_workers']
        )
    except Exception as e:
        print(f"Error initializing dataloaders: {e}")
        sys.exit(1)
    
    # Check lengths
    print(f"\ndataset Statistics:")
    print(f"  Train batches: {len(loaders['train'])}")
    print(f"  Val batches:   {len(loaders['val'])}")
    print(f"  Test batches:  {len(loaders['test'])}")
    
    # Get a batch
    print("\nfetching a sample batch...")
    try:
        images, masks = next(iter(loaders['train']))
    except Exception as e:
        print(f"Error fetching batch: {e}")
        sys.exit(1)
    
    # Verify shapes
    print(f"  Images shape: {images.shape} (Expected: [{data_cfg['batch_size']}, 3, {img_size[0]}, {img_size[1]}])")
    print(f"  Masks shape:  {masks.shape} (Expected: [{data_cfg['batch_size']}, 1, {img_size[0]}, {img_size[1]}])")
    
    # Assertions
    assert images.shape == (data_cfg['batch_size'], 3, *img_size), f"Image shape mismatch! Got {images.shape}"
    assert masks.shape == (data_cfg['batch_size'], 1, *img_size), f"Mask shape mismatch! Got {masks.shape}"
    
    print("\n✅ Verification Successful!")

if __name__ == "__main__":
    test_pipeline()
