import argparse
import sys
import torch
import random
import numpy as np
from src.utils.config import load_config
from src.utils.logger import Logger
from src.data_loader.dataset import get_dataloaders
from src.data_loader.transforms import get_transforms
from src.models.model_factory import get_model
from src.trainer.trainer import Trainer

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_args():
    parser = argparse.ArgumentParser(description="FetoGuard Training/Testing Pipeline")
    
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config file')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help='Execution mode')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint to load')
    
    # Overrides
    parser.add_argument('--epochs', type=int, help='Override number of epochs')
    parser.add_argument('--batch-size', type=int, help='Override batch size')
    parser.add_argument('--lr', type=float, help='Override learning rate')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load Config
    config = load_config(args.config)
    
    # Apply Overrides
    if args.epochs: config['train']['epochs'] = args.epochs
    if args.batch_size: config['data']['batch_size'] = args.batch_size
    if args.lr: config['train']['learning_rate'] = args.lr
    
    # Set Seed
    set_seed(config['train']['seed'])
    
    # Init Logger
    logger = Logger(save_dir=config['train']['save_dir'])
    
    # Data Setup
    print("Initializing Data Loaders...")
    img_size = tuple(config['data']['input_size'])
    train_transform = get_transforms(mode='train', img_size=img_size)
    val_transform = get_transforms(mode='val', img_size=img_size)
    
    loaders = get_dataloaders(
        data_dir=config['data']['data_dir'],
        batch_size=config['data']['batch_size'],
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=val_transform,
        val_split=config['data']['val_split'],
        num_workers=config['data']['num_workers']
    )
    
    # Model Setup
    print("Initializing Model...")
    model = get_model(config)
    
    # Trainer
    trainer = Trainer(model, loaders, config, logger)
    
    # Resume / Load Checkpoint
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)
        
    # Execution
    if args.mode == 'train':
        trainer.run()
        
    elif args.mode == 'test':
        loss, dice = trainer.evaluate('test')
        print(f"Test Results - Loss: {loss:.4f}, Dice: {dice:.4f}")

if __name__ == "__main__":
    main()
