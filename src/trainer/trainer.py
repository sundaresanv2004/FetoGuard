import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from src.utils.logger import Logger

class Trainer:
    def __init__(self, model, loaders, config, logger: Logger):
        self.model = model
        self.loaders = loaders
        self.config = config
        self.logger = logger
        self.logger = logger
        
        # Device Logic
        device_str = config['train']['device']
        if device_str == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device_str)
            
        print(f"Using device: {self.device}")
        
        self.model = self.model.to(self.device)
        
        # Loss: BCEWithLogitsLoss (Standard for binary seg) + Dice (optional, typically manual impl or from library)
        # Using simple BCEWithLogits for stability first
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Optimizer
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config['train']['learning_rate'])
        
        # Resume
        self.start_epoch = 0
        self.best_metric = 0.0

    def load_checkpoint(self, checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_metric = checkpoint.get('metric', 0.0)
        print(f"Resumed from epoch {self.start_epoch}")

    def train_one_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        pbar = tqdm(self.loaders['train'], desc=f"Epoch {epoch}/{self.config['train']['epochs']}")
        
        for images, masks in pbar:
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix({'loss': running_loss / (pbar.n + 1)})
            
        return running_loss / len(self.loaders['train'])

    def evaluate(self, phase='val'):
        loader = self.loaders[phase]
        self.model.eval()
        running_loss = 0.0
        dice_score = 0.0
        
        with torch.no_grad():
            for images, masks in tqdm(loader, desc=f"Evaluating ({phase})"):
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                running_loss += loss.item()
                
                # Dice Score Calculation
                preds = (torch.sigmoid(outputs) > 0.5).float()
                dice_score += (2. * (preds * masks).sum()) / (preds.sum() + masks.sum() + 1e-8)
                
        epoch_loss = running_loss / len(loader)
        epoch_dice = dice_score / len(loader)
        return epoch_loss, epoch_dice.item()

    def run(self):
        dataset_len = len(self.loaders['train'].dataset)
        print(f"Starting training on {self.device} with {dataset_len} samples.")
        
        for epoch in range(self.start_epoch, self.config['train']['epochs']):
            train_loss = self.train_one_epoch(epoch)
            val_loss, val_dice = self.evaluate('val')
            
            print(f"Epoch {epoch}: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}")
            
            # Log
            metrics = {
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_dice': val_dice,
                'lr': self.optimizer.param_groups[0]['lr']
            }
            self.logger.log_metrics(metrics)
            
            # Save
            is_best = val_dice > self.best_metric
            if is_best:
                self.best_metric = val_dice
                
            self.logger.save_model(self.model, self.optimizer, epoch, val_dice, is_best)
