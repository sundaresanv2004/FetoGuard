import os
import csv
from datetime import datetime
import torch
import pandas as pd
import matplotlib.pyplot as plt

class Logger:
    def __init__(self, save_dir):
        """
        Initializes the Logger.
        
        Args:
            save_dir (str): Base output directory.
        """
        # Create timestamped directory: output/dd-mm-yyyy/timestamp/
        now = datetime.now()
        date_str = now.strftime("%d-%m-%Y")
        time_str = now.strftime("%H-%M-%S")
        
        self.experiment_dir = os.path.join(os.path.abspath(save_dir), date_str, time_str)
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        print(f"Logging experiment to: {self.experiment_dir}")
        
        self.log_file = os.path.join(self.experiment_dir, "training_log.csv")
        self.fields = ['epoch', 'train_loss', 'val_loss', 'val_dice', 'lr']
        
        # Initialize CSV
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fields)
            writer.writeheader()
            
        self.best_metric = 0.0 # Assuming higher is better (Dice)

    def log_metrics(self, metrics):
        """
        Logs a dictionary of metrics to CSV.
        """
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fields)
            writer.writeheader() # Ensure header is written if file is empty (handled in init, but safe)
            # wait, DictWriter doesn't auto-append header if missing in 'a' mode, init handled it.
            # Just write row.
            writer.writerow(metrics)
            
    def save_model(self, model, optimizer, epoch, metric, is_best=False):
        """
        Saves the model checkpoint.
        """
        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metric': metric,
        }
        
        # Save Last
        last_path = os.path.join(self.experiment_dir, "last_model.pth")
        torch.save(state, last_path)
        
        # Save Best
        if is_best:
            best_path = os.path.join(self.experiment_dir, "best_model.pth")
            torch.save(state, best_path)
            print(f"Saved best model with metric: {metric:.4f}")
            
    def get_checkpoint_path(self, name="last_model.pth"):
        return os.path.join(self.experiment_dir, name)

    def plot_metrics(self):
        """
        Reads the log file and plots training metrics.
        """
        graphs_dir = os.path.join(self.experiment_dir, "graphs")
        Logger.plot_from_csv(self.log_file, graphs_dir)

    @staticmethod
    def plot_from_csv(csv_path, output_dir):
        """
        Static method to plot metrics from any CSV file to an output directory.
        """
        try:
            df = pd.read_csv(csv_path)
            
            os.makedirs(output_dir, exist_ok=True)
            
            # 1. Loss Plot
            plt.figure(figsize=(10, 5))
            plt.plot(df['epoch'], df['train_loss'], label='Train Loss')
            plt.plot(df['epoch'], df['val_loss'], label='Val Loss')
            plt.title('Training & Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, "loss_plot.png"))
            plt.close()
            
            # 2. Dice Plot
            plt.figure(figsize=(10, 5))
            plt.plot(df['epoch'], df['val_dice'], label='Val Dice', color='green')
            plt.title('Validation Dice Score')
            plt.xlabel('Epoch')
            plt.ylabel('Dice Score')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, "dice_plot.png"))
            plt.close()
            
            print(f"Graphs saved to: {output_dir}")
            
        except Exception as e:
            print(f"Error plotting metrics: {e}")
