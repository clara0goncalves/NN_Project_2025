# model/train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import json
from datetime import datetime
import sys
import matplotlib.pyplot as plt

# This allows finding the 'utils' and 'preprocessing' directories
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from .unet import get_model
from utils.losses import DiceLoss, CombinedLoss
from utils.prepare_data import train_loader, val_loader
from utils.metrics import dice_score, iou_score

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        self.model = get_model(
            n_channels=config['n_channels'],
            n_classes=config['n_classes']
        ).to(self.device)

        # ... (loss, optimizer, scheduler setup is unchanged)
        if config['loss_type'] == 'bce':
            self.criterion = nn.BCEWithLogitsLoss()
        elif config['loss_type'] == 'dice':
            self.criterion = DiceLoss()
        elif config['loss_type'] == 'combined':
            self.criterion = CombinedLoss()
        else:
            raise ValueError(f"Unknown loss type: {config['loss_type']}")

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            patience=config['scheduler_patience'],
            factor=config['scheduler_factor']
        )


        self.writer = SummaryWriter(f"runs/{config['experiment_name']}")
        self.checkpoint_dir = f"checkpoints/{config['experiment_name']}"
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.best_dice = 0.0
        self.early_stopping_counter = 0
        
        ## ADDED: Dictionary to store metrics history for plotting
        self.history = {
            'train_loss': [], 'train_dice': [], 'train_iou': [],
            'val_loss': [], 'val_dice': [], 'val_iou': []
        }
        
        config_save_path = os.path.join(self.writer.log_dir, 'config.json')
        with open(config_save_path, 'w') as f:
            json.dump(config, f, indent=2)

    # train_epoch and validate_epoch methods are unchanged...
    def train_epoch(self, dataloader, epoch):
        self.model.train()
        total_loss, total_dice, total_iou = 0, 0, 0

        pbar = tqdm(dataloader, desc=f'Training Epoch {epoch}')
        for images, masks in pbar:
            images, masks = images.to(self.device), masks.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            with torch.no_grad():
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                total_dice += dice_score(preds, masks).item()
                total_iou += iou_score(preds, masks).item()

            pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Dice': f'{total_dice/len(pbar):.4f}'})

        return total_loss / len(dataloader), total_dice / len(dataloader), total_iou / len(dataloader)

    def validate_epoch(self, dataloader, epoch):
        self.model.eval()
        total_loss, total_dice, total_iou = 0, 0, 0
        
        with torch.no_grad():
            pbar = tqdm(dataloader, desc=f'Validation Epoch {epoch}')
            for images, masks in pbar:
                images, masks = images.to(self.device), masks.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                total_loss += loss.item()
                
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                total_dice += dice_score(preds, masks).item()
                total_iou += iou_score(preds, masks).item()
                pbar.set_postfix({'Val Loss': f'{total_loss/len(pbar):.4f}', 'Val Dice': f'{total_dice/len(pbar):.4f}'})

        return total_loss / len(dataloader), total_dice / len(dataloader), total_iou / len(dataloader)


    ## ADDED: New method to save plots at the end of training
    def save_plots(self):
        epochs = range(1, len(self.history['train_loss']) + 1)

        plt.figure(figsize=(18, 5))

        # Plot Loss
        plt.subplot(1, 3, 1)
        plt.plot(epochs, self.history['train_loss'], label='Train Loss')
        plt.plot(epochs, self.history['val_loss'], label='Validation Loss')
        plt.title('Loss over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        # Plot Dice Score
        plt.subplot(1, 3, 2)
        plt.plot(epochs, self.history['train_dice'], label='Train Dice')
        plt.plot(epochs, self.history['val_dice'], label='Validation Dice')
        plt.title('Dice Score over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Dice Score')
        plt.legend()
        plt.grid(True)

        # Plot IoU Score
        plt.subplot(1, 3, 3)
        plt.plot(epochs, self.history['train_iou'], label='Train IoU')
        plt.plot(epochs, self.history['val_iou'], label='Validation IoU')
        plt.title('IoU Score over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('IoU Score')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        save_path = os.path.join(self.writer.log_dir, 'training_plots.png')
        plt.savefig(save_path)
        plt.close()
        print(f"Training plots saved to '{save_path}'")

    def train(self, train_loader, val_loader, num_epochs):
        print(f"Starting training for {num_epochs} epochs")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(1, num_epochs + 1):
            train_loss, train_dice, train_iou = self.train_epoch(train_loader, epoch)
            val_loss, val_dice, val_iou = self.validate_epoch(val_loader, epoch)
            
            # ... (scheduler and writer logging is unchanged)
            self.scheduler.step(val_dice)
            
            self.writer.add_scalar('Train/Loss_epoch', train_loss, epoch)
            self.writer.add_scalar('Train/Dice_epoch', train_dice, epoch)
            self.writer.add_scalar('Train/IoU_epoch', train_iou, epoch)
            self.writer.add_scalar('Val/Loss_epoch', val_loss, epoch)
            self.writer.add_scalar('Val/Dice_epoch', val_dice, epoch)
            self.writer.add_scalar('Val/IoU_epoch', val_iou, epoch)
            self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            ## ADDED: Append metrics to history
            self.history['train_loss'].append(train_loss)
            self.history['train_dice'].append(train_dice)
            self.history['train_iou'].append(train_iou)
            self.history['val_loss'].append(val_loss)
            self.history['val_dice'].append(val_dice)
            self.history['val_iou'].append(val_iou)
            
            print(f"\nEpoch {epoch}/{num_epochs} -> Train Loss: {train_loss:.4f}, Dice: {train_dice:.4f} | Val Loss: {val_loss:.4f}, Dice: {val_dice:.4f}")
            
            is_best = val_dice > self.best_dice
            if is_best:
                self.best_dice = val_dice
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1

            # ... (checkpoint saving and early stopping is unchanged)
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'dice_score': val_dice
            }
            if is_best:
                best_path = os.path.join(self.checkpoint_dir, 'best.pth')
                torch.save(checkpoint_data, best_path)
                print(f"New best model saved to '{best_path}' with Dice: {val_dice:.4f}")

            if epoch % self.config['save_checkpoint_freq'] == 0:
                freq_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
                torch.save(checkpoint_data, freq_path)
                print(f"Periodic checkpoint saved to '{freq_path}'")
            
            if self.early_stopping_counter >= self.config['early_stopping_patience']:
                print(f"\nEarly stopping triggered at epoch {epoch} as validation dice did not improve for {self.config['early_stopping_patience']} epochs.")
                break

        
        print(f"Training completed. Best Validation Dice Score: {self.best_dice:.4f}")
        self.writer.close()
        ## ADDED: Call to save plots at the end
        self.save_plots()

# The main function is unchanged
def main():
    config = {
        'experiment_name': f'unet_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        'n_channels': 3,
        'n_classes': 1,
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'num_epochs': 100,
        'loss_type': 'combined',
        'scheduler_patience': 10,
        'scheduler_factor': 0.1,
        'save_checkpoint_freq': 10,
        'early_stopping_patience': 15,
    }
    
    trainer = Trainer(config)
    trainer.train(train_loader, val_loader, config['num_epochs'])

if __name__ == "__main__":
    main()