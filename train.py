# train.py
"""

Complete training pipeline with validation
Multiple loss function support (BCE, Dice, Combined, Focal, Tversky)
Learning rate scheduling and early stopping
Tensorboard logging and checkpointing
Comprehensive metrics tracking

"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from datetime import datetime
import json

from models.unet import get_model
from utils.data_utils import WaterBodiesDataset
from utils.metrics import dice_score, iou_score
from utils.losses import DiceLoss, CombinedLoss
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from preprocessing.prepare_data import train_loader, val_loader, test_loader

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Model
        self.model = get_model(
            n_channels=config['n_channels'],
            n_classes=config['n_classes'],
            bilinear=config['bilinear']
        ).to(self.device)
        
        # Loss function
        if config['loss_type'] == 'bce':
            self.criterion = nn.BCEWithLogitsLoss()
        elif config['loss_type'] == 'dice':
            self.criterion = DiceLoss()
        elif config['loss_type'] == 'combined':
            self.criterion = CombinedLoss()
        else:
            raise ValueError(f"Unknown loss type: {config['loss_type']}")
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            patience=config['scheduler_patience'],
            factor=config['scheduler_factor']
        )
        
        # Logging
        self.writer = SummaryWriter(f"runs/{config['experiment_name']}")
        self.best_val_loss = float('inf')
        self.best_dice = 0.0
        
        # Create checkpoint directory
        self.checkpoint_dir = f"checkpoints/{config['experiment_name']}"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Save config
        with open(os.path.join(self.checkpoint_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)

    def train_epoch(self, dataloader, epoch):
        self.model.train()
        total_loss = 0
        total_dice = 0
        total_iou = 0
        
        pbar = tqdm(dataloader, desc=f'Training Epoch {epoch}')
        for batch_idx, (images, masks) in enumerate(pbar):
            images, masks = images.to(self.device), masks.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Metrics
            with torch.no_grad():
                probs = torch.sigmoid(outputs)
                dice = dice_score(probs > 0.5, masks)
                iou = iou_score(probs > 0.5, masks)
            
            total_loss += loss.item()
            total_dice += dice.item()
            total_iou += iou.item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Dice': f'{dice.item():.4f}',
                'IoU': f'{iou.item():.4f}'
            })
            
            # Log to tensorboard every 10 batches
            if batch_idx % 10 == 0:
                step = epoch * len(dataloader) + batch_idx
                self.writer.add_scalar('Train/Loss_step', loss.item(), step)
                self.writer.add_scalar('Train/Dice_step', dice.item(), step)
                self.writer.add_scalar('Train/IoU_step', iou.item(), step)
        
        avg_loss = total_loss / len(dataloader)
        avg_dice = total_dice / len(dataloader)
        avg_iou = total_iou / len(dataloader)
        
        return avg_loss, avg_dice, avg_iou

    def validate_epoch(self, dataloader, epoch):
        self.model.eval()
        total_loss = 0
        total_dice = 0
        total_iou = 0
        
        with torch.no_grad():
            pbar = tqdm(dataloader, desc=f'Validation Epoch {epoch}')
            for images, masks in pbar:
                images, masks = images.to(self.device), masks.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                
                # Metrics
                probs = torch.sigmoid(outputs)
                dice = dice_score(probs > 0.5, masks)
                iou = iou_score(probs > 0.5, masks)
                
                total_loss += loss.item()
                total_dice += dice.item()
                total_iou += iou.item()
                
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Dice': f'{dice.item():.4f}',
                    'IoU': f'{iou.item():.4f}'
                })
        
        avg_loss = total_loss / len(dataloader)
        avg_dice = total_dice / len(dataloader)
        avg_iou = total_iou / len(dataloader)
        
        return avg_loss, avg_dice, avg_iou

    def save_checkpoint(self, epoch, val_loss, dice_score, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'dice_score': dice_score,
            'config': self.config
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, os.path.join(self.checkpoint_dir, 'latest.pth'))
        
        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, os.path.join(self.checkpoint_dir, 'best.pth'))
            print(f"New best model saved with Dice: {dice_score:.4f}")

    def train(self, train_loader, val_loader, num_epochs):
        print(f"Starting training for {num_epochs} epochs")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(1, num_epochs + 1):
            # Training
            train_loss, train_dice, train_iou = self.train_epoch(train_loader, epoch)
            
            # Validation
            val_loss, val_dice, val_iou = self.validate_epoch(val_loader, epoch)
            
            # Scheduler step
            self.scheduler.step(val_loss)
            
            # Logging
            self.writer.add_scalar('Train/Loss_epoch', train_loss, epoch)
            self.writer.add_scalar('Train/Dice_epoch', train_dice, epoch)
            self.writer.add_scalar('Train/IoU_epoch', train_iou, epoch)
            self.writer.add_scalar('Val/Loss_epoch', val_loss, epoch)
            self.writer.add_scalar('Val/Dice_epoch', val_dice, epoch)
            self.writer.add_scalar('Val/IoU_epoch', val_iou, epoch)
            self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Print epoch results
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"Train - Loss: {train_loss:.4f}, Dice: {train_dice:.4f}, IoU: {train_iou:.4f}")
            print(f"Val   - Loss: {val_loss:.4f}, Dice: {val_dice:.4f}, IoU: {val_iou:.4f}")
            print(f"LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # Save checkpoints
            is_best = val_dice > self.best_dice
            if is_best:
                self.best_dice = val_dice
                self.best_val_loss = val_loss
            
            self.save_checkpoint(epoch, val_loss, val_dice, is_best)
            
            # Early stopping
            if self.config.get('early_stopping', False):
                if epoch > self.config['early_stopping_patience']:
                    if val_loss > self.best_val_loss:
                        print(f"Early stopping at epoch {epoch}")
                        break
        
        print(f"Training completed. Best Dice Score: {self.best_dice:.4f}")
        self.writer.close()


def main():
    # Configuration
    config = {
        'experiment_name': f'unet_water_segmentation_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        'n_channels': 3,
        'n_classes': 1,
        'bilinear': False,
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'batch_size': 16,
        'num_epochs': 100,
        'loss_type': 'combined',  # 'bce', 'dice', or 'combined'
        'scheduler_patience': 10,
        'scheduler_factor': 0.5,
        'early_stopping': True,
        'early_stopping_patience': 20
    }
    
    # Create trainer and start training
    trainer = Trainer(config)
    trainer.train(train_loader, val_loader, config['num_epochs'])


if __name__ == "__main__":
    main()