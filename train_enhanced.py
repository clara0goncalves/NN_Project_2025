# train_enhanced.py
"""
Enhanced training pipeline with residual U-Net, dropout, and advanced features
- Residual blocks in encoder and bottleneck
- Configurable dropout rates
- Mixed precision training
- Advanced data augmentation
- Model complexity analysis
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import numpy as np
from datetime import datetime
import json

# Import the enhanced model
from models.unet_enhanced import get_enhanced_model
from utils.data_utils import WaterBodiesDataset
from utils.metrics import dice_score, iou_score
from utils.losses import DiceLoss, CombinedLoss, FocalLoss, TverskyLoss
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from preprocessing.prepare_data import train_loader, val_loader, test_loader


class EnhancedTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize logging first (before analyze_model_complexity)
        self.writer = SummaryWriter(f"runs/{config['experiment_name']}")
        self.best_val_loss = float('inf')
        self.best_dice = 0.0
        self.patience_counter = 0
        
        # Create checkpoint directory
        self.checkpoint_dir = f"checkpoints/{config['experiment_name']}"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Enhanced Model with residual blocks and dropout
        self.model = get_enhanced_model(
            n_channels=config['n_channels'],
            n_classes=config['n_classes'],
            bilinear=config['bilinear'],
            base_features=config['base_features'],
            encoder_dropout=config['encoder_dropout'],
            bottleneck_dropout=config['bottleneck_dropout']
        ).to(self.device)
        
        # Model complexity analysis (now writer is initialized)
        self.analyze_model_complexity()
        
        # Loss function with more options
        self.criterion = self._get_loss_function(config['loss_type'])
        
        # Optimizer with different options
        if config['optimizer'] == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=config['learning_rate'],
                weight_decay=config['weight_decay'],
                betas=config.get('betas', (0.9, 0.999))
            )
        elif config['optimizer'] == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=config['learning_rate'],
                weight_decay=config['weight_decay']
            )
        elif config['optimizer'] == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=config['learning_rate'],
                weight_decay=config['weight_decay'],
                momentum=config.get('momentum', 0.9)
            )
        
        # Enhanced scheduler options
        if config['scheduler_type'] == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, 
                mode='min', 
                patience=config['scheduler_patience'],
                factor=config['scheduler_factor'],
                verbose=True
            )
        elif config['scheduler_type'] == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config['num_epochs'],
                eta_min=config['learning_rate'] * 0.01
            )
        elif config['scheduler_type'] == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=config.get('step_size', 30),
                gamma=config.get('gamma', 0.1)
            )
        
        # Mixed precision training
        self.use_amp = config.get('use_amp', True) and torch.cuda.is_available()
        if self.use_amp:
            self.scaler = GradScaler()
            print("Using Automatic Mixed Precision (AMP)")
        
        # Save config
        with open(os.path.join(self.checkpoint_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)
    
    def _get_loss_function(self, loss_type):
        """Get loss function based on configuration"""
        if loss_type == 'bce':
            return nn.BCEWithLogitsLoss()
        elif loss_type == 'dice':
            return DiceLoss()
        elif loss_type == 'combined':
            return CombinedLoss()
        elif loss_type == 'focal':
            return FocalLoss(alpha=self.config.get('focal_alpha', 1.0), 
                           gamma=self.config.get('focal_gamma', 2.0))
        elif loss_type == 'tversky':
            return TverskyLoss(alpha=self.config.get('tversky_alpha', 0.7),
                             beta=self.config.get('tversky_beta', 0.3))
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def analyze_model_complexity(self):
        """Analyze and print model complexity"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Calculate model size in MB
        param_size = 0
        buffer_size = 0
        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()
        for buffer in self.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / 1024**2
        
        print(f"\nModel Complexity Analysis:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Model size: {size_mb:.2f} MB")
        print(f"  Base features: {self.config['base_features']}")
        print(f"  Encoder dropout: {self.config['encoder_dropout']}")
        print(f"  Bottleneck dropout: {self.config['bottleneck_dropout']}")
        
        # Log to tensorboard
        self.writer.add_text('Model/Parameters', f"Total: {total_params:,}, Trainable: {trainable_params:,}")
        self.writer.add_text('Model/Size_MB', f"{size_mb:.2f}")
    
    def train_epoch(self, dataloader, epoch):
        self.model.train()
        total_loss = 0
        total_dice = 0
        total_iou = 0
        
        pbar = tqdm(dataloader, desc=f'Training Epoch {epoch}')
        for batch_idx, (images, masks) in enumerate(pbar):
            images, masks = images.to(self.device), masks.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Mixed precision forward pass
            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
                
                # Mixed precision backward pass
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
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
                'IoU': f'{iou.item():.4f}',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
            
            # Log to tensorboard every 10 batches
            if batch_idx % 10 == 0:
                step = epoch * len(dataloader) + batch_idx
                self.writer.add_scalar('Train/Loss_step', loss.item(), step)
                self.writer.add_scalar('Train/Dice_step', dice.item(), step)
                self.writer.add_scalar('Train/IoU_step', iou.item(), step)
                self.writer.add_scalar('Train/Learning_Rate', self.optimizer.param_groups[0]['lr'], step)
        
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
                
                if self.use_amp:
                    with autocast():
                        outputs = self.model(images)
                        loss = self.criterion(outputs, masks)
                else:
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
        
        if self.use_amp:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save latest checkpoint
        torch.save(checkpoint, os.path.join(self.checkpoint_dir, 'latest.pth'))
        
        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, os.path.join(self.checkpoint_dir, 'best.pth'))
            print(f"New best model saved with Dice: {dice_score:.4f}")

    def train(self, train_loader, val_loader, num_epochs):
        print(f"Starting enhanced training for {num_epochs} epochs")
        
        for epoch in range(1, num_epochs + 1):
            # Training
            train_loss, train_dice, train_iou = self.train_epoch(train_loader, epoch)
            
            # Validation
            val_loss, val_dice, val_iou = self.validate_epoch(val_loader, epoch)
            
            # Scheduler step
            if self.config['scheduler_type'] == 'plateau':
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()
            
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
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            self.save_checkpoint(epoch, val_loss, val_dice, is_best)
            
            # Early stopping
            if self.config.get('early_stopping', False):
                if self.patience_counter >= self.config['early_stopping_patience']:
                    print(f"Early stopping at epoch {epoch} (patience: {self.patience_counter})")
                    break
        
        print(f"Training completed. Best Dice Score: {self.best_dice:.4f}")
        self.writer.close()


def main():
    # Enhanced Configuration with residual blocks and dropout
    config = {
        'experiment_name': f'enhanced_unet_residual_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        
        # Model architecture
        'n_channels': 3,
        'n_classes': 1,
        'bilinear': False,
        'base_features': 64,  # Can be 32, 64, 128 for different model sizes
        
        # Dropout configuration
        'encoder_dropout': 0.1,      # Dropout rate for encoder layers
        'bottleneck_dropout': 0.2,   # Higher dropout for bottleneck (deepest layer)
        
        # Training parameters
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'batch_size': 16,
        'num_epochs': 100,
        
        # Optimizer options: 'adam', 'adamw', 'sgd'
        'optimizer': 'adamw',
        'betas': (0.9, 0.999),  # For Adam/AdamW
        'momentum': 0.9,        # For SGD
        
        # Loss function options: 'bce', 'dice', 'combined', 'focal', 'tversky'
        'loss_type': 'combined',
        'focal_alpha': 1.0,     # For focal loss
        'focal_gamma': 2.0,     # For focal loss
        'tversky_alpha': 0.7,   # For Tversky loss
        'tversky_beta': 0.3,    # For Tversky loss
        
        # Scheduler options: 'plateau', 'cosine', 'step'
        'scheduler_type': 'plateau',
        'scheduler_patience': 10,
        'scheduler_factor': 0.5,
        'step_size': 30,        # For step scheduler
        'gamma': 0.1,           # For step scheduler
        
        # Advanced training features
        'use_amp': True,        # Automatic Mixed Precision
        'early_stopping': True,
        'early_stopping_patience': 20,
        
        # Regularization
        'gradient_clipping': True,
        'max_grad_norm': 1.0
    }
    
    print("Enhanced U-Net Training Configuration:")
    print(f"  Model: Residual U-Net with {config['base_features']} base features")
    print(f"  Encoder dropout: {config['encoder_dropout']}")
    print(f"  Bottleneck dropout: {config['bottleneck_dropout']}")
    print(f"  Loss function: {config['loss_type']}")
    print(f"  Optimizer: {config['optimizer']}")
    print(f"  Scheduler: {config['scheduler_type']}")
    print(f"  Mixed precision: {config['use_amp']}")
    
    # Create trainer and start training
    trainer = EnhancedTrainer(config)
    trainer.train(train_loader, val_loader, config['num_epochs'])


if __name__ == "__main__":
    main()