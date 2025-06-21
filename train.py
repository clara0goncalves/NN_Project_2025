# train_unified.py
"""
Unified training pipeline supporting multiple U-Net architectures.
Now includes automatic history logging (CSV) with hyperparameters and plotting of training metrics.
- Basic U-Net (unet)
- Enhanced U-Net with residual blocks (enhanced) 
- Attention U-Net (attention)

Usage:
    python train_unified.py --model unet
    python train_unified.py --model enhanced --base_features 64 --encoder_dropout 0.1
    python train_unified.py --model attention --loss_type focal --optimizer adamw

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
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import seaborn as sns
import segmentation_models_pytorch.losses as smp_losses

# Model imports
from models.unet import get_model as get_basic_model
from models.unet_enhanced import get_enhanced_model
from models.attention import get_attention_model
from models.unet_plus_plus import get_unet_plus_plus_model
from models.aer_unet import get_aer_unet_model
from models.segformer import get_segformer_model

from utils.data_utils import WaterBodiesDataset
from utils.metrics import dice_score, iou_score
from utils.losses import DiceLoss, CombinedLoss, FocalLoss, TverskyLoss, FocalLovaszLoss

# Model imports - adjust these based on your actual model files
from models.unet import get_model as get_basic_model
from models.unet_enhanced import get_enhanced_model
from models.attention import get_attention_model

from utils.data_utils import WaterBodiesDataset
from utils.metrics import dice_score, iou_score
from utils.losses import DiceLoss, CombinedLoss, FocalLoss, TverskyLoss
import sys

# Ensure the project root is in the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing.prepare_data import train_loader, val_loader, test_loader

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize logging first
        self.writer = SummaryWriter(f"runs/{config['experiment_name']}")
        self.best_val_loss = float('inf')
        self.best_iou = 0.0
        self.patience_counter = 0
        # History tracking
        self.history = []
        
        # Create checkpoint directory
        self.checkpoint_dir = f"checkpoints/{config['experiment_name']}"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Initialize model based on type
        self.model = self._get_model(config).to(self.device)
        
        # Model complexity analysis
        self.analyze_model_complexity()
        
        # Loss function
        self.criterion = self._get_loss_function(config['loss_type'])
        
        # Optimizer
        self.optimizer = self._get_optimizer(config)
        
        # Scheduler
        self.scheduler = self._get_scheduler(config)
        
        # Mixed precision training
        self.use_amp = (config.get('use_amp', False) and 
                       torch.cuda.is_available() and 
                       config['model_type'] in ['enhanced', 'attention', 'unet++', 'aer-unet'])

        if self.use_amp:
            self.scaler = GradScaler()
            print("Using Automatic Mixed Precision (AMP)")
        
        # Save config
        with open(os.path.join(self.checkpoint_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)
    
    def _get_model(self, config):
        """Initialize model based on type"""
        model_type = config['model_type']
        
        if model_type == 'unet++-pretrained-encoder':
            print("Initializing U-Net with pre-trained efficientnet64 encoder.")
            return smp.UnetPlusPlus(
                encoder_name="efficientnet-b4",
                encoder_weights="imagenet",
                in_channels=config['n_channels'],
                classes=config['n_classes'],
            )
        elif model_type == 'segformer-b4':
            print("Initializing SegFormer-B4 model pre-trained on ADE20K.")
            return get_segformer_model(
                n_classes=config['n_classes']
            )
        elif model_type == 'unet':
            return get_basic_model(
                n_channels=config['n_channels'],
                n_classes=config['n_classes'],
                bilinear=config['bilinear']
            )
        elif model_type == 'enhanced':
            return get_enhanced_model(
                n_channels=config['n_channels'],
                n_classes=config['n_classes'],
                bilinear=config['bilinear'],
                base_features=config['base_features'],
                encoder_dropout=config['encoder_dropout'],
                bottleneck_dropout=config['bottleneck_dropout']
            )
        elif model_type == 'attention':
            return get_attention_model(
                n_channels=config['n_channels'],
                n_classes=config['n_classes'],
                bilinear=config['bilinear'],
                base_features=config['base_features'],
                encoder_dropout=config['encoder_dropout'],
                bottleneck_dropout=config['bottleneck_dropout']
            )
        elif model_type == 'unet++':
            return get_unet_plus_plus_model(
                n_channels=config['n_channels'],
                n_classes=config['n_classes'],
                bilinear=config['bilinear'],
                base_features=config['base_features'],
                deep_supervision=config.get('deep_supervision', False)
            )
        elif model_type == 'aer-unet':
            return get_aer_unet_model(
                n_channels=config['n_channels'],
                n_classes=config['n_classes'],
                base_features=config['base_features'],
                dropout_rate=config['dropout_rate']
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _get_loss_function(self, loss_type):
        """Get loss function based on configuration"""
        if loss_type == 'bce':
            return nn.BCEWithLogitsLoss()
        elif loss_type == 'dice':
            return DiceLoss()
        elif loss_type == 'combined':
            return CombinedLoss()
        elif loss_type == 'focal':
            return FocalLoss(
                alpha=self.config.get('focal_alpha', 1.0),
                gamma=self.config.get('focal_gamma', 2.0)
            )
        elif loss_type == 'tversky':
            return TverskyLoss(
                alpha=self.config.get('tversky_alpha', 0.7),
                beta=self.config.get('tversky_beta', 0.3)
            )
        elif loss_type == 'focal_lovasz':
            print("Using combined Focal + Lovasz loss.")
            return FocalLovaszLoss(
            focal_weight=self.config.get('focal_weight', 0.5),
            lovasz_weight=self.config.get('lovasz_weight', 0.5)
            )
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def _get_optimizer(self, config):
        """Initialize optimizer based on configuration"""
        if config['optimizer'] == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=config['learning_rate'],
                weight_decay=config['weight_decay'],
                betas=config.get('betas', (0.9, 0.999))
            )
        elif config['optimizer'] == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=config['learning_rate'],
                weight_decay=config['weight_decay']
            )
        elif config['optimizer'] == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=config['learning_rate'],
                weight_decay=config['weight_decay'],
                momentum=config.get('momentum', 0.9)
            )
        else:
            raise ValueError(f"Unknown optimizer: {config['optimizer']}")
    
    def _get_scheduler(self, config):
        """Initialize scheduler based on configuration"""
        if config['scheduler_type'] == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=config['scheduler_patience'],
                factor=config['scheduler_factor'],
                verbose=True
            )
        elif config['scheduler_type'] == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config['num_epochs'],
                eta_min=config['learning_rate'] * 0.01
            )
        elif config['scheduler_type'] == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=config.get('step_size', 30),
                gamma=config.get('gamma', 0.1)
            )
        else:
            raise ValueError(f"Unknown scheduler type: {config['scheduler_type']}")
    
    def analyze_model_complexity(self):
        """Analyze and print model complexity"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
       
        param_size = sum(param.nelement() * param.element_size() for param in self.model.parameters())
        buffer_size = sum(buffer.nelement() * buffer.element_size() for buffer in self.model.buffers())
        size_mb = (param_size + buffer_size) / 1024**2
        
        print(f"\nModel Complexity Analysis:")
        print(f"  Model type: {self.config['model_type']}")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Model size: {size_mb:.2f} MB")
        
        if self.config['model_type'] in ['enhanced', 'attention', 'unet++', 'aer-unet']:
            print(f"  Base features: {self.config['base_features']}")
        if self.config['model_type'] in ['enhanced', 'attention']:
            print(f"  Encoder dropout: {self.config['encoder_dropout']}")
            print(f"  Bottleneck dropout: {self.config['bottleneck_dropout']}")
        if self.config['model_type'] == 'unet++':
            print(f"  Deep Supervision: {self.config.get('deep_supervision', False)}")
        
        if self.config['model_type'] in ['enhanced', 'attention']:
            print(f"  Base features: {self.config['base_features']}")
            print(f"  Encoder dropout: {self.config['encoder_dropout']}")
            print(f"  Bottleneck dropout: {self.config['bottleneck_dropout']}")
        
        # Log to tensorboard
        self.writer.add_text('Model/Type', self.config['model_type'])
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
           
            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    if isinstance(outputs, list):
                        loss = sum(self.criterion(o, masks) for o in outputs)
                        outputs = outputs[-1] 
                    else:
                        loss = self.criterion(outputs, masks)
                self.scaler.scale(loss).backward()
                if self.config.get('gradient_clipping', False):
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.get('max_grad_norm', 1.0))
            # Forward pass with optional mixed precision
            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
                
                # Mixed precision backward pass
                self.scaler.scale(loss).backward()
                
                # Gradient clipping if enabled
                if self.config.get('gradient_clipping', False):
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.get('max_grad_norm', 1.0)
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                if isinstance(outputs, list):
                    loss = sum(self.criterion(o, masks) for o in outputs)
                    outputs = outputs[-1]
                else:
                    loss = self.criterion(outputs, masks)
                loss.backward()
                if self.config.get('gradient_clipping', False):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.get('max_grad_norm', 1.0))
                self.optimizer.step()
            
                loss = self.criterion(outputs, masks)
                loss.backward()
                
                # Gradient clipping if enabled
                if self.config.get('gradient_clipping', False):
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.get('max_grad_norm', 1.0)
                    )
                
                self.optimizer.step()
            
            # Metrics calculation
            with torch.no_grad():
                probs = torch.sigmoid(outputs)
                dice = dice_score(probs > 0.5, masks)
                iou = iou_score(probs > 0.5, masks)
            
            total_loss += loss.item()
            total_dice += dice.item()
            total_iou += iou.item()
            
            pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Dice': f'{dice.item():.4f}', 'IoU': f'{iou.item():.4f}', 'LR': f'{self.optimizer.param_groups[0]["lr"]:.2e}'})
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Dice': f'{dice.item():.4f}',
                'IoU': f'{iou.item():.4f}',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
            
            if batch_idx % 10 == 0:
                step = epoch * len(dataloader) + batch_idx
                self.writer.add_scalar('Train/Loss_step', loss.item(), step)
                self.writer.add_scalar('Train/Dice_step', dice.item(), step)
                self.writer.add_scalar('Train/IoU_step', iou.item(), step)
                self.writer.add_scalar('Train/Learning_Rate', self.optimizer.param_groups[0]['lr'], step)
        
        return total_loss / len(dataloader), total_dice / len(dataloader), total_iou / len(dataloader)

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
                        if isinstance(outputs, list):
                            loss = sum(self.criterion(o, masks) for o in outputs)
                            outputs = outputs[-1]
                        else:
                            loss = self.criterion(outputs, masks)
                else:
                    outputs = self.model(images)
                    if isinstance(outputs, list):
                        loss = sum(self.criterion(o, masks) for o in outputs)
                        outputs = outputs[-1]
                    else:
                        loss = self.criterion(outputs, masks)
                
                probs = torch.sigmoid(outputs)
                dice = dice_score(probs > 0.5, masks)
                iou = iou_score(probs > 0.5, masks)
                
                total_loss += loss.item()
                total_dice += dice.item()
                total_iou += iou.item()
                
                pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Dice': f'{dice.item():.4f}', 'IoU': f'{iou.item():.4f}'})
        
        return total_loss / len(dataloader), total_dice / len(dataloader), total_iou / len(dataloader)

    def save_checkpoint(self, epoch, val_loss, val_iou, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'iou_score': val_iou,
            'config': self.config
        }
        
        if self.use_amp:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, os.path.join(self.checkpoint_dir, 'latest.pth'))
        
        if is_best:
            torch.save(checkpoint, os.path.join(self.checkpoint_dir, 'best.pth'))
            print(f"New best model saved with IoU: {val_iou:.4f}")
        save_interval = 20
        if epoch > 0 and epoch % save_interval == 0:
            periodic_path = os.path.join(self.checkpoint_dir, f'epoch_{epoch}.pth')
            torch.save(checkpoint, periodic_path)
            print(f"Saved periodic checkpoint at epoch {epoch} to '{periodic_path}'")

    def _plot_and_save_history(self):
        """
        Plots training metrics and saves the history, including all hyperparameters, to a CSV file.
        """
        history_df = pd.DataFrame(self.history)
        
        for key, value in self.config.items():
            history_df[key] = value

        csv_path = os.path.join(self.checkpoint_dir, 'training_history.csv')
        history_df.to_csv(csv_path, index=False)
        
        # --- FIX: Use the modern seaborn function for styling ---
        sns.set_theme(style="darkgrid")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # Plotting Loss
        ax1.plot(history_df['epoch'], history_df['train_loss'], 'o-', label='Train Loss')
        ax1.plot(history_df['epoch'], history_df['val_loss'], 'o-', label='Validation Loss')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training & Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plotting IoU
        ax2.plot(history_df['epoch'], history_df['train_iou'], 'o-', label='Train IoU')
        ax2.plot(history_df['epoch'], history_df['val_iou'], 'o-', label='Validation IoU')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('IoU Score')
        ax2.set_title('Training & Validation IoU')
        ax2.legend()
        ax2.grid(True)
        
        fig.tight_layout()
        plot_path = os.path.join(self.checkpoint_dir, 'training_plots.png')
        plt.savefig(plot_path)
        plt.close()
        
        print(f"\nTraining history (with hyperparameters) and plots saved to: {self.checkpoint_dir}")

    def train(self, train_loader, val_loader, num_epochs):
        print(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(1, num_epochs + 1):
            train_loss, train_dice, train_iou = self.train_epoch(train_loader, epoch)
            val_loss, val_dice, val_iou = self.validate_epoch(val_loader, epoch)
            
            self.history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_dice': train_dice,
                'train_iou': train_iou,
                'val_loss': val_loss,
                'val_dice': val_dice,
                'val_iou': val_iou,
                'lr': self.optimizer.param_groups[0]['lr']
            })

            if self.config['scheduler_type'] == 'plateau':
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()
            
            self.writer.add_scalar('Train/Loss_epoch', train_loss, epoch)
            self.writer.add_scalar('Train/Dice_epoch', train_dice, epoch)
            self.writer.add_scalar('Train/IoU_epoch', train_iou, epoch)
            self.writer.add_scalar('Val/Loss_epoch', val_loss, epoch)
            self.writer.add_scalar('Val/Dice_epoch', val_dice, epoch)
            self.writer.add_scalar('Val/IoU_epoch', val_iou, epoch)
            self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"Train - Loss: {train_loss:.4f}, Dice: {train_dice:.4f}, IoU: {train_iou:.4f}")
            print(f"Val   - Loss: {val_loss:.4f}, Dice: {val_dice:.4f}, IoU: {val_iou:.4f}")
            print(f"LR: {self.optimizer.param_groups[0]['lr']:.2e}")
           
            is_best = val_iou > self.best_iou
            if is_best:
                self.best_iou = val_iou
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            if self.config.get('early_stopping', False):
                print(f"Patience: {self.patience_counter} / {self.config['early_stopping_patience']}")
            
            self.save_checkpoint(epoch, val_loss, val_iou, is_best)
            
            if self.config.get('early_stopping', False) and self.patience_counter >= self.config['early_stopping_patience']:
                print(f"Early stopping at epoch {epoch} (patience: {self.patience_counter})")
                break
        
        print(f"Training completed. Best IoU Score: {self.best_iou:.4f}")
            
        self.save_checkpoint(epoch, val_loss, val_iou, is_best)
        
        print(f"Training completed. Best Dice Score: {self.best_iou:.4f}")
        self.writer.close()
        
        self._plot_and_save_history()

def parse_args():
    parser = argparse.ArgumentParser(description='Unified U-Net Training Script')
    
    parser.add_argument('--model', type=str, 
                        choices=['unet', 'enhanced', 'attention', 'unet++', 'aer-unet', 'unet++-pretrained-encoder', 'segformer-b4'],
                        default='unet', 
                        help='Model architecture to use')
    parser.add_argument('--n_channels', type=int, default=3, help='Number of input channels')
    parser.add_argument('--n_classes', type=int, default=1, help='Number of output classes')
    parser.add_argument('--bilinear', action='store_true', help='Use bilinear upsampling')
    parser.add_argument('--base_features', type=int, default=32, help='Base features for advanced models')
    parser.add_argument('--encoder_dropout', type=float, default=0.1, help='Dropout for encoder (enhanced/attention)')
    parser.add_argument('--bottleneck_dropout', type=float, default=0.2, help='Dropout for bottleneck (enhanced/attention)')
    parser.add_argument('--deep_supervision', action='store_true', help='Enable deep supervision for U-Net++')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--optimizer', type=str, choices=['adam', 'adamw', 'sgd'], default='adam', help='Optimizer type')
    parser.add_argument('--scheduler_type', type=str, choices=['plateau', 'cosine', 'step'], default='plateau', help='Scheduler type')
    parser.add_argument('--scheduler_patience', type=int, default=10, help='Patience for plateau scheduler')
    parser.add_argument('--scheduler_factor', type=float, default=0.5, help='Factor for plateau scheduler')
    parser.add_argument('--loss_type', type=str, 
                        choices=['bce', 'dice', 'combined', 'focal', 'tversky', 'lovasz', 'focal_lovasz'], # Add 'focal_lovasz' and 'lovasz'
                        default='combined', 
                        help='Loss function type')
    parser.add_argument('--dropout_rate', type=float, default=0.3, help='Dropout rate for models that support it (AER-UNet)')
    parser.add_argument('--focal_weight', type=float, default=0.5, help='Weight for Focal Loss in FocalLovaszLoss')
    parser.add_argument('--lovasz_weight', type=float, default=0.5, help='Weight for Lovasz Loss in FocalLovaszLoss')
    parser.add_argument('--use_amp', action='store_true', help='Use automatic mixed precision')
    parser.add_argument('--early_stopping', action='store_true', help='Enable early stopping')
    parser.add_argument('--early_stopping_patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--gradient_clipping', action='store_true', help='Enable gradient clipping')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Max gradient norm')
    parser.add_argument('--experiment_name', type=str, default=None, help='Custom experiment name')
    
    return parser.parse_args()

def parse_args():
    parser = argparse.ArgumentParser(description='Unified U-Net Training Script')
    
    # Model selection
    parser.add_argument('--model', type=str, choices=['unet', 'enhanced', 'attention'],
                       default='unet', help='Model architecture to use')
    
    # Basic model parameters
    parser.add_argument('--n_channels', type=int, default=3, help='Number of input channels')
    parser.add_argument('--n_classes', type=int, default=1, help='Number of output classes')
    parser.add_argument('--bilinear', action='store_true', help='Use bilinear upsampling')
    
    # Enhanced/Attention model parameters
    parser.add_argument('--base_features', type=int, default=64, 
                       help='Base number of features (enhanced/attention models)')
    parser.add_argument('--encoder_dropout', type=float, default=0.1,
                       help='Dropout rate for encoder layers')
    parser.add_argument('--bottleneck_dropout', type=float, default=0.2,
                       help='Dropout rate for bottleneck layer')
    
    # Training parameters
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    
    # Optimizer and scheduler
    parser.add_argument('--optimizer', type=str, choices=['adam', 'adamw', 'sgd'],
                       default='adam', help='Optimizer type')
    parser.add_argument('--scheduler_type', type=str, choices=['plateau', 'cosine', 'step'],
                       default='plateau', help='Scheduler type')
    parser.add_argument('--scheduler_patience', type=int, default=10,
                       help='Patience for plateau scheduler')
    parser.add_argument('--scheduler_factor', type=float, default=0.5,
                       help='Factor for plateau scheduler')
    
    # Loss function
    parser.add_argument('--loss_type', type=str, 
                       choices=['bce', 'dice', 'combined', 'focal', 'tversky'],
                       default='combined', help='Loss function type')
    
    # Advanced features
    parser.add_argument('--use_amp', action='store_true', help='Use automatic mixed precision')
    parser.add_argument('--early_stopping', action='store_true', help='Enable early stopping')
    parser.add_argument('--early_stopping_patience', type=int, default=20,
                       help='Early stopping patience')
    parser.add_argument('--gradient_clipping', action='store_true', help='Enable gradient clipping')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Max gradient norm')
    
    # Experiment naming
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Custom experiment name')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.experiment_name = f"{args.model}_water_segmentation_{timestamp}"
    
    config = vars(args)
    config['model_type'] = config.pop('model')

    print("Training Configuration:")
    for key, value in config.items():
        if key in ['base_features', 'encoder_dropout', 'bottleneck_dropout', 'deep_supervision'] and config['model_type'] not in ['enhanced', 'attention', 'unet++', 'aer-unet']:
            continue
        print(f"  {key.replace('_', ' ').capitalize()}: {value}")
    
    # Create experiment name if not provided
    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.experiment_name = f"{args.model}_water_segmentation_{timestamp}"
    
    # Convert args to config dictionary
    config = {
        'experiment_name': args.experiment_name,
        'model_type': args.model,
        'n_channels': args.n_channels,
        'n_classes': args.n_classes,
        'bilinear': args.bilinear,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'optimizer': args.optimizer,
        'scheduler_type': args.scheduler_type,
        'scheduler_patience': args.scheduler_patience,
        'scheduler_factor': args.scheduler_factor,
        'loss_type': args.loss_type,
        'use_amp': args.use_amp,
        'early_stopping': args.early_stopping,
        'early_stopping_patience': args.early_stopping_patience,
        'gradient_clipping': args.gradient_clipping,
        'max_grad_norm': args.max_grad_norm,
    }
    
    # Add enhanced/attention specific parameters
    if args.model in ['enhanced', 'attention']:
        config.update({
            'base_features': args.base_features,
            'encoder_dropout': args.encoder_dropout,
            'bottleneck_dropout': args.bottleneck_dropout,
        })
    
    # Print configuration
    print("Training Configuration:")
    print(f"  Model: {config['model_type']}")
    print(f"  Loss function: {config['loss_type']}")
    print(f"  Optimizer: {config['optimizer']}")
    print(f"  Scheduler: {config['scheduler_type']}")
    print(f"  Learning rate: {config['learning_rate']}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Epochs: {config['num_epochs']}")
    
    if config['model_type'] in ['enhanced', 'attention']:
        print(f"  Base features: {config['base_features']}")
        print(f"  Encoder dropout: {config['encoder_dropout']}")
        print(f"  Bottleneck dropout: {config['bottleneck_dropout']}")
        print(f"  Mixed precision: {config['use_amp']}")
    
    print(f"  Early stopping: {config['early_stopping']}")
    print(f"  Experiment: {config['experiment_name']}")
    
    # Create trainer and start training
    trainer = Trainer(config)
    
    train_loader_adj = DataLoader(train_loader.dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
    val_loader_adj = DataLoader(val_loader.dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4, pin_memory=True)

    trainer.train(train_loader_adj, val_loader_adj, config['num_epochs'])

if __name__ == "__main__":
    main()
