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
import sys

# Ensure the project root is in the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

# Model imports for the remaining models
from models.aer_unet import get_aer_unet_model
from models.segformer import get_segformer_model

from utils.data_utils import WaterBodiesDataset
from utils.metrics import dice_score, iou_score
from utils.losses import DiceLoss, CombinedLoss, FocalLoss, TverskyLoss, FocalLovaszLoss
from utils.prepare_data import get_data_loaders

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.writer = SummaryWriter(f"runs/{config['experiment_name']}")
        self.best_iou = 0.0
        self.patience_counter = 0
        self.history = []
        
        self.checkpoint_dir = f"checkpoints/{config['experiment_name']}"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.model = self._get_model(config).to(self.device)
        self.analyze_model_complexity()
        self.criterion = self._get_loss_function(config['loss_type'])
        self.optimizer = self._get_optimizer(config)
        self.scheduler = self._get_scheduler(config)
        
        self.use_amp = (config.get('use_amp', False) and torch.cuda.is_available())
        if self.use_amp:
            self.scaler = GradScaler()
            print("Using Automatic Mixed Precision (AMP)")
        
        with open(os.path.join(self.checkpoint_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)
    
    def _get_model(self, config):
        """Initialize model based on type"""
        model_type = config['model_type']
        
        if model_type == 'unet++-pretrained-encoder':
            print("Initializing U-Net++ with pre-trained efficientnet-b4 encoder.")
            return smp.UnetPlusPlus(
                encoder_name="efficientnet-b4",
                encoder_weights="imagenet",
                in_channels=config['n_channels'],
                classes=config['n_classes'],
            )
        elif model_type == 'segformer-b4':
            print("Initializing SegFormer-B4 model pre-trained on ADE20K.")
            return get_segformer_model(n_classes=config['n_classes'])
        elif model_type == 'aer-unet':
            print("Initializing AER U-Net.")
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
            return FocalLoss()
        elif loss_type == 'tversky':
            return TverskyLoss()
        elif loss_type == 'focal_lovasz':
            return FocalLovaszLoss(
                focal_weight=self.config.get('focal_weight', 0.5),
                lovasz_weight=self.config.get('lovasz_weight', 0.5)
            )
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def _get_optimizer(self, config):
        """Initialize optimizer based on configuration"""
        if config['optimizer'] == 'adam':
            return optim.Adam(self.model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        elif config['optimizer'] == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        elif config['optimizer'] == 'sgd':
            return optim.SGD(self.model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'], momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {config['optimizer']}")

    def _get_scheduler(self, config):
        """Initialize scheduler based on configuration"""
        if config['scheduler_type'] == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=config['scheduler_patience'], factor=config['scheduler_factor'])
        elif config['scheduler_type'] == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config['num_epochs'], eta_min=config['learning_rate'] * 0.01)
        elif config['scheduler_type'] == 'step':
            return optim.lr_scheduler.StepLR(self.optimizer, step_size=config.get('step_size', 30), gamma=config.get('gamma', 0.1))
        else:
            raise ValueError(f"Unknown scheduler type: {config['scheduler_type']}")

    def analyze_model_complexity(self):
        """Analyze and print model complexity"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        size_mb = sum(p.nelement() * p.element_size() for p in self.model.parameters()) / 1024**2
        
        print(f"\nModel Complexity Analysis:")
        print(f"  Model type: {self.config['model_type']}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Model size: {size_mb:.2f} MB")
        
        if self.config['model_type'] == 'aer-unet':
            print(f"  Base features: {self.config['base_features']}")
        if self.config['model_type'] == 'unet++-pretrained-encoder':
            print(f"  Deep Supervision: {self.config.get('deep_supervision', False)}")
        
        self.writer.add_text('Model/Type', self.config['model_type'])
        self.writer.add_text('Model/Parameters', f"Trainable: {trainable_params:,}")
        self.writer.add_text('Model/Size_MB', f"{size_mb:.2f}")

    def train_epoch(self, dataloader, epoch):
        self.model.train()
        total_loss, total_dice, total_iou = 0, 0, 0
        pbar = tqdm(dataloader, desc=f'Training Epoch {epoch}')
        for images, masks in pbar:
            images, masks = images.to(self.device), masks.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            
            # FIX: Use the modern torch.amp.autocast with device_type
            with autocast(enabled=self.use_amp):
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
            
            if self.use_amp:
                self.scaler.scale(loss).backward()
                if self.config.get('gradient_clipping', False):
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['max_grad_norm'])
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.config.get('gradient_clipping', False):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['max_grad_norm'])
                self.optimizer.step()

            with torch.no_grad():
                preds = torch.sigmoid(outputs) > 0.5
                dice = dice_score(preds, masks)
                iou = iou_score(preds, masks)
            
            total_loss += loss.item()
            total_dice += dice.item()
            total_iou += iou.item()
            pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'IoU': f'{iou.item():.4f}', 'LR': f'{self.optimizer.param_groups[0]["lr"]:.2e}'})
        
        return total_loss / len(dataloader), total_dice / len(dataloader), total_iou / len(dataloader)

    def validate_epoch(self, dataloader, epoch):
        self.model.eval()
        total_loss, total_dice, total_iou = 0, 0, 0
        with torch.no_grad():
            pbar = tqdm(dataloader, desc=f'Validation Epoch {epoch}')
            for images, masks in pbar:
                images, masks = images.to(self.device), masks.to(self.device)
                
                with autocast(enabled=self.use_amp):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
                
                preds = torch.sigmoid(outputs) > 0.5
                dice = dice_score(preds, masks)
                iou = iou_score(preds, masks)
                total_loss += loss.item()
                total_dice += dice.item()
                total_iou += iou.item()
                pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'IoU': f'{iou.item():.4f}'})
        
        # FIX: Ensure 3 values are returned
        return total_loss / len(dataloader), total_dice / len(dataloader), total_iou / len(dataloader)

    def save_checkpoint(self, epoch, val_iou, filename):
        """Saves a model checkpoint with a given filename."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'iou_score': val_iou,
            'config': self.config
        }
        if self.use_amp and self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, os.path.join(self.checkpoint_dir, filename))

    def _plot_and_save_history(self):
        history_df = pd.DataFrame(self.history)
        for key, value in self.config.items():
            if key not in history_df.columns: history_df[key] = value
        history_df.to_csv(os.path.join(self.checkpoint_dir, 'training_history.csv'), index=False)
        
        sns.set_theme(style="darkgrid")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        ax1.plot(history_df['epoch'], history_df['train_loss'], 'o-', label='Train Loss')
        ax1.plot(history_df['epoch'], history_df['val_loss'], 'o-', label='Validation Loss')
        ax1.set_ylabel('Loss'); ax1.set_title('Training & Validation Loss'); ax1.legend()
        ax2.plot(history_df['epoch'], history_df['train_iou'], 'o-', label='Train IoU')
        ax2.plot(history_df['epoch'], history_df['val_iou'], 'o-', label='Validation IoU')
        ax2.set_xlabel('Epoch'); ax2.set_ylabel('IoU Score'); ax2.set_title('Training & Validation IoU'); ax2.legend()
        plt.savefig(os.path.join(self.checkpoint_dir, 'training_plots.png')); plt.close()
        print(f"\nTraining history and plots saved to: {self.checkpoint_dir}")


    def train(self, train_loader, val_loader):
        print(f"Starting training for {self.config['num_epochs']} epochs...")
        
        for epoch in range(1, self.config['num_epochs'] + 1):
            train_loss, _, train_iou = self.train_epoch(train_loader, epoch)
            val_loss, _, val_iou = self.validate_epoch(val_loader, epoch)
            
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history.append({'epoch': epoch, 'train_loss': train_loss, 'train_iou': train_iou, 'val_loss': val_loss, 'val_iou': val_iou, 'lr': current_lr})
            self.writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch)
            self.writer.add_scalars('IoU', {'train': train_iou, 'val': val_iou}, epoch)
            self.writer.add_scalar('Learning_Rate', current_lr, epoch)
            
            self.scheduler.step(val_loss) if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau) else self.scheduler.step()
            
            print(f"Epoch {epoch}/{self.config['num_epochs']} -> Train IoU: {train_iou:.4f}, Val IoU: {val_iou:.4f}, LR: {current_lr:.2e}")
            
            # --- FIX: New, robust checkpointing logic ---
            is_best = val_iou > self.best_iou
            if is_best:
                self.best_iou = val_iou
                self.patience_counter = 0
                print(f"New best model found with IoU: {val_iou:.4f}. Saving...")
                self.save_checkpoint(epoch, val_iou, "best.pth")
            else:
                self.patience_counter += 1

            # Always save the latest checkpoint
            self.save_checkpoint(epoch, val_iou, "latest.pth")

            # Save periodic checkpoint if the feature is enabled
            save_interval = self.config.get('save_every_n_epochs', 0)
            if save_interval > 0 and epoch % save_interval == 0:
                print(f"Saving periodic checkpoint for epoch {epoch}...")
                self.save_checkpoint(epoch, val_iou, f"epoch_{epoch}.pth")
            
            if self.config.get('early_stopping', False) and self.patience_counter >= self.config['early_stopping_patience']:
                print(f"Early stopping triggered at epoch {epoch}. Best Val IoU: {self.best_iou:.4f}")
                break
        
        self.writer.close()
        self._plot_and_save_history()

def parse_args():
    parser = argparse.ArgumentParser(description='Streamlined Segmentation Training Script')
    
    parser.add_argument('--model', type=str, required=True,
                        choices=['aer-unet', 'unet++-pretrained-encoder', 'segformer-b4'],
                        help='Model architecture to use')
    parser.add_argument('--n_channels', type=int, default=3, help='Number of input channels')
    parser.add_argument('--n_classes', type=int, default=1, help='Number of output classes')
    parser.add_argument('--base_features', type=int, default=32, help='Base features for AER U-Net')
    parser.add_argument('--dropout_rate', type=float, default=0.3, help='Dropout rate for AER U-Net')
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
                        choices=['bce', 'dice', 'combined', 'focal', 'tversky', 'focal_lovasz'], 
                        default='focal_lovasz', 
                        help='Loss function type')
    parser.add_argument('--focal_weight', type=float, default=0.5, help='Weight for Focal Loss in FocalLovaszLoss')
    parser.add_argument('--lovasz_weight', type=float, default=0.5, help='Weight for Lovasz Loss in FocalLovaszLoss')
    parser.add_argument('--use_amp', action='store_true', help='Use automatic mixed precision')
    parser.add_argument('--early_stopping', action='store_true', help='Enable early stopping')
    parser.add_argument('--early_stopping_patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--gradient_clipping', action='store_true', help='Enable gradient clipping')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Max gradient norm')
    parser.add_argument('--experiment_name', type=str, default=None, help='Custom experiment name')
    parser.add_argument('--save_every_n_epochs', type=int, default=20, help='Save a checkpoint every N epochs. Set to 0 to disable.')


    return parser.parse_args()

def main():
    args = parse_args()
    config = vars(args)
    
    if config.get('experiment_name') is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config['experiment_name'] = f"{config['model']}_water_segmentation_{timestamp}"
    
    config['model_type'] = config.pop('model')

    print("Training Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    trainer = Trainer(config)
    
    train_loader, val_loader, _ = get_data_loaders(batch_size=config['batch_size'])

    # FIX: Call train() with the correct number of arguments
    trainer.train(train_loader, val_loader)

if __name__ == "__main__":
    main()