import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import json
from datetime import datetime
import sys
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from .unet import get_model
from .losses import CombinedLoss
from utils.prepare_data import get_data_loaders
from utils.metrics import dice_score, iou_score

class Trainer:
    # ... The Trainer class is unchanged ...
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.model = get_model(
            n_channels=config.get('n_channels', 3), n_classes=config.get('n_classes', 1),
            base_filters=config.get('base_filters', 64), norm_type=config.get('norm_type', 'batch')
        ).to(self.device)
        if config.get('use_torch_compile', False) and hasattr(torch, 'compile'):
            self.model = torch.compile(self.model)
            print("Model compiled with torch.compile() for speedup.")
        self.criterion = CombinedLoss(
            base_loss_type=config.get('base_loss', 'focal'), focal_alpha=config.get('focal_loss_alpha', 0.25),
            focal_gamma=config.get('focal_loss_gamma', 2.0)
        )
        print(f"Using Combined Loss with base: {config.get('base_loss', 'focal')}")
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=config.get('learning_rate', 1e-4), weight_decay=config.get('weight_decay', 1e-4)
        )
        scheduler_type = config.get('scheduler_type', 'cosine')
        if scheduler_type == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', patience=config.get('scheduler_patience', 10), factor=config.get('scheduler_factor', 0.1))
            print("Using ReduceLROnPlateau scheduler.")
        elif scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config.get('num_epochs', 100), eta_min=1e-6)
            print("Using CosineAnnealingLR scheduler.")
        self.scaler = GradScaler(enabled=torch.cuda.is_available())
        self.writer = SummaryWriter(f"runs/{config['experiment_name']}")
        self.checkpoint_dir = f"checkpoints/{config['experiment_name']}"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        with open(os.path.join(self.writer.log_dir, 'config.json'), 'w') as f: json.dump(config, f, indent=2)
        self.best_dice = 0.0
        self.early_stopping_counter = 0
        self.history = {'train_loss': [], 'train_dice': [], 'train_iou': [], 'val_loss': [], 'val_dice': [], 'val_iou': []}

    def train_epoch(self, dataloader):
        self.model.train()
        running_loss, running_dice, running_iou = 0.0, 0.0, 0.0
        pbar = tqdm(dataloader, desc="Training Epoch")
        for images, masks, ignore_masks in pbar:
            images, masks, ignore_masks = images.to(self.device), masks.to(self.device), ignore_masks.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            with autocast():
                outputs = self.model(images)
                loss = self.criterion(outputs, masks, ignore_masks)
            self.scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            running_loss += loss.item()
            with torch.no_grad():
                valid_preds = (torch.sigmoid(outputs) > 0.5).view(-1)[ignore_masks.view(-1)]
                valid_targets = masks.view(-1)[ignore_masks.view(-1)]
                running_dice += dice_score(valid_preds, valid_targets).item()
                running_iou += iou_score(valid_preds, valid_targets).item()
            pbar.set_postfix(Loss=f'{running_loss/len(pbar):.4f}', Dice=f'{running_dice/len(pbar):.4f}')
        return running_loss/len(dataloader), running_dice/len(dataloader), running_iou/len(dataloader)

    def validate_epoch(self, dataloader):
        self.model.eval()
        running_loss, running_dice, running_iou = 0.0, 0.0, 0.0
        with torch.no_grad():
            pbar = tqdm(dataloader, desc="Validation Epoch")
            for images, masks, ignore_masks in pbar:
                images, masks, ignore_masks = images.to(self.device), masks.to(self.device), ignore_masks.to(self.device)
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks, ignore_masks)
                running_loss += loss.item()
                valid_preds = (torch.sigmoid(outputs) > 0.5).view(-1)[ignore_masks.view(-1)]
                valid_targets = masks.view(-1)[ignore_masks.view(-1)]
                running_dice += dice_score(valid_preds, valid_targets).item()
                running_iou += iou_score(valid_preds, valid_targets).item()
                pbar.set_postfix(Val_Loss=f'{running_loss/len(pbar):.4f}', Val_Dice=f'{running_dice/len(pbar):.4f}')
        return running_loss/len(dataloader), running_dice/len(dataloader), running_iou/len(dataloader)

    def save_plots(self):
        epochs = range(1, len(self.history['train_loss']) + 1)
        plt.figure(figsize=(20, 6)); plt.subplot(1, 3, 1)
        plt.plot(epochs, self.history['train_loss'], 'b-o', label='Train Loss'); plt.plot(epochs, self.history['val_loss'], 'r-o', label='Validation Loss')
        plt.title('Loss'); plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.legend(); plt.grid(True); plt.subplot(1, 3, 2)
        plt.plot(epochs, self.history['train_dice'], 'b-o', label='Train Dice'); plt.plot(epochs, self.history['val_dice'], 'r-o', label='Validation Dice')
        plt.title('Dice Score'); plt.xlabel('Epochs'); plt.ylabel('Dice'); plt.legend(); plt.grid(True); plt.subplot(1, 3, 3)
        plt.plot(epochs, self.history['train_iou'], 'b-o', label='Train IoU'); plt.plot(epochs, self.history['val_iou'], 'r-o', label='Validation IoU')
        plt.title('IoU Score'); plt.xlabel('Epochs'); plt.ylabel('IoU'); plt.legend(); plt.grid(True); plt.tight_layout()
        save_path = os.path.join(self.writer.log_dir, 'training_plots.png')
        plt.savefig(save_path); plt.close()
        print(f"Training plots saved to '{save_path}'")

    def train(self, train_loader, val_loader):
        num_epochs = self.config.get('num_epochs', 100); patience = self.config.get('early_stopping_patience', 15); save_freq = self.config.get('save_checkpoint_freq', 10)
        print(f"Starting training for {num_epochs} epochs... Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        tqdm_epochs = tqdm(range(1, num_epochs + 1), desc="Total Progress")
        for epoch in tqdm_epochs:
            train_loss, train_dice, train_iou = self.train_epoch(train_loader); val_loss, val_dice, val_iou = self.validate_epoch(val_loader)
            self.history['train_loss'].append(train_loss); self.history['val_loss'].append(val_loss); self.history['train_dice'].append(train_dice)
            self.history['val_dice'].append(val_dice); self.history['train_iou'].append(train_iou); self.history['val_iou'].append(val_iou)
            self.writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch); self.writer.add_scalars('Dice', {'train': train_dice, 'val': val_dice}, epoch)
            self.writer.add_scalars('IoU', {'train': train_iou, 'val': val_iou}, epoch); self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau): self.scheduler.step(val_dice)
            else: self.scheduler.step()
            print(f"\nEpoch {epoch}/{num_epochs} -> Train Loss: {train_loss:.4f}, Dice: {train_dice:.4f}, IoU: {train_iou:.4f} | Val Loss: {val_loss:.4f}, Dice: {val_dice:.4f}, IoU: {val_iou:.4f}")
            is_best = val_dice > self.best_dice
            if is_best:
                self.best_dice = val_dice; self.early_stopping_counter = 0
                checkpoint_data = {'epoch': epoch, 'model_state_dict': self.model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict(),
                                   'scheduler_state_dict': self.scheduler.state_dict(), 'dice_score': val_dice, 'config': self.config}
                torch.save(checkpoint_data, os.path.join(self.checkpoint_dir, 'best.pth')); tqdm_epochs.write(f"Epoch {epoch}: New best model saved with Val Dice: {val_dice:.4f}")
            else: self.early_stopping_counter += 1
            if epoch > 0 and epoch % save_freq == 0:
                torch.save(checkpoint_data, os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth'))
            if self.early_stopping_counter >= patience: print(f"\nEarly stopping triggered at epoch {epoch}."); break
        print(f"\nTraining completed. Best Validation Dice Score: {self.best_dice:.4f}"); self.writer.close(); self.save_plots()

def main():
    """Defines the configuration for a training run."""
    config = {
        'experiment_name': f'unet_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        'n_channels': 3, 'n_classes': 1, 'base_filters': 64, 'norm_type': 'batch',
        'learning_rate': 1e-4, 'weight_decay': 1e-4, 'num_epochs': 100,
        'batch_size': 16, 'num_workers': 8,
        'base_loss': 'focal',
        'focal_loss_alpha': 0.25, 'focal_loss_gamma': 2.0,
        'scheduler_type': 'cosine', 'scheduler_patience': 10, 'scheduler_factor': 0.1,
        'save_checkpoint_freq': 20, 'early_stopping_patience': 25,
        'use_torch_compile': True,
        
        ## ADDED: New hyperparameter to control augmentation
        'use_augmentation': False, # Set to False to disable all augmentations
    }
    
    print("Preparing data loaders...")
    train_loader, val_loader, _ = get_data_loaders(
        batch_size=config.get('batch_size', 16),
        num_workers=config.get('num_workers', 4),
        # Pass the new hyperparameter to the data loader function
        use_augmentation=config.get('use_augmentation', True)
    )

    trainer = Trainer(config)
    trainer.train(train_loader, val_loader)

if __name__ == "__main__":
    main()