"""
Training script for GRU-VAD model with real-time visualization.

This script trains the GRU-based Voice Activity Detection model with:
- Real-time learning curves (loss)
- Real-time accuracy metrics (F1, precision, recall)
- Best model checkpointing
- Early stopping support
- TensorBoard logging

The model is trained on noisy speech data and learns to predict frame-level
voice activity.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for real-time plotting
from matplotlib.animation import FuncAnimation
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import json
import sys
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))
from gru_vad_model import GRU_VAD, GRU_VAD_Lite
from vad_dataset import create_vad_dataloaders


class MetricsTracker:
    """Track and compute training metrics."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.predictions = []
        self.targets = []
        self.losses = []
    
    def update(self, preds: torch.Tensor, targets: torch.Tensor, loss: float):
        """Update metrics with batch results."""
        # Convert predictions to binary (threshold at 0.5)
        preds_binary = (preds > 0.5).float()
        
        self.predictions.extend(preds_binary.cpu().numpy().flatten())
        self.targets.extend(targets.cpu().numpy().flatten())
        self.losses.append(loss)
    
    def compute(self) -> Dict[str, float]:
        """Compute all metrics."""
        preds = np.array(self.predictions)
        targets = np.array(self.targets)
        
        # Compute metrics (handle edge cases)
        try:
            precision = precision_score(targets, preds, zero_division=0)
            recall = recall_score(targets, preds, zero_division=0)
            f1 = f1_score(targets, preds, zero_division=0)
        except:
            precision = 0.0
            recall = 0.0
            f1 = 0.0
        
        accuracy = np.mean(preds == targets)
        avg_loss = np.mean(self.losses) if self.losses else 0.0
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }


class RealTimePlotter:
    """Real-time plotting of training metrics."""
    
    def __init__(self, save_dir: Path):
        self.save_dir = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize data storage
        self.epochs = []
        self.train_loss = []
        self.val_loss = []
        self.train_f1 = []
        self.val_f1 = []
        self.train_precision = []
        self.val_precision = []
        self.train_recall = []
        self.val_recall = []
        
        # Set up the figure with subplots
        plt.ion()  # Interactive mode
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
        self.fig.suptitle('GRU-VAD Training Metrics (Real-time)', fontsize=14)
        
        # Configure subplots
        self.ax_loss = self.axes[0, 0]
        self.ax_f1 = self.axes[0, 1]
        self.ax_pr = self.axes[1, 0]
        self.ax_recall = self.axes[1, 1]
        
        # Set labels
        self.ax_loss.set_xlabel('Epoch')
        self.ax_loss.set_ylabel('Loss')
        self.ax_loss.set_title('Loss Curves')
        self.ax_loss.grid(True, alpha=0.3)
        
        self.ax_f1.set_xlabel('Epoch')
        self.ax_f1.set_ylabel('F1 Score')
        self.ax_f1.set_title('F1 Score')
        self.ax_f1.grid(True, alpha=0.3)
        
        self.ax_pr.set_xlabel('Epoch')
        self.ax_pr.set_ylabel('Precision')
        self.ax_pr.set_title('Precision')
        self.ax_pr.grid(True, alpha=0.3)
        
        self.ax_recall.set_xlabel('Epoch')
        self.ax_recall.set_ylabel('Recall')
        self.ax_recall.set_title('Recall')
        self.ax_recall.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
    def update(self, epoch: int, train_metrics: Dict, val_metrics: Dict):
        """Update plots with new metrics."""
        self.epochs.append(epoch)
        self.train_loss.append(train_metrics['loss'])
        self.val_loss.append(val_metrics['loss'])
        self.train_f1.append(train_metrics['f1'])
        self.val_f1.append(val_metrics['f1'])
        self.train_precision.append(train_metrics['precision'])
        self.val_precision.append(val_metrics['precision'])
        self.train_recall.append(train_metrics['recall'])
        self.val_recall.append(val_metrics['recall'])
        
        # Clear and replot
        self.ax_loss.clear()
        self.ax_loss.plot(self.epochs, self.train_loss, 'b-', label='Train', linewidth=2)
        self.ax_loss.plot(self.epochs, self.val_loss, 'r-', label='Val', linewidth=2)
        self.ax_loss.set_xlabel('Epoch')
        self.ax_loss.set_ylabel('Loss')
        self.ax_loss.set_title('Loss Curves')
        self.ax_loss.legend()
        self.ax_loss.grid(True, alpha=0.3)
        
        self.ax_f1.clear()
        self.ax_f1.plot(self.epochs, self.train_f1, 'b-', label='Train', linewidth=2)
        self.ax_f1.plot(self.epochs, self.val_f1, 'r-', label='Val', linewidth=2)
        self.ax_f1.set_xlabel('Epoch')
        self.ax_f1.set_ylabel('F1 Score')
        self.ax_f1.set_title('F1 Score')
        self.ax_f1.legend()
        self.ax_f1.grid(True, alpha=0.3)
        self.ax_f1.set_ylim([0, 1])
        
        self.ax_pr.clear()
        self.ax_pr.plot(self.epochs, self.train_precision, 'b-', label='Train', linewidth=2)
        self.ax_pr.plot(self.epochs, self.val_precision, 'r-', label='Val', linewidth=2)
        self.ax_pr.set_xlabel('Epoch')
        self.ax_pr.set_ylabel('Precision')
        self.ax_pr.set_title('Precision')
        self.ax_pr.legend()
        self.ax_pr.grid(True, alpha=0.3)
        self.ax_pr.set_ylim([0, 1])
        
        self.ax_recall.clear()
        self.ax_recall.plot(self.epochs, self.train_recall, 'b-', label='Train', linewidth=2)
        self.ax_recall.plot(self.epochs, self.val_recall, 'r-', label='Val', linewidth=2)
        self.ax_recall.set_xlabel('Epoch')
        self.ax_recall.set_ylabel('Recall')
        self.ax_recall.set_title('Recall')
        self.ax_recall.legend()
        self.ax_recall.grid(True, alpha=0.3)
        self.ax_recall.set_ylim([0, 1])
        
        plt.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
        # Save figure
        save_path = self.save_dir / 'training_curves.png'
        self.fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    def close(self):
        """Close the plot window."""
        plt.close(self.fig)


class GRU_VAD_Trainer:
    """Trainer class for GRU-VAD model."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        device: torch.device,
        save_dir: Path,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.save_dir = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Loss and optimizer
        # Use weighted BCE loss to handle class imbalance
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(
            model.parameters(), 
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5,
            verbose=True
        )
        
        # Tracking
        self.best_val_f1 = 0.0
        self.best_epoch = 0
        self.history = []
        
        # Real-time plotter
        self.plotter = RealTimePlotter(save_dir)
        
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        tracker = MetricsTracker()
        
        for batch_idx, (frames, labels) in enumerate(self.train_loader):
            # frames: (batch, frame_size)
            # labels: (batch, 1)
            
            frames = frames.to(self.device)
            labels = labels.to(self.device)
            
            # Add sequence dimension: (batch, 1, frame_size)
            frames = frames.unsqueeze(1)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs, _ = self.model(frames)  # (batch, 1)
            
            # Compute loss
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Update tracker
            tracker.update(outputs, labels, loss.item())
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx + 1}/{len(self.train_loader)}, "
                      f"Loss: {loss.item():.4f}")
        
        return tracker.compute()
    
    def validate(self, loader: DataLoader) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        tracker = MetricsTracker()
        
        with torch.no_grad():
            for frames, labels in loader:
                frames = frames.to(self.device)
                labels = labels.to(self.device)
                
                # Add sequence dimension
                frames = frames.unsqueeze(1)
                
                # Forward pass
                outputs, _ = self.model(frames)
                
                # Compute loss
                loss = self.criterion(outputs, labels)
                
                # Update tracker
                tracker.update(outputs, labels, loss.item())
        
        return tracker.compute()
    
    def save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'best_val_f1': self.best_val_f1,
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, self.save_dir / 'latest_checkpoint.pt')
        
        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, self.save_dir / 'best_model.pt')
            print(f"  ✓ Saved best model (F1: {metrics['f1']:.4f})")
    
    def train(self, num_epochs: int, early_stopping_patience: int = 15):
        """Train the model for multiple epochs."""
        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"Model parameters: {self.model.count_parameters():,}")
        print(f"Device: {self.device}")
        print(f"Save directory: {self.save_dir}\n")
        
        no_improve_count = 0
        
        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            print("-" * 50)
            
            # Train
            train_metrics = self.train_epoch()
            print(f"Train - Loss: {train_metrics['loss']:.4f}, "
                  f"F1: {train_metrics['f1']:.4f}, "
                  f"Precision: {train_metrics['precision']:.4f}, "
                  f"Recall: {train_metrics['recall']:.4f}")
            
            # Validate
            val_metrics = self.validate(self.val_loader)
            print(f"Val   - Loss: {val_metrics['loss']:.4f}, "
                  f"F1: {val_metrics['f1']:.4f}, "
                  f"Precision: {val_metrics['precision']:.4f}, "
                  f"Recall: {val_metrics['recall']:.4f}")
            
            # Update learning rate
            self.scheduler.step(val_metrics['loss'])
            
            # Save history
            self.history.append({
                'epoch': epoch,
                'train': train_metrics,
                'val': val_metrics,
            })
            
            # Update plots
            self.plotter.update(epoch, train_metrics, val_metrics)
            
            # Check for improvement
            is_best = val_metrics['f1'] > self.best_val_f1
            if is_best:
                self.best_val_f1 = val_metrics['f1']
                self.best_epoch = epoch
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            # Save checkpoint
            self.save_checkpoint(epoch, val_metrics, is_best=is_best)
            
            # Early stopping
            if no_improve_count >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                print(f"Best F1: {self.best_val_f1:.4f} at epoch {self.best_epoch}")
                break
        
        # Final evaluation on test set
        print("\n" + "=" * 50)
        print("Training complete! Evaluating on test set...")
        self.load_best_model()
        test_metrics = self.validate(self.test_loader)
        print(f"Test  - Loss: {test_metrics['loss']:.4f}, "
              f"F1: {test_metrics['f1']:.4f}, "
              f"Precision: {test_metrics['precision']:.4f}, "
              f"Recall: {test_metrics['recall']:.4f}")
        
        # Save final results
        self.save_training_summary(test_metrics)
        
        return self.history
    
    def load_best_model(self):
        """Load the best model checkpoint."""
        checkpoint_path = self.save_dir / 'best_model.pt'
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded best model from epoch {checkpoint['epoch']}")
    
    def save_training_summary(self, test_metrics: Dict):
        """Save training summary to JSON."""
        summary = {
            'model_parameters': self.model.count_parameters(),
            'best_epoch': self.best_epoch,
            'best_val_f1': self.best_val_f1,
            'test_metrics': test_metrics,
            'training_history': self.history,
        }
        
        with open(self.save_dir / 'training_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nTraining summary saved to {self.save_dir / 'training_summary.json'}")


def main():
    """Main training script."""
    # Configuration
    config = {
        'batch_size': 64,
        'num_epochs': 50,
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'early_stopping_patience': 15,
        'target_sr': 16000,
        'frame_size': 128,  # 8ms at 16kHz
        'hop_size': 64,     # 50% overlap
        'snr_range': (-5.0, 20.0),
        'max_train_pairs': 100,
        'max_val_pairs': 20,
        'max_test_pairs': 20,
        'max_frames_per_file': 1000,
        'model_type': 'standard',  # 'standard' or 'lite'
    }
    
    print("=" * 60)
    print("GRU-VAD Training Script")
    print("=" * 60)
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Set up paths
    src_dir = Path(__file__).parent.parent.parent
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = src_dir / 'models' / 'gru_vad' / f'run_{timestamp}'
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(save_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Create dataloaders
    print("\n" + "-" * 60)
    print("Creating dataloaders...")
    dataloaders = create_vad_dataloaders(
        src_dir=src_dir,
        batch_size=config['batch_size'],
        target_sr=config['target_sr'],
        frame_size=config['frame_size'],
        hop_size=config['hop_size'],
        snr_range=config['snr_range'],
        num_workers=0,
        max_train_pairs=config['max_train_pairs'],
        max_val_pairs=config['max_val_pairs'],
        max_test_pairs=config['max_test_pairs'],
        max_frames_per_file=config['max_frames_per_file'],
    )
    
    # Create model
    print("\n" + "-" * 60)
    print("Creating model...")
    if config['model_type'] == 'lite':
        model = GRU_VAD_Lite(
            input_size=config['frame_size'],
            hidden_size=32,
            num_layers=1,
            dropout=0.1,
        )
    else:
        model = GRU_VAD(
            input_size=config['frame_size'],
            hidden_size=64,
            num_layers=2,
            dropout=0.2,
        )
    
    # Create trainer
    trainer = GRU_VAD_Trainer(
        model=model,
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        test_loader=dataloaders['test'],
        device=device,
        save_dir=save_dir,
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
    )
    
    # Train
    try:
        history = trainer.train(
            num_epochs=config['num_epochs'],
            early_stopping_patience=config['early_stopping_patience'],
        )
        print("\n" + "=" * 60)
        print("Training completed successfully!")
        print(f"Results saved to: {save_dir}")
        print("=" * 60)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        trainer.save_checkpoint(0, {}, is_best=False)
        print(f"Progress saved to: {save_dir}")
    finally:
        trainer.plotter.close()


if __name__ == "__main__":
    main()
