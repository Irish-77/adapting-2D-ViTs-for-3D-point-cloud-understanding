import os
import csv
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR

from tqdm import tqdm
from typing import  Tuple, Optional
from data import ScanObjectNN
from datetime import datetime
from torch.utils.data import DataLoader
from models import AdaptPointFormer, AdaptPointFormerWithSampling
from train.train_utils import save_configs


class APFTrainer:
    """Trainer class for AdaptPointFormer model on ScanObjectNN dataset"""
    
    def __init__(
        self,
        model_config: dict,
        dataset_config: dict,
        train_config: dict,
        device: str = "cpu",
        output_dir: str = "./output",
    ) -> None:
        """Initialize the trainer with configs for model, dataset, and training.
        
        Args:
            model_config: Configuration for the AdaptPointFormer model.
            dataset_config: Configuration for the ScanObjectNN dataset.
            train_config: Configuration for training parameters.
            device: Device to use for training ('cuda' or 'cpu').
            output_dir: Directory to save checkpoints and logs.
        """
        self.model_config = model_config
        self.dataset_config = dataset_config
        self.train_config = train_config
        self.output_dir = output_dir
        
        # Set device
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Path for metrics CSV file
        self.metrics_csv_path = os.path.join(output_dir, "training_metrics.csv")
        
        # Save all configurations to a file
        save_configs(model_config, dataset_config, train_config, output_dir, self.device)
        
        # Initialize the model, datasets, and loaders
        self._init_model()
        self._init_datasets()
        self._init_loaders()
        self._init_optimizer()
        self._init_metrics_csv()
        
    def _init_model(self) -> None:
        """Initialize the AdaptPointFormer model.
        
        Creates the appropriate model based on model_config and moves it to the 
        specified device.
        
        TODO: The first option should not be used, maybe revisit later, and remove it.
        """
        if self.model_config['model_name'] == 'AdaptPointFormer':
            self.model = AdaptPointFormer(
                num_classes=self.model_config['num_classes'],
                num_points=self.model_config['num_points'],
                vit_name=self.model_config['vit_name'],
                adapter_dim=self.model_config['adapter_dim'],
                pretrained=self.model_config['pretrained'],
                dropout_rate=self.model_config['dropout_rate']
            )
        else:
            self.model = AdaptPointFormerWithSampling(
                num_classes=self.model_config['num_classes'],
                num_points=self.model_config['num_points'],
                vit_name=self.model_config['vit_name'],
                adapter_dim=self.model_config['adapter_dim'],
                pretrained=self.model_config['pretrained'],
                dropout_rate=self.model_config['dropout_rate'],
                fps_sampling_func=self.model_config['fps_sampling_func'],
                n_samples=self.model_config['n_samples'],
                k_neighbors=self.model_config['k_neighbors']
            )
        
        self.model.to(self.device)
        self.model.print_trainable_params()
        
    def _init_datasets(self) -> None:
        """Initialize training and testing datasets.
        
        Creates ScanObjectNN dataset instances for both training and testing
        based on the dataset_config.
        """
        print("Loading training dataset...")
        self.train_dataset = ScanObjectNN(
            root_dir=self.dataset_config['root_dir'],
            split='training',
            variant=self.dataset_config['variant'],
            augmentation=self.dataset_config['augmentation'],
            num_points=self.dataset_config['num_points'],
            normalize=self.dataset_config['normalize'],
            sampling_method=self.dataset_config.get('sampling_method', 'all'),
            use_custom_augmentation=self.dataset_config.get('use_custom_augmentation', False)
        )
        
        print("Loading test dataset...")
        self.test_dataset = ScanObjectNN(
            root_dir=self.dataset_config['root_dir'],
            split='test',
            variant=self.dataset_config['variant'],
            augmentation=self.dataset_config['augmentation'],
            num_points=self.dataset_config['num_points'],
            normalize=self.dataset_config['normalize'],
            sampling_method=self.dataset_config.get('sampling_method', 'all'),
            use_custom_augmentation=False
        )
        
        # As model config and dataset config are initialized separately,
        # this makes sure we not forget to update one of them.
        # Unfortunately, it happend to me multiple times, so I added this check xD.
        assert self.train_dataset.num_classes == self.model_config['num_classes'], \
            "Number of classes in model config must match dataset classes"
        
        print(f"Dataset loaded: {len(self.train_dataset)} train samples, {len(self.test_dataset)} test samples")
        print(f"Number of classes: {self.train_dataset.num_classes}")
        
    def _init_loaders(self) -> None:
        """Initialize data loaders.
        
        Creates DataLoader instances for both training and testing datasets.
        """
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.train_config['batch_size'],
            shuffle=True,
            num_workers=4,
            drop_last=True
        )
        
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.train_config['batch_size'],
            shuffle=False,
            num_workers=4
        )
        
    def _init_optimizer(self) -> None:
        """Initialize optimizer, learning rate scheduler, and criterion."""
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            self.model.get_trainable_params(),
            lr=self.train_config['learning_rate'],
            weight_decay=self.train_config['weight_decay']
        )
        
        # Initialize learning rate scheduler if enabled
        self.scheduler = None
        if self.train_config.get('use_lr_scheduler', False):
            total_epochs = self.train_config['epochs']
            warmup_epochs = self.train_config.get('warmup_epochs', 0)
            warmup_factor = self.train_config.get('warmup_factor', 0.1)
            min_factor = self.train_config.get('min_factor', 1e-6)
            
            # Create warmup + cosine annealing scheduler
            if warmup_epochs > 0:
                # Define warmup function
                def lr_lambda(epoch):
                    if epoch < warmup_epochs:
                        # Linear warmup
                        return warmup_factor + (1.0 - warmup_factor) * epoch / warmup_epochs
                    else:
                        # Cosine annealing
                        return min_factor + 0.5 * (1.0 + np.cos(np.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)))
                
                self.scheduler = LambdaLR(self.optimizer, lr_lambda)
            else:
                # cosine annealing without warmup
                self.scheduler = CosineAnnealingLR(self.optimizer, T_max=total_epochs, eta_min=min_factor)
        
    def _init_metrics_csv(self) -> None:
        """Initialize the CSV file for tracking metrics."""
        headers = ['epoch', 'train_loss', 'train_accuracy', 'test_loss', 'test_accuracy']
        
        # Add learning rate to headers if scheduler is used
        if self.train_config.get('use_lr_scheduler', False):
            headers.append('learning_rate')
        
        with open(self.metrics_csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)
    
    def _update_metrics_csv(self, epoch: int, train_loss: float, 
                           train_acc: float, test_loss: float, test_acc: float) -> None:
        """Append metrics for the current epoch to the CSV file."""
        row = [epoch + 1, train_loss, train_acc, test_loss, test_acc]
        
        # Add current learning rate to metrics if scheduler is used
        if self.train_config.get('use_lr_scheduler', False):
            current_lr = self.optimizer.param_groups[0]['lr']
            row.append(current_lr)
        
        with open(self.metrics_csv_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(row)
    
    def train(self) -> None:
        """Main training loop."""
        best_acc = 0.0
        
        for epoch in range(self.train_config['epochs']):
            print(f"\nEpoch {epoch+1}/{self.train_config['epochs']}")
            
            # Print current learning rate
            if self.scheduler:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"Current learning rate: {current_lr:.6f}")
            
            # Training phase
            train_loss, train_acc = self._train_epoch()
            
            # Testing phase
            test_loss, test_acc = self._test_epoch()
            
            # Update metrics CSV
            self._update_metrics_csv(epoch, train_loss, train_acc, test_loss, test_acc)
            
            # Save model if it has better accuracy
            if test_acc > best_acc:
                best_acc = test_acc
                self._save_checkpoint("model_best.pt", epoch, test_acc)
                print(f"New best model saved with accuracy: {best_acc:.4f}")
                
            # Save model periodically
            if (epoch + 1) % self.train_config['save_interval'] == 0:
                self._save_checkpoint(f"model_epoch_{epoch+1}.pt", epoch, test_acc)
            
            # Step the learning rate scheduler
            if self.scheduler:
                self.scheduler.step()
                
            # Print epoch results
            print(f"Epoch {epoch+1} results:")
            print(f"  Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
            print(f"  Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
            
        print(f"\nTraining completed. Best test accuracy: {best_acc:.4f}")
        print(f"Training metrics saved to {self.metrics_csv_path}")
            
    def _train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Get gradient accumulation settings
        use_grad_accumulation = self.train_config.get('use_grad_accumulation', False)
        grad_accumulation_steps = self.train_config.get('grad_accumulation_steps', 1)
        
        pbar = tqdm(self.train_loader, desc="Training")
        for i, (points, labels) in enumerate(pbar):
            points = points.to(self.device)
            labels = labels.to(self.device)
            
            # Zero the gradients only at the beginning of accumulation cycle
            if not use_grad_accumulation or (i % grad_accumulation_steps == 0):
                self.optimizer.zero_grad()
            
            # Forward pass
            logits = self.model(points)
            loss = self.criterion(logits, labels)
            
            # Scale loss for gradient accumulation to maintain correct gradient magnitude
            if use_grad_accumulation:
                loss = loss / grad_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Optimize only after accumulating enough gradients or at the end of the epoch
            if not use_grad_accumulation or ((i + 1) % grad_accumulation_steps == 0) or (i == len(self.train_loader) - 1):
                # Apply gradient clipping if enabled - moved here to clip after accumulation
                if self.train_config.get('use_grad_clip', False):
                    max_norm = self.train_config.get('grad_clip_norm', 1.0)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
                
                self.optimizer.step()
            
            # For reporting, use the unscaled loss
            if use_grad_accumulation:
                unscaled_loss = loss.item() * grad_accumulation_steps
                total_loss += unscaled_loss
            else:
                total_loss += loss.item()
            
            # Statistics
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar with unscaled loss for clarity
            display_loss = loss.item() * grad_accumulation_steps if use_grad_accumulation else loss.item()
            pbar.set_postfix({
                'loss': f"{display_loss:.4f}",
                'acc': f"{100.0 * correct / total:.2f}%"
            })
            
        epoch_loss = total_loss / len(self.train_loader)
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
            
    def _test_epoch(self) -> Tuple[float, float]:
        """Evaluate on the test set.
        
        Returns:
            tuple: A tuple containing:
                - float: Average loss for the test set
                - float: Accuracy for the test set
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc="Testing")
            for points, labels in pbar:
                points = points.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                logits = self.model(points)
                loss = self.criterion(logits, labels)
                
                # Statistics
                total_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{100.0 * correct / total:.2f}%"
                })
                
        epoch_loss = total_loss / len(self.test_loader)
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def _save_checkpoint(self, filename: str, epoch: int, accuracy: float) -> None:
        """Save model checkpoint.
        
        Args:
            filename: Name of the checkpoint file.
            epoch: Current epoch number.
            accuracy: Validation accuracy achieved.
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'accuracy': accuracy,
            'model_config': self.model_config,
            'dataset_config': self.dataset_config,
        }
        
        torch.save(checkpoint, os.path.join(self.output_dir, filename))
        
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file.
            
        TODO: After implementing this, I never used it again, so revisit later and remove it if not needed.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Update configs if needed
        self.model_config = checkpoint.get('model_config', self.model_config)
        self.dataset_config = checkpoint.get('dataset_config', self.dataset_config)
        
        # Reinitialize model with loaded config
        self._init_model()
        
        # Load state dictionaries
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']} with accuracy {checkpoint['accuracy']:.4f}")