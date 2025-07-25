import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from typing import Tuple
from data import ScanObjectNN
from data.augment import (
    scale_point_cloud,
    center_and_normalize_point_cloud,
    rotate_point_cloud,
)
from models import AdaptPointFormer
from torch.utils.data import DataLoader
from train.train_utils import save_configs
from timm.scheduler import CosineLRScheduler

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
        
        # Save configurations to a file
        save_configs(model_config, dataset_config, train_config, output_dir, self.device)

        # Path for metrics CSV file
        self.metrics_csv_path = os.path.join(output_dir, "training_metrics.csv")
        
        # Initialize the model, datasets, and loaders
        self._init_model()
        self._init_datasets()
        self._init_loaders()
        self._init_optimizer()
        self._init_metrics_csv()

    def _init_model(self) -> None:
        """Initialize the AdaptPointFormer model."""
        self.model = AdaptPointFormer(
            # General architecture
            num_classes = self.model_config['num_classes'],
            in_channels = self.model_config['in_channels'],
            vit_name = self.model_config['vit_name'],
            pretrained = self.model_config.get('pretrained', True),
            embedding_dim= self.model_config.get('embedding_dim', 768),
            # Data sampling
            npoint = self.model_config.get('npoint', 196),
            nsample = self.model_config.get('nsample', 32),
            # Optimization
            dropout_rate = self.model_config.get('dropout_rate', 0.1),
            dropout_path_rate = self.model_config.get('drop_path_rate', 0.1),
        )

        self.model.to(self.device)

    def _init_datasets(self) -> None:
        """Initialize training and testing datasets.
        
        Creates ScanObjectNN dataset instances for both training and testing
        based on the dataset_config.
        """


        train_transforms = [
            scale_point_cloud,
            center_and_normalize_point_cloud,
            rotate_point_cloud,
        ]
        test_transforms = [
            center_and_normalize_point_cloud,
        ]

        print("Loading training dataset...")
        self.train_dataset = ScanObjectNN(
            root_dir=self.dataset_config['root_dir'],
            # Split related
            split='training',
            variant=self.dataset_config['variant'],
            augmentation=self.dataset_config['augmentation'],
            background=self.dataset_config.get('background', False),
            use_newsplit=self.dataset_config.get('use_newsplit', False),
            # Data related
            num_points=self.dataset_config['train_num_points'],
            normalize=self.dataset_config.get('normalize', False),
            sampling_method=self.dataset_config.get('sampling_method', 'fps'),
            use_height= self.dataset_config.get('use_height', False),
            # Augmentation
            use_custom_augmentation=self.dataset_config.get('use_custom_augmentation', False),
            augmentation_probability=self.dataset_config.get('augmentation_probability', 0.0),
            transform=train_transforms
        )
        
        print("Loading test dataset...")
        self.test_dataset = ScanObjectNN(
            root_dir=self.dataset_config['root_dir'],
            # Split related
            split='test',
            variant=self.dataset_config['variant'],
            augmentation=self.dataset_config['augmentation'],
            background=self.dataset_config.get('background', False),
            use_newsplit=self.dataset_config.get('use_newsplit', False),
            # Data related
            num_points=self.dataset_config['test_num_points'],
            normalize=self.dataset_config.get('normalize', False),
            sampling_method=self.dataset_config.get('sampling_method', 'fps'),
            use_height= self.dataset_config.get('use_height', False),
            # Augmentation
            use_custom_augmentation=self.dataset_config.get('use_custom_augmentation', False),
            augmentation_probability=self.dataset_config.get('augmentation_probability', 0.0),
            transform=test_transforms
        )

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
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=self.train_config.get('label_smoothing', 0.3)
        )
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            betas=(0.9, 0.999),
            eps=1e-8,
            lr=self.train_config['learning_rate'],
            weight_decay=self.train_config['weight_decay']
        )
    
        self.scheduler = CosineLRScheduler(
            self.optimizer,
            t_initial=self.train_config['epochs'],
            warmup_t=self.train_config.get('warmup_epochs', 10),
            warmup_lr_init=self.train_config.get('warmup_lr_init', 1e-3),
            cycle_decay=0.05
        )
        
    def _init_metrics_csv(self) -> None:
        """Initialize the CSV file for tracking metrics."""
        headers = ['epoch', 'train_loss', 'train_accuracy', 'test_loss', 'test_accuracy', 'learning_rate']
        
        with open(self.metrics_csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)
    
    def _update_metrics_csv(self, epoch: int, train_loss: float, 
                           train_acc: float, test_loss: float, test_acc: float) -> None:
        """Append metrics for the current epoch to the CSV file."""
        row = [epoch + 1, train_loss, train_acc, test_loss, test_acc]
        row.append(
            self.optimizer.param_groups[0]['lr']
        )
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
                self.scheduler.step(epoch)
                
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
        
        pbar = tqdm(self.train_loader, desc="Training")
        for i, (points, labels) in enumerate(pbar):
            points = points.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            
            # Forward pass
            logits = self.model(points)
            loss = self.criterion(logits, labels)
            
            # Backward pass
            loss.backward()

            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Statistics
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            display_loss = loss.item()
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