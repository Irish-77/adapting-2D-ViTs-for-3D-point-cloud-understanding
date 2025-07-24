import os
import csv
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.figure import Figure

from tqdm import tqdm
from typing import Tuple
from data import ScanObjectNN
from torch.utils.data import DataLoader
from models import PointCloudRendererClassifier
from train.train_utils import save_configs

class RendererTrainer:
    """Trainer class for PointCloudRendererClassifier model on ScanObjectNN dataset"""
    
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
            model_config: Configuration for the PointCloudRendererClassifier model.
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
        
        # Create directory for rendered views
        self.views_dir = os.path.join(output_dir, "rendered_views")
        os.makedirs(self.views_dir, exist_ok=True)
        
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
        
        # Current epoch tracker for warmup logic
        self.current_epoch = 0
        
    def _init_model(self) -> None:
        """Initialize the PointCloudRendererClassifier model.
        
        Creates the model based on model_config and moves it to the
        specified device.
        """
        self.model = PointCloudRendererClassifier(
            num_classes=self.model_config['num_classes'],
            vit_name=self.model_config['vit_name'],
            adapter_dim=self.model_config['adapter_dim'],
            num_views=self.model_config['num_views'],
            img_size=self.model_config['img_size'],
            pretrained=self.model_config['pretrained'],
            dropout_rate=self.model_config['dropout_rate'],
            diff_renderer=self.model_config.get('diff_renderer', False),
            view_transform_hidden=self.model_config.get('view_transform_hidden', 256),
        )
        
        self.model.to(self.device)
        trainable, total = self.model.count_parameters()
        print(f"Model initialized: {trainable:,} trainable parameters out of {total:,} total")
        
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
        
        # Verify model and dataset configurations are compatible
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
        """Initialize optimizer and criterion.
        """
        self.criterion = nn.CrossEntropyLoss()
        
        # For differentiable renderer, set up separate optimizers
        if self.model_config.get('diff_renderer', False):
            # Get parameters for the view transformation network
            view_transform_params = []
            other_trainable_params = []
            
            for name, param in self.model.named_parameters():
                if 'view_transform_net' in name and param.requires_grad:
                    view_transform_params.append(param)
                elif param.requires_grad:
                    other_trainable_params.append(param)
            
            # Create two separate optimizers
            self.optimizer = optim.AdamW(
                other_trainable_params,
                lr=self.train_config['learning_rate'],
                weight_decay=self.train_config['weight_decay']
            )
            
            # Set up view transform optimizer with separate learning rate
            self.view_optimizer = optim.AdamW(
                view_transform_params,
                lr=self.train_config.get('view_learning_rate', self.train_config['learning_rate'] * 0.1),
                weight_decay=self.train_config.get('view_weight_decay', self.train_config['weight_decay'])
            )
            
            # Set up schedulers
            if self.train_config.get('use_lr_scheduler', False):
                self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=self.train_config['epochs'],
                    eta_min=self.train_config.get('min_lr', 1e-6)
                )
                
                # Separate scheduler for view network - using ReduceLROnPlateau for better adaptivity
                self.view_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    self.view_optimizer, 
                    mode='max', 
                    factor=0.5, 
                    patience=5,
                    min_lr=self.train_config.get('view_min_lr', 1e-7)
                )
            else:
                self.scheduler = None
                self.view_scheduler = None
        else:
            # Standard optimizer for non-diff renderer
            self.optimizer = optim.AdamW(
                self.model.get_trainable_params(),
                lr=self.train_config['learning_rate'],
                weight_decay=self.train_config['weight_decay']
            )
            
            if self.train_config.get('use_lr_scheduler', False):
                self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=self.train_config['epochs'],
                    eta_min=self.train_config.get('min_lr', 1e-6)
                )
            else:
                self.scheduler = None
            
            # These don't exist for non-diff renderer model
            self.view_optimizer = None
            self.view_scheduler = None
        
    def _init_metrics_csv(self) -> None:
        """Initialize the CSV file for tracking metrics.
        
        Creates a new CSV file with headers for recording training metrics.
        """
        headers = ['epoch', 'train_loss', 'train_accuracy', 'test_loss', 'test_accuracy']
        
        with open(self.metrics_csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)
    
    def _update_metrics_csv(self, epoch: int, train_loss: float, 
                           train_acc: float, test_loss: float, test_acc: float) -> None:
        """Append metrics for the current epoch to the CSV file.
        
        Args:
            epoch: Current training epoch.
            train_loss: Average training loss for the epoch.
            train_acc: Training accuracy for the epoch.
            test_loss: Average test loss for the epoch.
            test_acc: Test accuracy for the epoch.
        """
        with open(self.metrics_csv_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([epoch + 1, train_loss, train_acc, test_loss, test_acc])
    
    def _save_rendered_views(self, points: torch.Tensor, epoch: int) -> None:
        """Save rendered views of a point cloud as images.
        
        Args:
            points: Point cloud tensor of shape (B, N, 3)
            epoch: Current epoch number for filename
        """
        # Use only the first point cloud in the batch
        point_cloud = points[0].unsqueeze(0)  # Shape (1, N, 3)
        
        # Get rendered views
        self.model.eval()  # Set to eval mode
        with torch.no_grad():
            rendered_views = self.model._get_rendered_views(point_cloud)  # Shape (1, num_views, 3, H, W)
        
        # Convert tensor to numpy array and denormalize if needed
        views = rendered_views[0].cpu().numpy()  # Shape (num_views, 3, H, W)
        
        # Create a grid of images
        num_views = views.shape[0]
        rows = int(np.ceil(num_views / 3))
        cols = min(num_views, 3)
        
        fig = plt.figure(figsize=(cols * 4, rows * 4))
        
        for i in range(num_views):
            # Get the view (C, H, W) and convert to (H, W, C)
            img = np.transpose(views[i], (1, 2, 0))
            
            # Normalize to [0, 1] if needed
            img = np.clip(img, 0, 1)
            
            # Add subplot
            ax = fig.add_subplot(rows, cols, i + 1)
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(f'View {i}')
        
        plt.tight_layout()
        
        # Save figure
        filename = os.path.join(self.views_dir, f"test_views_epoch_{epoch}.png")
        plt.savefig(filename)
        plt.close(fig)
        
        print(f"Test rendered views saved to {filename}")
    
    def train(self) -> None:
        """Main training loop.
        
        Executes the full training process including evaluation and model checkpointing.
        """
        best_acc = 0.0
        
        for epoch in range(self.train_config['epochs']):
            # Store current epoch for view saving logic and warmup
            self.current_epoch = epoch
            
            print(f"\nEpoch {epoch+1}/{self.train_config['epochs']}")
            
            # Training phase
            train_loss, train_acc = self._train_epoch()
            
            # Testing phase
            test_loss, test_acc = self._test_epoch()
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"Learning rate: {current_lr:.6f}")
                
            # Update view network scheduler if using diff renderer
            if hasattr(self, 'view_scheduler') and self.view_scheduler is not None:
                if isinstance(self.view_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.view_scheduler.step(test_acc)  # Use accuracy as metric
                else:
                    self.view_scheduler.step()
                    
                current_view_lr = self.view_optimizer.param_groups[0]['lr']
                print(f"View network learning rate: {current_view_lr:.6f}")
            
            # Update metrics CSV
            self._update_metrics_csv(epoch, train_loss, train_acc, test_loss, test_acc)
            
            # Save best model
            if test_acc > best_acc:
                best_acc = test_acc
                self._save_checkpoint("best_model.pt", epoch, test_acc)
                print(f"New best model with accuracy: {best_acc:.4f}")
                
            # Save model periodically
            if (epoch + 1) % self.train_config['save_interval'] == 0:
                self._save_checkpoint(f"model_epoch_{epoch+1}.pt", epoch, test_acc)
                
            # Print epoch results
            print(f"Epoch {epoch+1} results:")
            print(f"  Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
            print(f"  Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
            
        print(f"\nTraining completed. Best test accuracy: {best_acc:.4f}")
        print(f"Training metrics saved to {self.metrics_csv_path}")
        print(f"Rendered views saved to {self.views_dir}")
    
    def _train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch.
        
        Returns:
            tuple: A tuple containing:
                - float: Average loss for the epoch
                - float: Accuracy for the epoch
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Check if we're in warmup period
        warmup_epochs = self.train_config.get('view_warmup_epochs', 0)
        in_warmup = self.current_epoch < warmup_epochs
        
        # If using diff renderer and in warmup, freeze view network
        if self.model_config.get('diff_renderer', False) and in_warmup:
            # Freeze view network during warmup
            for name, param in self.model.named_parameters():
                if 'view_transform_net' in name:
                    param.requires_grad = False
            print(f"Epoch {self.current_epoch+1}: In warmup phase - view transformation network frozen")
        elif self.model_config.get('diff_renderer', False) and not in_warmup:
            # Unfreeze view network after warmup
            for name, param in self.model.named_parameters():
                if 'view_transform_net' in name:
                    param.requires_grad = True
            if self.current_epoch == warmup_epochs:
                print(f"Epoch {self.current_epoch+1}: Warmup complete - view transformation network unfrozen")
        
        pbar = tqdm(self.train_loader, desc="Training")
        for points, labels in pbar:
            points = points.to(self.device)
            labels = labels.to(self.device)
            
            # Zero the gradients for all optimizers
            self.optimizer.zero_grad()
            if self.model_config.get('diff_renderer', False) and not in_warmup:
                self.view_optimizer.zero_grad()
            
            # Forward pass
            logits = self.model(points)
            loss = self.criterion(logits, labels)
            
            # Backward pass and optimize
            loss.backward()
            
            # Optional gradient clipping
            if self.train_config.get('clip_grad_norm', 0) > 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.train_config['clip_grad_norm']
                )
                
            # Step optimizers
            self.optimizer.step()
            if self.model_config.get('diff_renderer', False) and not in_warmup:
                self.view_optimizer.step()
            
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
            for batch_idx, (points, labels) in enumerate(pbar):
                points = points.to(self.device)
                labels = labels.to(self.device)
                
                # Save rendered views for the first batch of test data
                if batch_idx == 0 and hasattr(self.model, '_get_rendered_views'):
                    # Check if we should save views this epoch
                    current_epoch = self.current_epoch if hasattr(self, 'current_epoch') else 0
                    if current_epoch % self.train_config.get('save_views_interval', 5) == 0:
                        self._save_rendered_views(points, current_epoch)
                
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
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save view optimizer state if using diff renderer
        if self.model_config.get('diff_renderer', False):
            checkpoint['view_optimizer_state_dict'] = self.view_optimizer.state_dict()
            if self.view_scheduler:
                checkpoint['view_scheduler_state_dict'] = self.view_scheduler.state_dict()
        
        torch.save(checkpoint, os.path.join(self.output_dir, filename))
        print(f"Model checkpoint saved to {os.path.join(self.output_dir, filename)}")
        
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file.
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
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        # Load view optimizer if using diff renderer
        if self.model_config.get('diff_renderer', False):
            if 'view_optimizer_state_dict' in checkpoint:
                self.view_optimizer.load_state_dict(checkpoint['view_optimizer_state_dict'])
            if self.view_scheduler and 'view_scheduler_state_dict' in checkpoint:
                self.view_scheduler.load_state_dict(checkpoint['view_scheduler_state_dict'])
        
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']} with accuracy {checkpoint['accuracy']:.4f}")
        
    def predict(self, points: torch.Tensor) -> torch.Tensor:
        """Make predictions for a batch of point clouds.
        
        Args:
            points: Tensor of shape (B, N, 3) containing 3D point coordinates.
            
        Returns:
            Tensor of shape (B,) containing predicted class indices.
        """
        self.model.eval()
        with torch.no_grad():
            points = points.to(self.device)
            logits = self.model(points)
            _, predictions = torch.max(logits, dim=1)
            return predictions
