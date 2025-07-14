import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Tuple

from data import ScanObjectNN
from models import Pix4Point

class Pix4PointTrainer:
    """Trainer for the Pix4Point model on ScanObjectNN."""

    def __init__(
        self,
        model_config: dict,
        dataset_config: dict,
        train_config: dict,
        device: str = "cpu",
        output_dir: str = "./output_pix4point",
    ) -> None:
        self.model_config = model_config
        self.dataset_config = dataset_config
        self.train_config = train_config
        self.output_dir = output_dir

        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using Device: {self.device}")

        os.makedirs(output_dir, exist_ok=True)
        self.metrics_csv_path = os.path.join(output_dir, "training_metrics.csv")

        self._init_model()
        self._init_datasets()
        self._init_loaders()
        self._init_optimizer()
        self._init_metrics_csv()

    def _init_model(self) -> None:
        self.model = Pix4Point(
            num_classes=self.model_config['num_classes'],
            pretrained_model=self.model_config['pretrained_model'],
            k_neighbors=self.model_config['k_neighbors'],
            embed_dim=self.model_config['embed_dim']
        )
        self.model.to(self.device)
        self.model.print_trainable_params()

    def _init_datasets(self) -> None:
        print("Loading training dataset...")
        self.train_dataset = ScanObjectNN(
            root_dir=self.dataset_config['root_dir'],
            split='training',
            variant=self.dataset_config['variant'],
            augmentation=self.dataset_config['augmentation'],
            num_points=self.dataset_config['num_points'],
            normalize=self.dataset_config['normalize'],
            sampling_method=self.dataset_config.get('sampling_method', 'all'),
            use_custom_augmentation=self.dataset_config.get('use_custom_augmentation', False),
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
            use_custom_augmentation=False,
        )

        assert self.train_dataset.num_classes == self.model_config['num_classes'], \
            "Number of classes in model config must match dataset classes"
        print(f"Dataset loaded: {len(self.train_dataset)} train samples, {len(self.test_dataset)} test samples")

    def _init_loaders(self) -> None:
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.train_config['batch_size'],
            shuffle=True,
            num_workers=1,
            drop_last=True,
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.train_config['batch_size'],
            shuffle=False,
            num_workers=1,
        )

    def _init_optimizer(self) -> None:
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            self.model.get_trainable_params(),
            lr=self.train_config['learning_rate'],
            weight_decay=self.train_config['weight_decay'],
        )

    def _init_metrics_csv(self) -> None:
        headers = ['epoch', 'train_loss', 'train_accuracy', 'test_loss', 'test_accuracy']
        with open(self.metrics_csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)

    def _update_metrics_csv(self, epoch: int, train_loss: float, train_acc: float, test_loss: float, test_acc: float) -> None:
        with open(self.metrics_csv_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([epoch + 1, train_loss, train_acc, test_loss, test_acc])

    def train(self) -> None:
        best_acc = 0.0
        for epoch in range(self.train_config['epochs']):
            print(f"\nEpoch {epoch+1}/{self.train_config['epochs']}")
            train_loss, train_acc = self._train_epoch()
            test_loss, test_acc = self._test_epoch()
            self._update_metrics_csv(epoch, train_loss, train_acc, test_loss, test_acc)
            if test_acc > best_acc:
                best_acc = test_acc
                self._save_checkpoint("best_model.pt", epoch, best_acc)
            if (epoch + 1) % self.train_config['save_interval'] == 0:
                self._save_checkpoint(f"model_epoch_{epoch+1}.pt", epoch, test_acc)
            print(f"Epoch {epoch+1} results:")
            print(f"  Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
            print(f"  Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
        print(f"\nTraining completed. Best test accuracy: {best_acc:.4f}")
        print(f"Training metrics saved to {self.metrics_csv_path}")

    def _train_epoch(self) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        pbar = tqdm(self.train_loader, desc="Training")
        for points, labels in pbar:
            points = points.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(points)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{100.0 * correct / total:.2f}%"})
        epoch_loss = total_loss / len(self.train_loader)
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    def _test_epoch(self) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc="Testing")
            for points, labels in pbar:
                points = points.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(points)
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{100.0 * correct / total:.2f}%"})
        epoch_loss = total_loss / len(self.test_loader)
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    def _save_checkpoint(self, filename: str, epoch: int, accuracy: float) -> None:
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
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model_config = checkpoint.get('model_config', self.model_config)
        self.dataset_config = checkpoint.get('dataset_config', self.dataset_config)
        self._init_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']} with accuracy {checkpoint['accuracy']:.4f}")
