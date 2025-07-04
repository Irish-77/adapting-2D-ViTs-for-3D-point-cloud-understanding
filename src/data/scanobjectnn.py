import os
import h5py
import torch
import numpy as np
from typing import Optional
from torch.utils.data import Dataset
from data.augment import normalize_point_cloud, rotate_point_cloud_y, random_scale_point_cloud, random_jitter_point_cloud


class ScanObjectNN(Dataset):
    """ScanObjectNN dataset for point cloud classification.
    
    To download the dataset, run:
    wget http://hkust-vgd.ust.hk/scanobjectnn/h5_files.zip
    
    Attributes:
        root_dir (str): Root directory containing the dataset.
        split (str): Dataset split ('training' or 'test').
        variant (str): Dataset variant.
        augmentation (str): Augmentation technique used.
        background (bool): Whether to include background.
        num_points (int): Number of points per point cloud.
        normalize (bool): Whether to normalize point clouds.
        use_newsplit (bool): Whether to use the newsplit variant.
        use_custom_augmentation (bool): Whether to use custom augmentation techniques.
        data (np.ndarray): Point cloud data.
        labels (np.ndarray): Class labels.
        num_classes (int): Number of unique classes.
    """
    
    def __init__(
        self,
        root_dir: str, 
        split: str = 'training',
        variant: str = 'main_split', 
        augmentation: str = 'base', 
        background:str = True, 
        num_points: Optional[int] = None, 
        normalize: bool = False, 
        use_newsplit: bool = False, 
        use_custom_augmentation: bool = False
    ) -> None:
        """Initialize ScanObjectNN dataset.
        
        Args:
            root_dir (str): Root directory containing the dataset.
            split (str): Dataset split, either 'training' or 'test'.
            variant (str): Dataset variant, one of ['main_split', 'split1', 'split2', 'split3', 'split4'].
            augmentation (str): Augmentation variant, one of ['base', 'augmented25_norot', 
                                'augmented25rot', 'augmentedrot', 'augmentedrot_scale75'].
            background (bool): If False, uses the _nobg variant without background.
            num_points (int or None): Number of points to sample for each point cloud.
                                     If None, returns all points without sampling.
            normalize (bool): Whether to normalize point clouds.
            use_newsplit (bool): If True and using augmentedrot_scale75, uses the newsplit variant.
            use_custom_augmentation (bool): Whether to use custom augmentation techniques.
        """
        self.root_dir = root_dir
        self.split = split
        self.variant = variant
        self.augmentation = augmentation
        self.background = background
        self.num_points = num_points
        self.normalize = normalize
        self.use_newsplit = use_newsplit
        self.use_custom_augmentation = use_custom_augmentation
        
        # Load data
        self.data, self.labels = self._load_data()
        self.num_classes = len(np.unique(self.labels))
        
    def _load_data(self):
        """Load ScanObjectNN data based on configuration.
        
        Constructs the appropriate file path based on the dataset configuration
        and loads the point cloud data and labels from the H5 file.
        
        Returns:
            tuple: (point_clouds, labels) where point_clouds is a numpy array of 
                  shape (N, P, 3) containing N point clouds with P points each,
                  and labels is a numpy array of shape (N,) containing class labels.
                  
        Raises:
            FileNotFoundError: If the dataset file cannot be found.
        """
        # Construct directory path
        dir_name = self.variant
        if not self.background:
            dir_name += '_nobg'
            
        # Construct filename
        if self.augmentation == 'base':
            filename = f'{self.split}_objectdataset.h5'
        else:
            aug_suffix = self.augmentation
            # Handle special case for newsplit
            if self.augmentation == 'augmentedrot_scale75' and self.use_newsplit and self.split == 'test':
                filename = f'{self.split}_objectdataset_{aug_suffix}_newsplit.h5'
            else:
                filename = f'{self.split}_objectdataset_{aug_suffix}.h5'
        
        h5_path = os.path.join(self.root_dir, dir_name, filename)
        
        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"Dataset file not found: {h5_path}")
            
        with h5py.File(h5_path, 'r') as f:
            data = f['data'][:]
            labels = f['label'][:]
            
        return data.astype(np.float32), labels.astype(np.int64).squeeze()
        
    def __len__(self):
        """Get the number of samples in the dataset.
        
        Returns:
            int: Number of point clouds in the dataset.
        """
        return len(self.labels)
        
    def __getitem__(self, idx):
        """Get a point cloud and its label by index.
        
        Samples or pads the point cloud to the specified number of points,
        normalizes if required, and applies augmentations during training.
        If num_points is None, returns all points without sampling.
        
        Args:
            idx (int): Index of the point cloud to retrieve.
            
        Returns:
            tuple: (point_cloud, label) where point_cloud is a tensor of shape (num_points, 3)
                  and label is a tensor containing the class label.
        """
        if self.num_points is None:
            # Return all points when num_points is None
            points = self.data[idx]
        else:
            # Sample or pad to the specified number of points
            points = self.data[idx][:self.num_points]
            
            # Handle variable point numbers
            if points.shape[0] < self.num_points:
                indices = np.random.choice(points.shape[0], self.num_points, replace=True)
                points = points[indices]
            elif points.shape[0] > self.num_points:
                indices = np.random.choice(points.shape[0], self.num_points, replace=False)
                points = points[indices]
                
        label = self.labels[idx]
        
        # Normalize
        if self.normalize:
            points = normalize_point_cloud(points)
                
        # Additional data augmentation for training (on top of pre-computed augmentations)
        if self.split == 'training':
            if self.use_custom_augmentation:
                # Apply custom augmentation techniques
                # Random rotation around y-axis with higher probability
                if np.random.random() > 0.2:  # 80% chance to apply rotation
                    points = rotate_point_cloud_y(points)
                
                # More aggressive scaling
                points = random_scale_point_cloud(points, scale_low=0.8, scale_high=1.2)
                
                # More aggressive jittering
                points = random_jitter_point_cloud(points, sigma=0.02, clip=0.04)
            else:
                # Only apply runtime augmentations if not using pre-augmented data with rotations
                if 'rot' not in self.augmentation:
                    # Random rotation around y-axis
                    points = rotate_point_cloud_y(points)
                
                # Random scaling
                points = random_scale_point_cloud(points, scale_low=0.95, scale_high=1.05)
                
                # Random jitter
                points = random_jitter_point_cloud(points, sigma=0.01, clip=0.02)
            
        return torch.FloatTensor(points), torch.LongTensor([label]).squeeze()