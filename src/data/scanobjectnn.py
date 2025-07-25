import os
import h5py
import torch
import numpy as np
from data.sampler import fps
from data.augment import (
    normalize_point_cloud,
    random_scale_point_cloud, 
    random_jitter_point_cloud,
    drop_and_replace_with_noise,
    random_rotate_point_cloud
)
from torch.utils.data import Dataset
from typing import Optional, Callable


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
        use_custom_augmentation: bool = False,
        augmentation_probability: float = 0.2,
        sampling_method: str = 'all',
        transform: Optional[Callable] = None,
        use_height: bool = False
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
            augmentation_probability (float): Probability of applying custom augmentations.
            sampling_method (str): Method for sampling points ('all', 'first', 'random', 'fps'). 
                                 If 'random', randomly samples points.
            transform (Optional[Callable]): Optional transformations to apply to the point clouds.
            use_height (bool): If True, appends height information to the point clouds.
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
        self.sampling_method = sampling_method
        self.augmentation_probability = augmentation_probability
        self.transform = transform
        self.use_height = use_height
        
        # Load data
        self.data, self.labels = self._load_data()
        self.num_classes = len(np.unique(self.labels))

        if self.num_points is not None and self.sampling_method == 'fps':
            print("Applying FPS sampling to point clouds...")
            # self.data = fps(self.data, self.num_points)
            points = torch.from_numpy(self.data).float().cuda()
            self.data = fps(points, self.num_points).cpu().numpy()
            print(f"Sampled {self.num_points} points per point cloud.")

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
        
        Args:
            idx (int): Index of the point cloud to retrieve.
            
        Returns:
            tuple: (point_cloud, label) where point_cloud is a tensor of shape (num_points, 3)
                  or (fps_samples, knn, 3) if using FPS sampling
                  and label is a tensor containing the class label.
        """
        points = self.data[idx]
        label = self.labels[idx]
        
        # Apply different sampling methods
        if self.sampling_method == 'all' or self.num_points is None:
            # Return all points
            pass
        elif self.sampling_method == 'first':
            # Sample or pad to the specified number of points
            if points.shape[0] < self.num_points:
                indices = np.random.choice(points.shape[0], self.num_points, replace=True)
                points = points[indices]
            elif points.shape[0] > self.num_points:
                points = points[:self.num_points]
        elif self.sampling_method == 'random':
            # Randomly sample points
            if points.shape[0] < self.num_points:
                # If too few points, sample with replacement
                indices = np.random.choice(points.shape[0], self.num_points, replace=True)
            else:
                # If enough points, sample without replacement
                indices = np.random.choice(points.shape[0], self.num_points, replace=False)
            points = points[indices]
        
        # Normalize if needed
        if self.normalize:
            points = normalize_point_cloud(points)
                
        # Additional data augmentation for training
        if self.split == 'training':
            if self.use_custom_augmentation:
                # Apply custom augmentation techniques
                # if np.random.random() > self.augmentation_probability:
                #     points = rotate_point_cloud_y(points)
                # if np.random.random() > self.augmentation_probability:
                #     points = rotate_point_cloud_z(points)
                if np.random.random() > self.augmentation_probability:
                    points = random_rotate_point_cloud(points)
                if np.random.random() > self.augmentation_probability:
                    points = random_scale_point_cloud(points, scale_low=0.8, scale_high=1.2)
                if np.random.random() > self.augmentation_probability:
                    points = random_jitter_point_cloud(points, sigma=0.03, clip=0.05)
                if np.random.random() > self.augmentation_probability:
                    points = drop_and_replace_with_noise(points, drop_ratio=0.2, noise_std=0.05)
            
        heights = None
        if self.transform:
            data = {'xyz': points, 'label': label}

            for transform_func in self.transform:
                data = transform_func(data)

            label = data['label']
            points = data['xyz']
            heights = data['heights']
         
        if self.use_height and heights is not None:
            points = torch.cat(
                [
                    torch.from_numpy(points).float(), # (Num points, 3)
                    torch.from_numpy(heights).float()
                ],
                dim=1
            )
        else:
            points = torch.from_numpy(points).float()

        return points, torch.LongTensor([label]).squeeze()