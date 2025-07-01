import numpy as np

def normalize_point_cloud(points):
    """Normalize point cloud to be centered at origin and scaled to unit sphere.
    
    Args:
        points (numpy.ndarray): Point cloud of shape (N, 3)
        
    Returns:
        numpy.ndarray: Normalized point cloud
    """
    centroid = np.mean(points, axis=0)
    points = points - centroid
    max_dist = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
    if max_dist > 0:
        points = points / max_dist
    return points

def random_point_dropout(points, max_dropout_ratio=0.875):
    """Randomly drop out points.
    
    Args:
        points (numpy.ndarray): Point cloud of shape (N, 3)
        max_dropout_ratio (float): Maximum dropout ratio
        
    Returns:
        numpy.ndarray: Point cloud with random points dropped
    """
    dropout_ratio = np.random.random() * max_dropout_ratio
    drop_idx = np.where(np.random.random((points.shape[0])) <= dropout_ratio)[0]
    if len(drop_idx) > 0:
        points[drop_idx, :] = points[0, :]  # set to the first point
    return points

def random_scale_point_cloud(points, scale_low=0.8, scale_high=1.25):
    """Randomly scale the point cloud.
    
    Args:
        points (numpy.ndarray): Point cloud of shape (N, 3)
        scale_low (float): Minimum scale factor
        scale_high (float): Maximum scale factor
        
    Returns:
        numpy.ndarray: Scaled point cloud
    """
    scale = np.random.uniform(scale_low, scale_high)
    return points * scale

def random_shift_point_cloud(points, shift_range=0.1):
    """Randomly shift point cloud.
    
    Args:
        points (numpy.ndarray): Point cloud of shape (N, 3)
        shift_range (float): Range of shift
        
    Returns:
        numpy.ndarray: Shifted point cloud
    """
    shifts = np.random.uniform(-shift_range, shift_range, 3)
    return points + shifts

def random_jitter_point_cloud(points, sigma=0.01, clip=0.05):
    """Randomly jitter points.
    
    Args:
        points (numpy.ndarray): Point cloud of shape (N, 3)
        sigma (float): Standard deviation of jitter
        clip (float): Clip jitter values to this range
        
    Returns:
        numpy.ndarray: Jittered point cloud
    """
    jitter = np.clip(sigma * np.random.randn(*points.shape), -clip, clip)
    return points + jitter

def rotate_point_cloud_y(points):
    """Randomly rotate the point cloud around Y axis.
    
    Args:
        points (numpy.ndarray): Point cloud of shape (N, 3)
        
    Returns:
        numpy.ndarray: Rotated point cloud
    """
    angle = np.random.uniform(0, 2 * np.pi)
    cos_angle, sin_angle = np.cos(angle), np.sin(angle)
    rotation_matrix = np.array([
        [cos_angle, 0, sin_angle],
        [0, 1, 0],
        [-sin_angle, 0, cos_angle]
    ])
    return points @ rotation_matrix

def rotate_point_cloud_z(points):
    """Randomly rotate the point cloud around Z axis.
    
    Args:
        points (numpy.ndarray): Point cloud of shape (N, 3)
        
    Returns:
        numpy.ndarray: Rotated point cloud
    """
    angle = np.random.uniform(0, 2 * np.pi)
    cos_angle, sin_angle = np.cos(angle), np.sin(angle)
    rotation_matrix = np.array([
        [cos_angle, -sin_angle, 0],
        [sin_angle, cos_angle, 0],
        [0, 0, 1]
    ])
    return points @ rotation_matrix