import numpy as np
from scipy.linalg import expm, norm

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

def drop_and_replace_with_noise(point_cloud, drop_ratio=0.05, noise_std=0.02):
    """Drop points from the point cloud and replace them with noisy points.
    
    This augmentation randomly selects points to drop,
    then replaces them with noisy points placed randomly within the 
    point cloud's bounding box, plus some Gaussian noise.
    
    Args:
        point_cloud (np.ndarray): Input point cloud, shape (N, 3).
        drop_ratio (float): Ratio of points to drop (between 0 and 1).
        noise_std (float): Standard deviation for the Gaussian noise.
        
    Returns:
        np.ndarray: Augmented point cloud with same shape as input.
    """
    num_points = point_cloud.shape[0]
    drop_count = int(num_points * drop_ratio)
    
    # Randomly select indices to drop
    drop_indices = np.random.choice(num_points, drop_count, replace=False)
    keep_indices = np.setdiff1d(np.arange(num_points), drop_indices)
    
    # Keep the selected points
    kept_points = point_cloud[keep_indices]
    
    # Create new noisy points within range of other points
    min_bounds = np.min(point_cloud, axis=0)
    max_bounds = np.max(point_cloud, axis=0)
    
    # Generate random points within the bounding box
    random_points = np.random.uniform(
        min_bounds, max_bounds, size=(drop_count, 3)
    )
    
    # Add Gaussian noise
    gaussian_noise = np.random.normal(0, noise_std, size=(drop_count, 3))
    random_points += gaussian_noise
    
    # Combine kept points with new noisy points
    augmented_point_cloud = np.zeros_like(point_cloud)
    augmented_point_cloud[keep_indices] = kept_points
    augmented_point_cloud[drop_indices] = random_points
    
    return augmented_point_cloud

def random_rotate_point_cloud(points):
    """Randomly rotate the point cloud around all axes.
    
    Args:
        points (numpy.ndarray): Point cloud of shape (N, 3)
        
    Returns:
        numpy.ndarray: Rotated point cloud
    """
    # Z-axis rotation (full)
    angle_z = np.random.uniform(0, 2 * np.pi)
    cos_z, sin_z = np.cos(angle_z), np.sin(angle_z)
    R_z = np.array([[cos_z, -sin_z, 0],
                    [sin_z, cos_z, 0],
                    [0, 0, 1]])

    # Y-axis rotation (limited, between +/- 15 degrees)
    angle_y = np.random.uniform(-np.pi / 12, np.pi / 12)
    cos_y, sin_y = np.cos(angle_y), np.sin(angle_y)
    R_y = np.array([[cos_y, 0, sin_y],
                    [0, 1, 0],
                    [-sin_y, 0, cos_y]])
    
    # X-axis rotation (limited, between +/- 15 degrees)
    angle_x = np.random.uniform(-np.pi / 12, np.pi / 12)
    cos_x, sin_x = np.cos(angle_x), np.sin(angle_x)
    R_x = np.array([[1, 0, 0],
                    [0, cos_x, -sin_x],
                    [0, sin_x, cos_x]])

    # Combine rotations (Z -> Y -> X)
    rotation_matrix = R_z @ R_y @ R_x
    
    return points @ rotation_matrix.T

def scale_point_cloud(
    data,
    scale_range=(0.9, 1.1),
    anisotropic=True, 
    scale_xyz=(True, True, True)
):
    """
    Scale point cloud with optional anisotropic scaling and mirroring.
    
    Args:
        data: dict containing 'xyz' key with point cloud
        scale_range: tuple of (min_scale, max_scale)
        anisotropic: if True, scale each axis independently
        scale_xyz: tuple of bools, whether to scale each axis
    
    Returns:
        dict containing 'xyz'
    """
    points = data['xyz'].copy()
    
    # Generate random scale
    if anisotropic:
        scale = np.random.uniform(scale_range[0], scale_range[1], size=3)
    else:
        scale_val = np.random.uniform(scale_range[0], scale_range[1])
        scale = np.array([scale_val, scale_val, scale_val])
    
    # Apply scale_xyz mask
    for i, should_scale in enumerate(scale_xyz):
        if not should_scale:
            scale[i] = 1.0
    
    data['xyz'] = points * scale
    return data

def center_and_normalize_point_cloud(
    data,
    center=True,
    normalize=True, 
    gravity_dim=1
):
    """
    Center and normalize point cloud.
    
    Args:
        data: dict containing 'xyz' key with point cloud
        center: whether to center the point cloud
        normalize: whether to normalize to unit sphere
        gravity_dim: dimension index for height (default 1)
    Returns:
       dict containing 'xyz' and heights
    """
    points = data['xyz'].copy()

    heights = points[:, gravity_dim:gravity_dim + 1]
    data['heights'] = heights - np.min(heights, axis=0)
    
    if center:
        centroid = np.mean(points, axis=0)
        points = points - centroid
    
    if normalize:
        distances = np.sqrt(np.sum(points ** 2, axis=1))
        max_dist = np.max(distances)
        if max_dist > 0:
            points = points / max_dist

    data['xyz'] = points
    return data

def rotate_point_cloud(data, angle=(0.0, 1.0, 0.0), angle_units='radians'):
    """
    Apply random rotation to point cloud.
    
    Args:
        data: dict containing 'xyz' key with point cloud
        angle: tuple of rotation bounds for each axis (x, y, z)
        angle_units: 'radians' or 'degrees'
    
    Returns:
        dict containing 'xyz'
    """
    points = data['xyz'].copy()
    
    # Convert to radians if needed
    if angle_units == 'degrees':
        angle = np.array(angle) * np.pi / 180
    else:
        angle = np.array(angle)
    
    # Generate rotation matrices for each axis
    rot_matrices = []
    for axis_idx, rot_bound in enumerate(angle):
        if rot_bound != 0:
            # Random angle within bounds
            theta = np.random.uniform(-rot_bound, rot_bound)
            
            # Create axis vector
            axis = np.zeros(3)
            axis[axis_idx] = 1
            
            # Create rotation matrix using Rodrigues' formula
            rot_mat = expm(np.cross(np.eye(3), axis / norm(axis) * theta))
            rot_matrices.append(rot_mat)
        else:
            rot_matrices.append(np.eye(3))
    
    # Combine rotations in random order
    indices = np.random.permutation(3)
    combined_rot = np.eye(3)
    for i in indices:
        combined_rot = combined_rot @ rot_matrices[i]
    
    data['xyz'] = points @ combined_rot.T

    return data