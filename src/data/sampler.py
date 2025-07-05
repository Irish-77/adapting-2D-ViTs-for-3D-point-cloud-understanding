import torch
from typing import Tuple, Optional

def _farthest_point_sampling(points: torch.Tensor, n_samples: int) -> torch.Tensor:
    """
    Perform farthest point sampling (FPS) on a set of points.
    
    Args:
        points: Input point cloud tensor of shape (B, N, D) where B is the batch size,
               N is the number of points and D is the dimension of each point
        n_samples: Number of points to sample
        
    Returns:
        torch.Tensor: Indices of sampled points with shape (B, n_samples)
    """
    device = points.device
    batch_size, N, D = points.size()
    # Ensure we don't sample more points than available
    n_samples = min(n_samples, N)
    
    # Initialize output indices and distances
    centroids = torch.zeros(batch_size, n_samples, dtype=torch.long, device=device)
    distance = torch.ones(batch_size, N, device=device) * 1e10
    
    # Randomly select the first point
    farthest = torch.randint(0, N, (batch_size,), dtype=torch.long, device=device)
    
    batch_indices = torch.arange(batch_size, device=device)
    
    # Iteratively select farthest points
    for i in range(n_samples):
        # Set the farthest point as the current centroid
        centroids[:, i] = farthest
        
        # Get the coordinates of the current centroids
        centroid_points = points[batch_indices, farthest, :]
        centroid_points = centroid_points.view(batch_size, 1, D)
        
        # Compute distances from all points to this centroid
        dist = torch.sum((points - centroid_points) ** 2, dim=2)
        
        # Update minimum distances
        mask = dist < distance
        distance[mask] = dist[mask]
        
        # Select the point with maximum distance as the next centroid
        farthest = torch.max(distance, dim=1)[1]
    
    return centroids


def _get_knn_points(points: torch.Tensor, centroids: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Find k-nearest neighbors for each centroid point.
    
    Args:
        points: Input point cloud tensor of shape (B, N, D)
        centroids: Indices of centroid points with shape (B, n_samples)
        k: Number of neighbors to find for each centroid
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: 
            - Grouped points of shape (B, n_samples, k, D)
            - Indices of neighbors of shape (B, n_samples, k)
    """
    device = points.device
    batch_size, N, D = points.size()
    n_samples = centroids.size(1)
    
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * N
    
    # Reshape centroids for gathering: (B, n_samples) -> (B, n_samples, 1)
    centroids_expanded = centroids.unsqueeze(-1)
    
    # Get the coordinates of the centroid points: (B, n_samples, D)
    batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, n_samples)
    centroid_points = points[batch_indices, centroids, :]
    
    # Compute distances between all points and centroids
    # For each batch and centroid, compute distance to all points
    # Result shape: (B, n_samples, N)
    distances = torch.zeros(batch_size, n_samples, N, device=device)
    for b in range(batch_size):
        for i in range(n_samples):
            # Compute squared distances between this centroid and all points in the batch
            diff = points[b] - centroid_points[b, i].view(1, -1)
            distances[b, i] = torch.sum(diff ** 2, dim=1)
    
    # Find the k nearest points for each centroid
    _, knn_idx = torch.topk(distances, k=k, dim=2, largest=False, sorted=True)  # (B, n_samples, k)
    
    # Add base indices to get absolute indices in flattened array
    knn_idx = knn_idx + idx_base
    
    # Reshape for gathering
    knn_idx_flat = knn_idx.view(-1)
    
    # Gather the coordinates of the k nearest neighbors
    points_flat = points.reshape(-1, D)
    grouped_points = points_flat[knn_idx_flat].view(batch_size, n_samples, k, D)
    
    return grouped_points, knn_idx


def fps_sampling_with_knn(points: torch.Tensor, n_samples: int, k: int, return_centroids: bool = False) -> torch.Tensor:
    """
    Perform FPS sampling and then get k-nearest neighbors for each sampled point.
    
    Args:
        points: Input point cloud tensor of shape (B, N, D) or (N, D)
        n_samples: Number of centroid points to sample with FPS
        k: Number of nearest neighbors to gather for each centroid
        return_centroids: If True, also return the indices of the centroid points
        
    Returns:
        torch.Tensor: Either grouped points of shape (B, n_samples, k, D) or
                   if return_centroids is True, a tuple with (grouped_points, centroid_indices)
    """
    # Ensure input has batch dimension
    if len(points.shape) == 2:
        points = points.unsqueeze(0)  # Add batch dimension if not present
    
    # Perform FPS sampling
    centroid_indices = _farthest_point_sampling(points, n_samples)
    
    # Get k-nearest neighbors
    grouped_points, _ = _get_knn_points(points, centroid_indices, k)
    
    if return_centroids:
        return grouped_points, centroid_indices
    else:
        return grouped_points


def _farthest_point_sampling_optimized(points: torch.Tensor, n_samples: int) -> torch.Tensor:
    """
    Optimized farthest point sampling (FPS) using vectorized operations.
    
    Args:
        points: Input point cloud tensor of shape (B, N, D)
        n_samples: Number of points to sample
        
    Returns:
        torch.Tensor: Indices of sampled points with shape (B, n_samples)
    """
    device = points.device
    batch_size, N, D = points.size()
    n_samples = min(n_samples, N)
    
    # Initialize output indices and distances
    centroids = torch.zeros(batch_size, n_samples, dtype=torch.long, device=device)
    distance = torch.full((batch_size, N), 1e10, device=device)
    
    # Randomly select the first point
    farthest = torch.randint(0, N, (batch_size,), dtype=torch.long, device=device)
    batch_indices = torch.arange(batch_size, device=device)
    
    # Iteratively select farthest points
    for i in range(n_samples):
        centroids[:, i] = farthest
        
        # Get current centroid coordinates: (B, D)
        centroid_points = points[batch_indices, farthest, :]
        
        # Vectorized distance computation for all points
        # Broadcasting: (B, N, D) - (B, 1, D) -> (B, N, D)
        diff = points - centroid_points.unsqueeze(1)
        # Sum over dimension: (B, N, D) -> (B, N)
        dist = torch.sum(diff ** 2, dim=2)
        
        # Update minimum distances
        distance = torch.minimum(distance, dist)
        
        # Select the point with maximum distance as the next centroid
        farthest = torch.argmax(distance, dim=1)
    
    return centroids


def _get_knn_points_optimized(points: torch.Tensor, centroids: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Optimized k-nearest neighbors computation using vectorized operations.
    
    Args:
        points: Input point cloud tensor of shape (B, N, D)
        centroids: Indices of centroid points with shape (B, n_samples)
        k: Number of neighbors to find for each centroid
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: 
            - Grouped points of shape (B, n_samples, k, D)
            - Indices of neighbors of shape (B, n_samples, k)
    """
    device = points.device
    batch_size, N, D = points.size()
    n_samples = centroids.size(1)
    
    # Get centroid coordinates: (B, n_samples, D)
    batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, n_samples)
    centroid_points = points[batch_indices, centroids, :]
    
    # Vectorized distance computation
    # Expand dimensions for broadcasting: (B, n_samples, 1, D) and (B, 1, N, D)
    centroid_expanded = centroid_points.unsqueeze(2)  # (B, n_samples, 1, D)
    points_expanded = points.unsqueeze(1)  # (B, 1, N, D)
    
    # Compute all pairwise distances at once: (B, n_samples, N)
    distances = torch.sum((points_expanded - centroid_expanded) ** 2, dim=3)
    
    # Find k nearest neighbors for each centroid
    _, knn_idx = torch.topk(distances, k=k, dim=2, largest=False, sorted=True)
    
    # Gather the coordinates using advanced indexing
    # Create batch indices for gathering
    batch_idx = torch.arange(batch_size, device=device).view(-1, 1, 1).expand(-1, n_samples, k)
    
    # Gather points: (B, n_samples, k, D)
    grouped_points = points[batch_idx, knn_idx, :]
    
    return grouped_points, knn_idx


def fps_sampling_with_knn_optimized(points: torch.Tensor, n_samples: int, k: int, return_centroids: bool = False) -> torch.Tensor:
    """
    Optimized FPS sampling with k-nearest neighbors.
    
    Args:
        points: Input point cloud tensor of shape (B, N, D) or (N, D)
        n_samples: Number of centroid points to sample with FPS
        k: Number of nearest neighbors to gather for each centroid
        return_centroids: If True, also return the indices of the centroid points
        
    Returns:
        torch.Tensor: Either grouped points of shape (B, n_samples, k, D) or
                   if return_centroids is True, a tuple with (grouped_points, centroid_indices)
    """
    # Ensure input has batch dimension
    if len(points.shape) == 2:
        points = points.unsqueeze(0)
    
    # Perform optimized FPS sampling
    centroid_indices = _farthest_point_sampling_optimized(points, n_samples)
    
    # Get k-nearest neighbors
    grouped_points, _ = _get_knn_points_optimized(points, centroid_indices, k)
    
    if return_centroids:
        return grouped_points, centroid_indices
    else:
        return grouped_points


def _farthest_point_sampling_topk(points: torch.Tensor, n_samples: int) -> torch.Tensor:
    """
    FPS using torch.topk for finding farthest points. 
    More memory efficient for very large point clouds.
    
    Args:
        points: Input point cloud tensor of shape (B, N, D)
        n_samples: Number of points to sample
        
    Returns:
        torch.Tensor: Indices of sampled points with shape (B, n_samples)
    """
    device = points.device
    batch_size, N, D = points.size()
    n_samples = min(n_samples, N)
    
    centroids = torch.zeros(batch_size, n_samples, dtype=torch.long, device=device)
    distance = torch.full((batch_size, N), float('inf'), device=device)
    
    # Randomly select the first point
    farthest = torch.randint(0, N, (batch_size,), dtype=torch.long, device=device)
    batch_indices = torch.arange(batch_size, device=device)
    
    for i in range(n_samples):
        centroids[:, i] = farthest
        
        # Get current centroid coordinates
        centroid_points = points[batch_indices, farthest, :]
        
        # Compute distances using broadcasting
        diff = points - centroid_points.unsqueeze(1)
        dist = torch.sum(diff ** 2, dim=2)
        
        # Update minimum distances
        distance = torch.minimum(distance, dist)
        
        # Use topk to find the farthest point (k=1, largest=True)
        # This is more numerically stable than argmax
        _, farthest_indices = torch.topk(distance, k=1, dim=1, largest=True)
        farthest = farthest_indices.squeeze(1)
    
    return centroids


def _get_knn_points_topk(points: torch.Tensor, centroids: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    K-nearest neighbors using torch.topk with enhanced memory efficiency.
    
    Args:
        points: Input point cloud tensor of shape (B, N, D)
        centroids: Indices of centroid points with shape (B, n_samples)
        k: Number of neighbors to find for each centroid
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: 
            - Grouped points of shape (B, n_samples, k, D)
            - Indices of neighbors of shape (B, n_samples, k)
    """
    device = points.device
    batch_size, N, D = points.size()
    n_samples = centroids.size(1)
    k = min(k, N)  # Ensure k doesn't exceed available points
    
    # Get centroid coordinates
    batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, n_samples)
    centroid_points = points[batch_indices, centroids, :]
    
    # For very large point clouds, process in chunks to save memory
    if N > 10000:
        # Process centroids in chunks to avoid memory issues
        chunk_size = min(64, n_samples)  # Process up to 64 centroids at once
        all_knn_idx = []
        
        for start_idx in range(0, n_samples, chunk_size):
            end_idx = min(start_idx + chunk_size, n_samples)
            chunk_centroids = centroid_points[:, start_idx:end_idx, :]  # (B, chunk_size, D)
            
            # Compute distances for this chunk
            chunk_centroids_expanded = chunk_centroids.unsqueeze(2)  # (B, chunk_size, 1, D)
            points_expanded = points.unsqueeze(1)  # (B, 1, N, D)
            
            # Compute distances: (B, chunk_size, N)
            distances = torch.sum((points_expanded - chunk_centroids_expanded) ** 2, dim=3)
            
            # Use topk to find k nearest neighbors (smallest distances)
            _, chunk_knn_idx = torch.topk(distances, k=k, dim=2, largest=False, sorted=True)
            all_knn_idx.append(chunk_knn_idx)
        
        # Concatenate all chunks
        knn_idx = torch.cat(all_knn_idx, dim=1)
    else:
        # Standard processing for smaller point clouds
        centroid_expanded = centroid_points.unsqueeze(2)  # (B, n_samples, 1, D)
        points_expanded = points.unsqueeze(1)  # (B, 1, N, D)
        
        # Compute all pairwise distances: (B, n_samples, N)
        distances = torch.sum((points_expanded - centroid_expanded) ** 2, dim=3)
        
        # Use topk to find k nearest neighbors
        _, knn_idx = torch.topk(distances, k=k, dim=2, largest=False, sorted=True)
    
    # Gather the coordinates using advanced indexing
    batch_idx = torch.arange(batch_size, device=device).view(-1, 1, 1).expand(-1, n_samples, k)
    grouped_points = points[batch_idx, knn_idx, :]
    
    return grouped_points, knn_idx


def fps_sampling_with_knn_topk(points: torch.Tensor, n_samples: int, k: int, return_centroids: bool = False):
    """
    TopK-optimized FPS sampling with k-nearest neighbors.
    Best for very large point clouds and when numerical stability is important.
    
    Args:
        points: Input point cloud tensor of shape (B, N, D) or (N, D)
        n_samples: Number of centroid points to sample with FPS
        k: Number of nearest neighbors to gather for each centroid
        return_centroids: If True, also return the indices of the centroid points
        
    Returns:
        torch.Tensor: Either grouped points of shape (B, n_samples, k, D) or
                   if return_centroids is True, a tuple with (grouped_points, centroid_indices)
    """
    # Ensure input has batch dimension
    if len(points.shape) == 2:
        points = points.unsqueeze(0)
    
    # Perform TopK-optimized FPS sampling
    centroid_indices = _farthest_point_sampling_topk(points, n_samples)
    
    # Get k-nearest neighbors using TopK
    grouped_points, _ = _get_knn_points_topk(points, centroid_indices, k)
    
    if return_centroids:
        return grouped_points, centroid_indices
    else:
        return grouped_points


def _farthest_point_sampling_cuda_optimized(points: torch.Tensor, n_samples: int) -> torch.Tensor:
    """
    CUDA-optimized FPS using matrix operations and memory-efficient updates.
    Best for large point clouds on GPU.
    """
    device = points.device
    batch_size, N, D = points.size()
    n_samples = min(n_samples, N)
    
    centroids = torch.zeros(batch_size, n_samples, dtype=torch.long, device=device)
    distance = torch.full((batch_size, N), float('inf'), device=device)
    
    # Initialize with random points
    farthest = torch.randint(0, N, (batch_size,), dtype=torch.long, device=device)
    batch_indices = torch.arange(batch_size, device=device)
    
    for i in range(n_samples):
        centroids[:, i] = farthest
        
        # Get current centroids
        centroid_points = points[batch_indices, farthest, :]
        
        # Compute distances using efficient matrix operations
        # Use cdist for potentially better performance on large datasets
        if N > 1000:  # Use cdist for large point clouds
            # Reshape for cdist: (B*1, D) and (B*N, D)
            centroid_reshaped = centroid_points.unsqueeze(1)  # (B, 1, D)
            dist = torch.cdist(centroid_reshaped, points, p=2).squeeze(1) ** 2  # (B, N)
        else:
            # Use broadcasting for smaller point clouds
            diff = points - centroid_points.unsqueeze(1)
            dist = torch.sum(diff ** 2, dim=2)
        
        # In-place minimum update for memory efficiency
        torch.minimum(distance, dist, out=distance)
        
        # Find next farthest point
        farthest = torch.argmax(distance, dim=1)
    
    return centroids


def fps_sampling_with_knn_cuda_optimized(points: torch.Tensor, n_samples: int, k: int, return_centroids: bool = False):
    """
    CUDA-optimized version with memory-efficient operations.
    """
    if len(points.shape) == 2:
        points = points.unsqueeze(0)
    
    # Use CUDA-optimized FPS if on GPU and point cloud is large
    if points.is_cuda and points.size(1) > 1000:
        centroid_indices = _farthest_point_sampling_cuda_optimized(points, n_samples)
    else:
        centroid_indices = _farthest_point_sampling_optimized(points, n_samples)
    
    grouped_points, _ = _get_knn_points_optimized(points, centroid_indices, k)
    
    if return_centroids:
        return grouped_points, centroid_indices
    else:
        return grouped_points


def benchmark_fps_implementations(points: torch.Tensor, n_samples: int, k: int, num_runs: int = 10):
    """
    Benchmark different FPS implementations.
    """
    import time
    
    print(f"Benchmarking FPS on {points.shape} with {n_samples} samples, k={k}")
    print(f"Device: {points.device}")
    print("-" * 50)
    
    # Original implementation
    torch.cuda.synchronize() if points.is_cuda else None
    start_time = time.time()
    for _ in range(num_runs):
        _ = fps_sampling_with_knn(points, n_samples, k)
    torch.cuda.synchronize() if points.is_cuda else None
    original_time = (time.time() - start_time) / num_runs
    
    # Optimized implementation
    torch.cuda.synchronize() if points.is_cuda else None
    start_time = time.time()
    for _ in range(num_runs):
        _ = fps_sampling_with_knn_optimized(points, n_samples, k)
    torch.cuda.synchronize() if points.is_cuda else None
    optimized_time = (time.time() - start_time) / num_runs
    
    # TopK optimized implementation
    torch.cuda.synchronize() if points.is_cuda else None
    start_time = time.time()
    for _ in range(num_runs):
        _ = fps_sampling_with_knn_topk(points, n_samples, k)
    torch.cuda.synchronize() if points.is_cuda else None
    topk_time = (time.time() - start_time) / num_runs
    
    # CUDA optimized (if applicable)
    if points.is_cuda:
        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(num_runs):
            _ = fps_sampling_with_knn_cuda_optimized(points, n_samples, k)
        torch.cuda.synchronize()
        cuda_optimized_time = (time.time() - start_time) / num_runs
    else:
        cuda_optimized_time = None
    
    print(f"Original implementation: {original_time:.4f}s")
    print(f"Optimized implementation: {optimized_time:.4f}s")
    print(f"TopK implementation: {topk_time:.4f}s")
    print(f"Optimized speedup: {original_time/optimized_time:.2f}x")
    print(f"TopK speedup: {original_time/topk_time:.2f}x")
    
    if cuda_optimized_time:
        print(f"CUDA optimized: {cuda_optimized_time:.4f}s")