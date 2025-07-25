import torch
import torch.nn as nn
from typing import Tuple

class ViewTransformationNetwork(nn.Module):
    """
    Network to predict optimal view parameters for each input point cloud.
    
    This network takes a point cloud as input and predicts azimuth and elevation
    angles for multiple views that would be optimal for rendering the point cloud.
    """
    
    def __init__(self, num_views: int = 6, hidden_dim: int = 256) -> None:
        """
        Initialize the view transformation network.
        
        Args:
            num_views: Number of views to predict
            hidden_dim: Dimension of hidden features
        """
        super().__init__()
        self.num_views = num_views
        
        # Encoder for point cloud features
        self.point_encoder = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        # Global feature aggregation
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        
        # View parameter prediction
        self.view_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Separate heads for azimuth and elevation
        self.azimuth_head = nn.Linear(hidden_dim, num_views)
        self.elevation_head = nn.Linear(hidden_dim, num_views)
        
        # Initialize to reasonable defaults
        nn.init.zeros_(self.azimuth_head.weight)
        nn.init.zeros_(self.elevation_head.weight)
        # Initialize biases to spread views
        with torch.no_grad():
            default_azimuths = torch.linspace(0, 360, num_views + 1)[:-1]
            default_elevations = torch.tensor([0, 30, -30, 0, 0, 0])[:num_views]
            self.azimuth_head.bias.data = default_azimuths * torch.pi / 180
            self.elevation_head.bias.data = default_elevations * torch.pi / 180
    
    def forward(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict optimal view parameters for the input point cloud.
        
        Args:
            points: Point cloud tensor of shape (B, N, 3)
                - B: batch size
                - N: number of points
            
        Returns:
            Tuple containing:
                - azimuths: Tensor of shape (B, num_views) with azimuth angles in radians
                - elevations: Tensor of shape (B, num_views) with elevation angles in radians
        """
        # Encode point cloud
        points_t = points.transpose(1, 2)  # (B, 3, N)
        features = self.point_encoder(points_t)  # (B, hidden_dim, N)
        
        # Global pooling
        global_features = self.global_pool(features).squeeze(-1)  # (B, hidden_dim)
        
        # Predict view parameters
        view_features = self.view_predictor(global_features)
        
        # Predict angles (normalized to reasonable ranges)
        azimuths = self.azimuth_head(view_features)  # (B, num_views)
        elevations = self.elevation_head(view_features)  # (B, num_views)
        
        # Apply tanh and scale to reasonable ranges
        azimuths = torch.tanh(azimuths) * torch.pi  # [-π, π]
        elevations = torch.tanh(elevations) * torch.pi / 3  # [-π/3, π/3] (±60 degrees)
        
        return azimuths, elevations
    

class DifferentiablePointCloudRenderer(nn.Module):
    """
    Differentiable renderer that projects 3D point clouds to 2D images.
    
    Based on the MVTN paper's approach, this renderer uses 2D bilinear interpolation
    to scatter points onto an image grid.
        
    References and inspired by:
        - https://arxiv.org/pdf/2011.13244
        - https://github.com/ajhamdi/MVTN
    """
    def __init__(self, img_size: int = 224):
        """
        Initialize the differentiable point cloud renderer.
        
        Args:
            img_size: Size of the output image (img_size x img_size)
        """
        super().__init__()
        self.img_size = img_size

    def apply_rotation(self, points: torch.Tensor, azimuth: torch.Tensor, elevation: torch.Tensor) -> torch.Tensor:
        """
        Apply rotation matrices to point clouds based on azimuth and elevation angles.
        
        Args:
            points: Point cloud tensor of shape (B, N, 3)
            azimuth: Azimuth angles in radians of shape (B,)
            elevation: Elevation angles in radians of shape (B,)
            
        Returns:
            Rotated point cloud of shape (B, N, 3)
        """
        B = points.shape[0]
        device = points.device
        
        cos_az, sin_az = torch.cos(azimuth), torch.sin(azimuth)
        cos_el, sin_el = torch.cos(elevation), torch.sin(elevation)
        
        # Azimuth rotation (around y-axis)
        R_az = torch.zeros(B, 3, 3, device=device)
        R_az[:, 0, 0] = cos_az
        R_az[:, 0, 2] = sin_az
        R_az[:, 1, 1] = 1
        R_az[:, 2, 0] = -sin_az
        R_az[:, 2, 2] = cos_az
        
        # Elevation rotation (around x-axis)
        R_el = torch.zeros(B, 3, 3, device=device)
        R_el[:, 0, 0] = 1
        R_el[:, 1, 1] = cos_el
        R_el[:, 1, 2] = -sin_el
        R_el[:, 2, 1] = sin_el
        R_el[:, 2, 2] = cos_el
        
        R = torch.bmm(R_el, R_az)
        return torch.bmm(points, R.transpose(1, 2))

    def _render_bilinear(self, points_2d: torch.Tensor, point_features: torch.Tensor) -> torch.Tensor:
        """
        Render points using differentiable bilinear splatting.
        
        Args:
            points_2d: 2D projected points of shape (B, N, 2) in normalized coordinates [-1, 1]
            point_features: Features for each point of shape (B, N) or (B, N, C)
            
        Returns:
            Rendered image of shape (B, 3, H, W)
        """
        B, N, _ = points_2d.shape
        H, W = self.img_size, self.img_size
        device = points_2d.device

        # Scale normalized coords [-1, 1] to pixel space [0, W-1] / [0, H-1]
        # We subtract 0.5 to align with pixel centers
        px = (points_2d[..., 0] + 1) * 0.5 * W - 0.5
        py = (points_2d[..., 1] + 1) * 0.5 * H - 0.5

        # Get the int coords of 4 surrounding pixels
        px1, py1 = torch.floor(px), torch.floor(py)
        px2, py2 = px1 + 1, py1 + 1

        # Get bilinear interpolation weights for neighbors
        w11 = (px2 - px) * (py2 - py)
        w12 = (px2 - px) * (py - py1)
        w21 = (px - px1) * (py2 - py)
        w22 = (px - px1) * (py - py1)

        # Filter points that project outside the image boundaries
        mask = (px1 >= 0) & (py1 >= 0) & (px2 < W) & (py2 < H)

        # Prepare tensors for scatter operation
        weights = torch.stack([w11, w12, w21, w22], dim=2)[mask] # (n_valid_points, 4)
        
        # Expand features for each of 4 neighbors
        features = point_features[mask].unsqueeze(1).expand(-1, 4) # (n_valid_points, 4)
        
        # Weighted features to be scattered
        values = (features * weights).view(-1)

        # Get pixel coordinates for valid points
        px1_m, py1_m = px1[mask].long(), py1[mask].long()
        px2_m, py2_m = px2[mask].long(), py2[mask].long()

        # Get batch indices for valid points
        batch_idx = torch.arange(B, device=device).view(B, 1).expand(B, N)[mask]

        # Calculate flat 1D indices for scattering into a (B*H*W) tensor
        base_idx = batch_idx * H * W
        idx11 = base_idx + py1_m * W + px1_m
        idx12 = base_idx + py2_m * W + px1_m
        idx21 = base_idx + py1_m * W + px2_m
        idx22 = base_idx + py2_m * W + px2_m
        indices = torch.stack([idx11, idx12, idx21, idx22], dim=1).view(-1)

        # Scatter weighted features onto the image grid
        image = torch.zeros(B * H * W, device=device)
        image.scatter_add_(0, indices, values)
        
        # Reshape to image format and expand to 3 channels
        image = image.view(B, H, W).unsqueeze(1).expand(-1, 3, -1, -1)

        return image

    def forward(self, points: torch.Tensor, azimuth: torch.Tensor, elevation: torch.Tensor) -> torch.Tensor:
        """
        Render point clouds from given viewpoints.
        
        Args:
            points: Point cloud tensor of shape (B, N, 3)
            azimuth: Azimuth angle in radians of shape (B,)
            elevation: Elevation angle in radians of shape (B,)
            
        Returns:
            Rendered images of shape (B, 3, H, W) where H=W=img_size
        """
        # Rotate points
        rotated_points = self.apply_rotation(points, azimuth, elevation)

        # Project points and get depth
        points_2d = rotated_points[..., :2] # Use x, y for projection
        z_coords = rotated_points[..., 2]   # Use z for depth feature

        # Create feature for each point based on its depth
        z_min = z_coords.min(dim=1, keepdim=True)[0]
        z_max = z_coords.max(dim=1, keepdim=True)[0]
        z_normalized = (z_coords - z_min) / (z_max - z_min + 1e-6)
        point_features = 0.3 + 0.7 * z_normalized

        # Render the image using bilinear splatting
        return self._render_bilinear(points_2d, point_features)



# # Uncomment the following code if you want to use Gaussian splatting instead of bilinear interpolation
# # and if you have lots of time and aaaaaaaaa lot of memory ;(.
# class DifferentiablePointCloudRenderer(nn.Module):
#     """Fully differentiable point cloud renderer using Gaussian splatting"""
    
#     def __init__(self, img_size=224, sigma=1.0):
#         super().__init__()
#         self.img_size = img_size
#         self.sigma = sigma
        
#     def normalize_point_cloud(self, points):
#         """Center and scale point cloud"""
#         centroid = points.mean(dim=1, keepdim=True)
#         points = points - centroid
#         max_dist = torch.sqrt((points ** 2).sum(dim=-1)).max(dim=1, keepdim=True)[0].unsqueeze(-1)
#         points = points / (max_dist + 1e-6)
#         return points
    
#     def apply_rotation(self, points, azimuth, elevation):
#         """Apply rotation matrices in a differentiable way"""
#         B = points.shape[0]
#         device = points.device
        
#         # Ensure angles are properly shaped
#         if azimuth.dim() == 0:
#             azimuth = azimuth.unsqueeze(0).expand(B)
#         if elevation.dim() == 0:
#             elevation = elevation.unsqueeze(0).expand(B)
        
#         # Rotation matrices
#         cos_az = torch.cos(azimuth).view(B, 1, 1)
#         sin_az = torch.sin(azimuth).view(B, 1, 1)
#         cos_el = torch.cos(elevation).view(B, 1, 1)
#         sin_el = torch.sin(elevation).view(B, 1, 1)
        
#         # Azimuth rotation matrix (around y-axis)
#         R_az = torch.zeros(B, 3, 3, device=device)
#         R_az[:, 0, 0] = cos_az.squeeze()
#         R_az[:, 0, 2] = -sin_az.squeeze()
#         R_az[:, 1, 1] = 1
#         R_az[:, 2, 0] = sin_az.squeeze()
#         R_az[:, 2, 2] = cos_az.squeeze()
        
#         # Elevation rotation matrix (around x-axis)
#         R_el = torch.zeros(B, 3, 3, device=device)
#         R_el[:, 0, 0] = 1
#         R_el[:, 1, 1] = cos_el.squeeze()
#         R_el[:, 1, 2] = -sin_el.squeeze()
#         R_el[:, 2, 1] = sin_el.squeeze()
#         R_el[:, 2, 2] = cos_el.squeeze()
        
#         # Combined rotation
#         R = torch.bmm(R_el, R_az)
        
#         # Apply rotation
#         rotated_points = torch.bmm(points, R.transpose(1, 2))
#         return rotated_points
    
#     def render_gaussian(self, points_2d, z_coords, img_size):
#         """Differentiable Gaussian rendering"""
#         B, N, _ = points_2d.shape
#         device = points_2d.device
        
#         # Create coordinate grid
#         y_coords = torch.linspace(-1, 1, img_size, device=device)
#         x_coords = torch.linspace(-1, 1, img_size, device=device)
#         yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
#         grid = torch.stack([xx, yy], dim=-1).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W, 2)
        
#         # Expand dimensions for broadcasting
#         points_2d_exp = points_2d.unsqueeze(2).unsqueeze(2)  # (B, N, 1, 1, 2)
        
#         # Compute distances
#         distances = torch.sum((grid - points_2d_exp) ** 2, dim=-1)  # (B, N, H, W)
        
#         # Gaussian weights
#         weights = torch.exp(-distances / (2 * self.sigma ** 2))
        
#         # Normalize z-coordinates for intensity
#         z_min = z_coords.min(dim=1, keepdim=True)[0]
#         z_max = z_coords.max(dim=1, keepdim=True)[0]
#         z_normalized = (z_coords - z_min) / (z_max - z_min + 1e-6)
#         intensities = 0.3 + 0.7 * z_normalized  # (B, N)
        
#         # Apply intensities
#         weighted_intensities = weights * intensities.unsqueeze(-1).unsqueeze(-1)  # (B, N, H, W)
        
#         # Aggregate using softmax (differentiable approximation of max)
#         temperature = 0.1
#         attention = F.softmax(weighted_intensities / temperature, dim=1)
#         rendered = torch.sum(weighted_intensities * attention, dim=1)  # (B, H, W)
        
#         # Expand to 3 channels
#         rendered = rendered.unsqueeze(1).expand(-1, 3, -1, -1)  # (B, 3, H, W)
        
#         return rendered
    
#     def forward(self, points, azimuth, elevation):
#         """
#         Differentiable rendering of point cloud from given viewpoint
#         Args:
#             points: (B, N, 3) point cloud
#             azimuth: (B,) or scalar azimuth angle in radians
#             elevation: (B,) or scalar elevation angle in radians
#         Returns:
#             images: (B, 3, H, W) rendered images
#         """
#         # Normalize point cloud
#         points = self.normalize_point_cloud(points)
        
#         # Apply rotation
#         rotated_points = self.apply_rotation(points, azimuth, elevation)
        
#         # Project to 2D (take x, y coordinates)
#         points_2d = rotated_points[:, :, :2]
#         z_coords = rotated_points[:, :, 2]
        
#         # Render using Gaussian splatting
#         images = self.render_gaussian(points_2d, z_coords, self.img_size)
        
#         return images
    