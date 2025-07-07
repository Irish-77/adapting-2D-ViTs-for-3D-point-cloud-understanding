import torch
import torch.nn as nn
import torch.nn.functional as F
from models import get_vit
from typing import Tuple, Iterator
from models.apf import AdaptedViTBlock
from models.diff_renderer import DifferentiablePointCloudRenderer, ViewTransformationNetwork

class PointCloudRenderer(nn.Module):
    """Renders point clouds as multi-view images.
    
    This class provides functionality to render 3D point clouds as 2D images from
    multiple viewpoints. The rendering process involves projecting the points onto
    2D image planes from different camera positions, and applying a simple shading
    based on depth values.
    """
    
    def __init__(self, img_size: int = 224, num_views: int = 6) -> None:
        """Initialize the point cloud renderer.
        
        Args:
            img_size: Size of the rendered images (img_size × img_size).
            num_views: Number of viewpoints to render the point cloud from.
        """
        super().__init__()
        self.img_size = img_size
        self.num_views = num_views
        
        # Define view angles for rendering
        self.azimuth_angles = torch.linspace(0, 360, num_views + 1)[:-1]
        self.elevation_angles = torch.tensor([0, 30, -30, 0, 0, 0])[:num_views]
    
    def project_points(self, points: torch.Tensor, azimuth: torch.Tensor, 
                       elevation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project 3D points to 2D using spherical coordinates.
        
        Args:
            points: Tensor of shape (B, N, 3) containing 3D coordinates.
            azimuth: Azimuth angle in degrees.
            elevation: Elevation angle in degrees.
            
        Returns:
            tuple: A tuple containing:
                - 2D projected points tensor of shape (B, N, 2)
                - Depth values tensor of shape (B, N)
        """
        # Convert angles to radians
        azimuth_rad = azimuth * torch.pi / 180
        elevation_rad = elevation * torch.pi / 180
        
        # Rotation matrices
        cos_az = torch.cos(azimuth_rad)
        sin_az = torch.sin(azimuth_rad)
        cos_el = torch.cos(elevation_rad)
        sin_el = torch.sin(elevation_rad)
        
        # Apply rotation for view
        x, y, z = points[..., 0], points[..., 1], points[..., 2]
        
        # Azimuth rotation (around y-axis)
        x_rot = x * cos_az - z * sin_az
        z_rot = x * sin_az + z * cos_az
        
        # Elevation rotation (around x-axis)
        y_rot = y * cos_el - z_rot * sin_el
        z_final = y * sin_el + z_rot * cos_el
        
        # Project to 2D (orthographic projection)
        return torch.stack([x_rot, y_rot], dim=-1), z_final
        
    def _render_view(self, points: torch.Tensor, azimuth: torch.Tensor, 
                    elevation: torch.Tensor) -> torch.Tensor:
        """Fast rendering using scatter operations.
        
        Args:
            points: Tensor of shape (B, N, 3) containing 3D coordinates.
            azimuth: Azimuth angle in degrees.
            elevation: Elevation angle in degrees.
            
        Returns:
            Rendered images tensor of shape (B, 3, img_size, img_size).
        """
        B, N, _ = points.shape
        device = points.device
        
        # Project points to 2D
        points_2d, z_coords = self.project_points(points, azimuth, elevation)
        
        # Normalize depth
        z_normalized = (z_coords - z_coords.min(dim=1, keepdim=True)[0]) / \
                      (z_coords.max(dim=1, keepdim=True)[0] - z_coords.min(dim=1, keepdim=True)[0] + 1e-6)
        
        # Create point features (intensity based on depth)
        point_features = (0.3 + 0.7 * z_normalized).unsqueeze(-1).expand(-1, -1, 3)  # (B, N, 3)
        
        # Scale points to grid coordinates [-1, 1]
        grid_coords = points_2d.unsqueeze(1)  # (B, 1, N, 2)
        
        # Create a small kernel around each point
        kernel_size = 5
        offset = torch.linspace(-2/self.img_size, 2/self.img_size, kernel_size, device=device)
        dy, dx = torch.meshgrid(offset, offset, indexing='ij')
        kernel_offsets = torch.stack([dx, dy], dim=-1).reshape(-1, 2)  # (kernel_size^2, 2)
        
        # Expand grid coordinates with kernel
        expanded_coords = grid_coords.unsqueeze(3) + kernel_offsets.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        expanded_coords = expanded_coords.reshape(B, 1, -1, 2)  # (B, 1, N*kernel_size^2, 2)
        
        # Expand features accordingly
        expanded_features = point_features.unsqueeze(2).expand(-1, -1, kernel_size*kernel_size, -1)
        expanded_features = expanded_features.reshape(B, -1, 3).permute(0, 2, 1)  # (B, 3, N*kernel_size^2)
        
        # Create images using grid_sample inverse
        images = torch.zeros(B, 3, self.img_size, self.img_size, device=device)
        
        # Simple scatter-based approach for speed
        for b in range(B):
            # Convert normalized coords to pixel coords
            px = ((expanded_coords[b, 0, :, 0] + 1) * 0.5 * (self.img_size - 1)).long()
            py = ((expanded_coords[b, 0, :, 1] + 1) * 0.5 * (self.img_size - 1)).long()
            
            # Valid mask
            valid = (px >= 0) & (px < self.img_size) & (py >= 0) & (py < self.img_size)
            
            if valid.any():
                px_valid = px[valid]
                py_valid = py[valid]
                features_valid = expanded_features[b, :, valid]
                
                # Scatter to image
                for c in range(3):
                    images[b, c].reshape(-1).scatter_reduce_(
                        0, 
                        py_valid * self.img_size + px_valid,
                        features_valid[c],
                        reduce='amax'
                    )
        
        return images
        
    def render_view(self, points: torch.Tensor, azimuth: torch.Tensor, 
                   elevation: torch.Tensor) -> torch.Tensor:
        """Render a single view of a point cloud.
        
        Args:
            points: Tensor of shape (B, N, 3) containing 3D coordinates.
            azimuth: Azimuth angle in degrees.
            elevation: Elevation angle in degrees.
            
        Returns:
            Rendered images tensor of shape (B, 3, img_size, img_size).
        """
        return self._render_view(points, azimuth, elevation)
        
    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """Render point cloud from multiple views.
        
        Args:
            points: Tensor of shape (B, N, 3) containing 3D point coordinates.
            
        Returns:
            Tensor of shape (B, num_views, 3, img_size, img_size) containing
            rendered views from different angles.
        """
        B = points.shape[0]
        device = points.device
        
        # Move angle tensors to device
        azimuth_angles = self.azimuth_angles.to(device)
        elevation_angles = self.elevation_angles.to(device)
        
        # Render from multiple views
        views = []
        for i in range(self.num_views):
            view = self.render_view(points, azimuth_angles[i], elevation_angles[i])
            views.append(view)
            
        return torch.stack(views, dim=1)  # (B, num_views, 3, H, W)


class PointCloudRendererClassifier(nn.Module):
    """Point cloud classifier using rendered views and ViT with adapters.
    
    This model renders a point cloud from multiple viewpoints and processes the resulting
    images through a Vision Transformer with adapter layers for classification.

    Option to use a differentiable renderer for training based on the MVTN paper.
    """
    
    def __init__(
        self,
        num_classes: int = 15,  # ScanObjectNN has 15 classes
        vit_name: str = 'vit_b_16',
        adapter_dim: int = 64,
        num_views: int = 6,
        img_size: int = 224,
        diff_renderer: bool = False,
        view_transform_hidden: int = 256,
        pretrained: bool = True,
        dropout_rate: float = 0.1
    ) -> None:
        """Initialize the PointCloudRendererClassifier.
        
        Args:
            num_classes: Number of output classes for classification.
            vit_name: Name of the Vision Transformer backbone to use.
            adapter_dim: Dimension of the bottleneck in adapter layers.
            num_views: Number of viewpoints to render the point cloud from.
            img_size: Size of the rendered images (img_size × img_size).
            diff_renderer: Whether to use a differentiable renderer.
            view_transform_hidden: Hidden dimension for the view transformation network.
            pretrained: Whether to use pretrained weights for the ViT.
            dropout_rate: Dropout rate for regularization.
        """
        super().__init__()
        
        self.num_views = num_views
        self.diff_renderer = diff_renderer
        if diff_renderer is False:
            # Default Point cloud renderer
            self.renderer = PointCloudRenderer(img_size=img_size, num_views=num_views)    
        else:
            # Use DiffRenderer for point cloud rendering
            self.view_transform_net = ViewTransformationNetwork(
                num_views=num_views,
                hidden_dim=view_transform_hidden
            )
            self.renderer = DifferentiablePointCloudRenderer(img_size=img_size)
        
        # Load pre-trained ViT
        self.vit, self.embed_dim = get_vit(vit_name, pretrained)
        # Replace ViT's head with an identity layer to get feature embeddings
        self.vit.heads = nn.Identity()
        
        # Replace ViT blocks with adapted versions
        adapted_blocks = []
        for block in self.vit.encoder.layers:
            adapted_blocks.append(AdaptedViTBlock(block, adapter_dim))
        self.vit.encoder.layers = nn.Sequential(*adapted_blocks)
        
        # Freeze all ViT parameters except adapters
        for name, param in self.vit.named_parameters():
            if 'adapter' not in name:
                param.requires_grad = False
                
        # View aggregation
        self.view_aggregation = nn.Parameter(torch.ones(num_views) / num_views)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
    
    def _get_rendered_views(self, points: torch.Tensor) -> torch.Tensor:
        """Render point cloud from multiple views.
        
        Args:
            points: Point cloud tensor of shape (B, N, 3) where B is batch size,
                   N is number of points, and 3 represents XYZ coordinates.
        
        Returns:
            Rendered views tensor of shape (B, num_views, 3, img_size, img_size).
        """
        if self.diff_renderer is False:
            return self.renderer(points)
        else:
            # Old approach with for loop (not vectorized)
            # azimuths, elevations = self.view_transform_net(points)  # (B, num_views) each
            # # Render point cloud from learned views
            # rendered_views = []
            # for v in range(self.num_views):
            #     view_img = self.renderer(points, azimuths[:, v], elevations[:, v])
            #     rendered_views.append(view_img)
            # # Stack views
            # rendered_views = torch.stack(rendered_views, dim=1)  # (B, num_views, 3, H, W)

            # Vectorized rendering for all views
            B, N, _ = points.shape
            azimuths, elevations = self.view_transform_net(points)
            points_expanded = points.unsqueeze(1).expand(-1, self.num_views, -1, -1).reshape(B * self.num_views, N, 3)
            azimuths_flat = azimuths.reshape(B * self.num_views)
            elevations_flat = elevations.reshape(B * self.num_views)
            rendered_views_flat = self.renderer(points_expanded, azimuths_flat, elevations_flat)
            _, C, H, W = rendered_views_flat.shape
            rendered_views = rendered_views_flat.reshape(B, self.num_views, C, H, W)

            return rendered_views
        
    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """Process point clouds through rendering and classification.
        
        Args:
            points: Point cloud tensor of shape (B, N, 3) where B is batch size,
                   N is number of points, and 3 represents XYZ coordinates.
        
        Returns:
            Classification logits of shape (B, num_classes).
        """
        B = points.shape[0]
        
        # Render point cloud from multiple views
        rendered_views = self._get_rendered_views(points) 
        
        # Process each view through ViT
        view_features = []
        for v in range(self.num_views):
            view_img = rendered_views[:, v]  # (B, 3, H, W)
            # Forward through ViT
            features = self.vit(view_img)  # (B, embed_dim)
            view_features.append(features)
            
        # Stack view features
        view_features = torch.stack(view_features, dim=1)  # (B, num_views, embed_dim)
        
        # Aggregate views with learned weights
        view_weights = F.softmax(self.view_aggregation, dim=0)
        aggregated_features = torch.sum(view_features * view_weights.unsqueeze(0).unsqueeze(-1), dim=1)
        
        # Classification
        logits = self.classifier(aggregated_features)
        
        return logits
    
    def get_trainable_params(self) -> Iterator[nn.Parameter]:
        """Return only trainable parameters for optimizer.
        
        Returns:
            Iterator over trainable parameters of the model.
        """
        return filter(lambda p: p.requires_grad, self.parameters())
    
    def count_parameters(self) -> Tuple[int, int]:
        """Count trainable and total parameters.
        
        Returns:
            A tuple containing:
                - Number of trainable parameters
                - Total number of parameters
        """
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total

    def get_predicted_views(self, points: torch.Tensor) -> torch.Tensor:
        """Get the predicted view angles for visualization
        
        Args:
            points: Point cloud tensor of shape (B, N, 3).
        
        Returns:
            Rendered views tensor of shape (B, num_views, 3, img_size, img_size).
        """
        if self.diff_renderer is False:
            raise ValueError("This method is only available when using a differentiable renderer.")
        
        with torch.no_grad():
            azimuths, elevations = self.view_transform_net(points)
            # Convert to degrees
            azimuths_deg = azimuths * 180 / torch.pi
            elevations_deg = elevations * 180 / torch.pi
        return azimuths_deg, elevations_deg