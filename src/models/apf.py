import torch
from torch import nn
from data.sampler import (
    furthest_point_sample,
    index_points,
    knn_point
)
from models import get_timm_vit
from models.apf_utils import APFViTLayer
from models.apf_utils import MortonEncoder

class Group(nn.Module):
    """Group points in a point cloud based on their spatial coordinates using Morton encoding"""
    
    def __init__(self, num_group: int, group_size: int):
        """Initialize the Group module.
        
        Args:
            num_group: Number of groups to create from the point cloud
            group_size: Number of points in each group
        """
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.morton_encoder = MortonEncoder()

    def _morton_sorting(self, xyz: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
        """Sorts the points based on their spatial coordinates using Morton encoding.
        
        Args:
            xyz: Tensor of shape (B, N, C) with point coordinates
            center: Tensor of shape (B, G, C) with group center coordinates

        Returns:
            Tensor of sorted indices
        """
        batch_size, num_points, _ = xyz.shape
        distances_batch = torch.cdist(center, center)
        distances_batch[:, torch.eye(self.num_group).bool()] = float("inf")
        idx_base = torch.arange(
            0, batch_size, device=xyz.device) * self.num_group

        sorted_indices = self.morton_encoder.points_to_morton(center)
        # For batched data, we need to adjust the indices
        batch_size = center.shape[0]
        idx_base = torch.arange(0, batch_size, device=xyz.device) * self.num_group
        sorted_indices = sorted_indices + idx_base.unsqueeze(1)
        sorted_indices = sorted_indices.view(-1)

        return sorted_indices
    
    def forward(self, x: torch.tensor, xyz: torch.Tensor):
        """ Forward pass to group points based on their spatial coordinates.
        
        Args:
            x: Tensor of shape (B, N, C)
            xyz: Tensor with spatial coordinates of shape

        Returns:
            neighborhood: Tensor of shape (B, G, nsample, C)
            center: Tensor of shape (B, G, C)
        """
       
        batch_size, num_points, _ = xyz.shape
        xyz = xyz.contiguous()
        
        # Create evemly spaced groups
        fps_idx = furthest_point_sample(xyz,self.num_group).long()
        # Create anchor points
        center = index_points(xyz,fps_idx)
        new_points = index_points(x,fps_idx)
        # Associate points to the nearest center
        idx = knn_point(self.group_size, xyz, center)  # B G nsample
        idx_base = torch.arange(
            0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = x.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(
            batch_size, self.num_group, self.group_size, -1).contiguous() # B,G,nsample,3
        
        # Normalize neighborhood
        mean_x = new_points.unsqueeze(dim=-2)
        neighborhood = (neighborhood - mean_x)
        # Concats:
        # 1. Local point features
        # 2. Features of its group center
        neighborhood = torch.cat(
            [
                neighborhood,
                # Operations just make sure that dimensions match to neighborhood
                new_points.unsqueeze(2).repeat(1, 1, self.group_size, 1)
            ],
            dim = -1
        )

        # Sort the neighborhood and center based on the Morton code (it looks like Z curves)
        # For more informatio, see docu of MortonEncoder class
        sorted_indices = self._morton_sorting(xyz, center)
        # Now we do sorting based on Morton results
        # Rearrange the neighborhood point group
        neighborhood = neighborhood.view(
            batch_size * self.num_group, self.group_size, -1)[sorted_indices, :, :]
        neighborhood = neighborhood.view(
            batch_size, self.num_group, self.group_size, -1).contiguous()
        # Repeat for center points
        center = center.view(
            batch_size * self.num_group, -1)[sorted_indices, :]
        center = center.view(
            batch_size, self.num_group, -1).contiguous()

        return neighborhood, center

class Encoder(nn.Module):
    """Feature encoder for point groups using shared MLPs.
    
    Extracts features & local structures from point groups using MLPs.
    """
    
    def __init__(self, encoder_channel: int, in_channel: int):
        """Initialize the Encoder module.
        
        Args:
            encoder_channel: Number of channels for the encoder MLPs
            in_channel: Number of input channels for the point features
        """
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(in_channel, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(self.encoder_channel*2, self.encoder_channel*2, 1),
            nn.BatchNorm1d(self.encoder_channel*2),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.encoder_channel*2, self.encoder_channel, 1)
        )

    def get_features(self, point_groups: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to extract features from point groups.
        
        Args:
            point_groups: Tensor of shape (B, G, N, C) where B is batch size, G is number of groups, N is number of points per group, and C is number of channels.

        Returns:
            Tensor of shape (B, G, C) representing the global feature for each group.
        """
        B, G, N, _ = point_groups.shape
        # Reshape to BG N 3
        point_groups = point_groups.reshape(B * G, N, -1)
        # Apply first conv
        feature = self.first_conv(point_groups.transpose(2, 1))
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]
        # Concat global feature with local features
        feature = torch.cat(
            [feature_global.expand(-1, -1, N), feature], dim=1)
        # Apply second conv
        feature = self.second_conv(feature)
        # Retrieve global feature
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]
        # Reshape back to B G C
        return feature_global.reshape(B, G, self.encoder_channel)

    def forward(self, point_groups: torch.Tensor) -> torch.Tensor:
        """Extract features from point groups.
        
        Args:
            point_groups: Tensor of shape (B, G, N, C)

        Returns:
            Tensor of shape (B, G, C) with extracted features
        """
        feature = self.get_features(point_groups)
        return feature

class PointNet(nn.Module):
    """PointNet-based feature extractor for point clouds
    
    Groups points, encodes the groups into features (permutation invariant).
    """
    
    def __init__(self, embed_dim: int, num_group: int, group_size: int, in_channel: int):
        """Initialize the PointNet module.
        
        Args:
            embed_dim: Dimension of the embedding space
            num_group: Number of groups to create from the point cloud
            group_size: Number of points in each group
            in_channel: Number of input channels for the point features
        """
        super().__init__()
        self.group = Group(num_group, group_size)
        self.encoder = Encoder(embed_dim, in_channel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from the point cloud.
        
        Args:
            x: Input tensor of shape (B, N, C) where B is batch size, N is number of points, and C is number of channels

        Returns:
            Tensor of shape (B, G, C) with extracted features
        """
        # Extract xyz coordinates
        xyz = x[:,:,:3]
        # Group points
        new_points, new_xyz = self.group(x,xyz)
        # Encode grouped points
        new_points = self.encoder(new_points)
        return new_points

class ClassificationHead(nn.Module):
    """Classification head that processes high-dimensional features into class predictions"""
    
    def __init__(self, in_channels: int, num_classes: int):
        """Initialize the classification head.
        
        Args:
            in_channels: Dimension of input features
            num_classes: Number of output classes for classification
        """
        super(ClassificationHead, self).__init__()
        self.mlp_head = nn.Sequential(
            nn.Linear(in_channels,512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(512,256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(256,num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process features through the classification head.
        
        Args:
            x: Input features tensor of shape (B, in_channels) where B is batch size
            
        Returns:
            Class logits tensor of shape (B, num_classes)
        """
        return self.mlp_head(x)

class AdaptPointFormer(nn.Module):
    """Adapted PointFormer model for point cloud classification.

    Implementation according to the paper:
    https://arxiv.org/pdf/2407.13200

    The official implementation can be found here:
    https://github.com/Dali936/APF

    Note: some blocks were adapted as the paper description is in some parts not clear.
    """
    
    def __init__(
        self,
        num_classes: int = 15,
        embedding_dim: int = 768,
        vit_name: str = 'vit_base_patch16_224',
        pretrained: bool = True,
        npoint: int = 196,
        nsample: int = 32,
        in_channels: int = 3,
        dropout_rate: float = 0.1,
        dropout_path_rate: float = 0.1,
    ) -> None:
        """Initialize the AdaptPointFormer model.

        Args:
            num_classes: Number of output classes for classification
            embedding_dim: Dimension of the embedding space
            vit_name: Name of the ViT architecture to use
            pretrained: Whether to load pre-trained weights for the ViT
            npoint: Number of points to sample for each point cloud
            nsample: Number of points to sample in each group
            in_channels: Number of input channels for the point features
            dropout_rate: Dropout rate for the model
            dropout_path_rate: Dropout path rate for the model
        """
        super().__init__()

        # Multiply in_channels by 2 as model expects grouped features
        in_channels = in_channels * 2

        depth = 12
        
        # Calculate dropout path rates: the more layers, the more dropout
        dpr = [x.item() for x in torch.linspace(0, dropout_path_rate, depth)]

        # Model components
        self.dropout = nn.Dropout(dropout_rate)
        self.encoder_norm = nn.LayerNorm(embedding_dim)
        self.point_encoder = PointNet(
            embedding_dim,
            npoint,
            nsample,
            in_channels,
        )
        self.head = ClassificationHead(
            in_channels=embedding_dim,
            num_classes=num_classes,
        )
        self.blocks = nn.Sequential(*[
            APFViTLayer(
                dim = embedding_dim,
                num_heads = 12,
                drop_path = dpr[i],
                dropout = dropout_rate
            )
            for i in range(depth)])

        # Load pre-trained ViT weights
        vit_state_dict = get_timm_vit(
            vit_name,
            pretrained=pretrained,
            delete=['head.weight', 'head.bias']
        )
        # Load the state dict into this model
        self.load_state_dict(
            vit_state_dict, strict=False
        )

        self._freeze()

    def _freeze(self):
        """
        Freezes the model parameters.
        """
        # First, freeze all parameters
        for param in self.parameters():
            param.requires_grad_(False)

        # Then, unfreeze specific layers
        for name, param in self.named_parameters():
            if 'adaptmlp' in name or 'head' in name or 'enc_norm' in name or 'encoder' in name:
                param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the AdaptPointFormer model.
        
        Args:
            x: Input tensor of shape (B, N, C) where B is batch size, N is number of points, and C is number of channels

        Returns:
            Tensor of shape (B, num_classes) with class logits
        """
        # Get point grouping and encoding
        x = self.point_encoder(x)

        # Pass through ViT with Adaption Layer
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)

        x = self.encoder_norm(x)
        # Global pooling
        x = x.max(-2)[0]
        
        x = self.dropout(x)

        # Final classification head
        x = self.head(x)

        return x