import torch
import torch.nn as nn
import torch.nn.functional as F
from models import get_vit
from typing import Optional, Tuple, Iterator, Callable

class MortonEncoder:
    """Encodes 3D coordinates to Morton order (Z-order curve).
    
    The MortonEncoder provides functionality to convert 3D point coordinates into
    Morton order indices. Morton ordering (also known as Z-order curve) is a 
    space-filling curve that preserves locality, meaning points that are close in 
    3D space tend to be close in the 1D Morton ordering. This property is useful
    for organizing point cloud data in a way that respects spatial locality.
    
    The encoding process involves:
    1. Normalizing point coordinates to a discrete grid
    2. Interleaving the bits of the x, y, z coordinates
    3. Creating a single integer value (Morton code) for each point
    4. Sorting points based on their Morton codes
    
    This ordering helps preserve spatial relationships when processing point clouds
    sequentially, which is especially beneficial for transformer architectures
    that rely on sequential data.
    
    Credits (**heavily** inspired by):
        - https://github.com/trevorprater/pymorton/blob/master/pymorton/pymorton.py
        - https://fgiesen.wordpress.com/2009/12/13/decoding-morton-codes/
    """
    
    @staticmethod
    def part1by2(n: int) -> int:
        """Separate bits by 2 positions to prepare for Morton encoding.
        
        This function takes an integer and spreads its bits apart, inserting two
        zeros between each bit. This is a core operation in converting from
        Cartesian coordinates to Morton code.
        
        For example, if n = 5 (binary 101), the function will transform it to
        10001 (binary representation of 17), effectively separating each bit
        by two positions.
        
        Args:
            n: Integer value to process, representing a coordinate value.
            
        Returns:
            Integer with bits separated by 2 positions.
        """
        n &= 0x000003ff
        n = (n ^ (n << 16)) & 0xff0000ff
        n = (n ^ (n << 8)) & 0x0300f00f
        n = (n ^ (n << 4)) & 0x030c30c3
        n = (n ^ (n << 2)) & 0x09249249
        return n
    
    @staticmethod
    def encode_morton3(x: int, y: int, z: int) -> int:
        """Encode (x,y,z) coordinates to a single Morton code.
        
        This function combines three integer coordinates into a single Morton code
        by interleaving their bits. Each coordinate is first processed with part1by2
        to separate its bits, then they are combined so that the bits of x, y, and z
        are interleaved in the pattern: z-y-x, z-y-x, etc.
        
        The resulting Morton code has the property that nearby points in 3D space
        will typically have similar Morton codes, preserving spatial locality.
        
        Args:
            x: X-coordinate integer.
            y: Y-coordinate integer.
            z: Z-coordinate integer.
            
        Returns:
            Morton code representing the 3D position with interleaved bits.
        """
        return (MortonEncoder.part1by2(z) << 2) + \
               (MortonEncoder.part1by2(y) << 1) + \
                MortonEncoder.part1by2(x)
    
    @staticmethod
    def points_to_morton(points: torch.Tensor, resolution: int = 1024) -> torch.Tensor:
        """Convert points to Morton order and return sorting indices.
        
        This method processes a batch of 3D point clouds and converts each point
        to its corresponding Morton code. The steps are:
        
        1. Normalize the points to fit within [0, resolution-1]
        2. Convert to integer coordinates
        3. Compute Morton code for each point using encode_morton3
        4. Return indices that would sort the points by their Morton codes
        
        The returned indices can be used to reorder the original points to follow
        the Morton space-filling curve, which helps maintain spatial locality
        when processing points sequentially.
        
        Args:
            points: Tensor of shape (B, N, 3) containing 3D coordinates.
            resolution: Discretization resolution for normalizing coordinates.
                       Higher values provide more precise spatial encoding.
            
        Returns:
            Tensor of shape (B, N) containing indices for sorting by Morton order.
        """
        B, N, _ = points.shape
        
        # Normalize points to [0, resolution)
        points_min = points.min(dim=1, keepdim=True)[0]
        points_max = points.max(dim=1, keepdim=True)[0]
        points_normalized = (points - points_min) / (points_max - points_min + 1e-8)
        points_discrete = (points_normalized * (resolution - 1)).long()
        
        # Compute Morton codes
        morton_codes = torch.zeros(B, N, dtype=torch.long, device=points.device)
        for b in range(B):
            for n in range(N):
                x, y, z = points_discrete[b, n]
                morton_codes[b, n] = MortonEncoder.encode_morton3(
                    x.item(), y.item(), z.item()
                )
        
        # Get sorting indices
        indices = torch.argsort(morton_codes, dim=1)
        return indices

class PointEmbedding(nn.Module):
    """Point cloud embedding and sequencing module for grouped points."""
    
    def __init__(self, embed_dim: int = 768, k_neighbors: int = 16, dropout_rate: float = 0.1) -> None:
        """Initialize the PointEmbedding module.
        
        Args:
            embed_dim: Dimension of the embedding output.
            k_neighbors: Number of nearest neighbors for point grouping.
            dropout_rate: Dropout rate for regularization.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.k_neighbors = k_neighbors
        # (B*n_samples, k, 3) -> Output: (B*n_samples, embed_dim)
        self.group_embed = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )
        
        # Aggregation layer
        self.aggregation = nn.Sequential(
            nn.Linear(256, embed_dim),
            nn.Dropout(dropout_rate)
        )
        
        # Learnable position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, 1024, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
    def forward(self, grouped_points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process grouped points through the embedding network.
        
        Args:
            grouped_points: Tensor of shape (B, n_samples, k, 3) with grouped 3D coordinates.
            
        Returns:
            tuple: A tuple containing:
                - embedded points: Tensor of shape (B, n_samples, embed_dim)
                - indices: Tensor of shape (B, n_samples) with Morton ordering indices
        """
        B, n_samples, k, _ = grouped_points.shape
        
        # Get centroids (mean of each group) for Morton ordering
        centroids = grouped_points.mean(dim=2)  # (B, n_samples, 3)
        
        # Apply Morton ordering to centroids
        indices = MortonEncoder.points_to_morton(centroids)
        
        # Reorder grouped points according to Morton order
        grouped_points = torch.gather(
            grouped_points, 
            1, 
            indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, k, 3)
        )
        
        # Embed grouped points
        # Reshape for conv1d: (B*n_samples, k, 3) -> (B*n_samples, 3, k)
        x = grouped_points.reshape(B * n_samples, k, 3).transpose(1, 2)
        x = self.group_embed(x)  # (B*n_samples, 256, k)
        
        # Max pooling over neighbors
        x = F.max_pool1d(x, kernel_size=k).squeeze(-1)  # (B*n_samples, 256)
        
        # Final projection
        x = self.aggregation(x)  # (B*n_samples, embed_dim)
        x = x.reshape(B, n_samples, self.embed_dim)
        
        # Add position embeddings
        if n_samples <= self.pos_embed.shape[1]:
            x = x + self.pos_embed[:, :n_samples, :]
        else:
            # Interpolate position embeddings if needed
            pos_embed = F.interpolate(
                self.pos_embed.transpose(1, 2),
                size=n_samples,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)
            x = x + pos_embed
        
        return x, indices

class AdapterLayer(nn.Module):
    """Adapter layer for parameter-efficient fine-tuning."""
    
    def __init__(self, embed_dim: int, adapter_dim: int = 64) -> None:
        """Initialize the adapter layer.
        
        Args:
            embed_dim: Dimension of the input and output embeddings.
            adapter_dim: Dimension of the bottleneck in the adapter.
        """
        super().__init__()
        self.down_proj = nn.Linear(embed_dim, adapter_dim)
        self.up_proj = nn.Linear(adapter_dim, embed_dim)
        self.act = nn.GELU()
        
        # Initialize with near-identity
        nn.init.xavier_uniform_(self.down_proj.weight, gain=1e-3)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process input through the adapter layer.
        
        Args:
            x: Input tensor of shape (..., embed_dim).
            
        Returns:
            Output tensor of shape (..., embed_dim) after adaptation.
        """
        return x + self.up_proj(self.act(self.down_proj(x)))

class AdaptedViTBlock(nn.Module):
    """ViT block with adapter layers for parameter-efficient fine-tuning."""
    
    def __init__(self, vit_block: nn.Module, adapter_dim: int = 64) -> None:
        """Initialize the adapted ViT block.
        
        Args:
            vit_block: Original Vision Transformer block to be adapted.
            adapter_dim: Dimension of the bottleneck in the adapters.
        """
        super().__init__()
        self.vit_block = vit_block
        
        # Get the hidden dimension from the layer normalization
        hidden_dim = vit_block.ln_1.normalized_shape[0]
        
        self.adapter1 = AdapterLayer(hidden_dim, adapter_dim)
        self.adapter2 = AdapterLayer(hidden_dim, adapter_dim)
        
        # Freeze ViT parameters
        for param in self.vit_block.parameters():
            param.requires_grad = False
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process input through the adapted ViT block.
        
        Args:
            x: Input tensor.
            
        Returns:
            Output tensor after processing through the adapted block.
        """
        # Self-attention block with adapter
        y = self.vit_block.ln_1(x)
        y, _ = self.vit_block.self_attention(y, y, y, need_weights=False)
        y = self.vit_block.dropout(y)
        y = self.adapter1(y)
        x = x + y
        
        # MLP block with adapter
        y = self.vit_block.ln_2(x)
        y = self.vit_block.mlp(y)
        y = self.adapter2(y)
        x = x + y
        
        return x

class AdaptPointFormer(nn.Module):
    """Main Adapt PointFormer model with integrated FPS sampling.
    
    Implementation according to the paper:
    https://arxiv.org/pdf/2407.13200

    The official implementation can be found here:
    https://github.com/Dali936/APF

    However, this implementation used some design choices that we did not like,
    so we decided to implement our own version (also this was part of the project either way).
    """
    
    def __init__(
        self,
        num_classes: int = 40,
        num_points: int = 1024,
        n_samples: int = 512,
        k_neighbors: int = 16,
        vit_name: str = 'vit_b_16',
        adapter_dim: int = 64,
        pretrained: bool = True,
        dropout_rate: float = 0.1
    ) -> None:
        """Initialize the AdaptPointFormer model.
        
        Args:
            num_classes: Number of output classes for classification.
            num_points: Number of points in the input point cloud.
            n_samples: Number of samples after FPS sampling.
            k_neighbors: Number of nearest neighbors in point grouping.
            vit_name: Name of the Vision Transformer backbone to use.
            adapter_dim: Dimension of the bottleneck in adapter layers.
            pretrained: Whether to use pretrained weights for the ViT.
            dropout_rate: Dropout rate for regularization.
        """
        super().__init__()
        
        # Load pre-trained ViT
        self.vit, self.embed_dim = get_vit(vit_name, pretrained)
        self.num_points = num_points
        self.n_samples = n_samples
        self.k_neighbors = k_neighbors
        
        # Point embedding module - now handles grouped points
        self.point_embed = PointEmbedding(
            embed_dim=self.embed_dim,
            k_neighbors=k_neighbors,
            dropout_rate=dropout_rate
        )
        
        # Replace ViT blocks with adapted versions
        encoder_blocks = []
        for block in self.vit.encoder.layers:
            encoder_blocks.append(AdaptedViTBlock(block, adapter_dim))
        self.vit.encoder.layers = nn.Sequential(*encoder_blocks)
        
        # Replace the ViT head
        self.vit.heads = nn.Identity()
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
        # Freeze all ViT parameters except adapters
        for name, param in self.named_parameters():
            if 'adapter' not in name and 'point_embed' not in name and 'classifier' not in name:
                param.requires_grad = False
    
    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.
        
        Args:
            points: Either tensor of shape (B, N, 3) with raw points or 
                  tensor of shape (B, n_samples, k, 3) with grouped points.
            
        Returns:
            Classification logits of shape (B, num_classes).
            
        Raises:
            ValueError: If raw points are provided without FPS sampling implementation.
        """
        # Check if input is already grouped or raw points
        if len(points.shape) == 4:
            # Already grouped points (B, n_samples, k, 3)
            grouped_points = points
            B, n_samples, k, _ = grouped_points.shape
        else:
            # Raw points (B, N, 3) - need to apply FPS sampling
            B, N, _ = points.shape
            # TODO: Check if fps sampling can be introduced here, and if this will make AdaptPointFormerWithSampling obsolete
            # For now, current setup works...
            raise ValueError("Raw points input requires FPS sampling. Please provide grouped points or implement FPS sampling.")
        
        # Embed and sequence grouped points
        x, indices = self.point_embed(grouped_points)  # (B, n_samples, embed_dim)
        
        # Prepare for ViT (add CLS token)
        cls_token = self.vit.class_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        
        # Add position embeddings from ViT
        # Interpolate ViT position embeddings to match number of points
        vit_pos_embed = self.vit.encoder.pos_embedding[:, 1:, :]  # Remove CLS position
        if n_samples != vit_pos_embed.shape[1]:
            vit_pos_embed = F.interpolate(
                vit_pos_embed.transpose(1, 2),
                size=n_samples,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)
        
        cls_pos_embed = self.vit.encoder.pos_embedding[:, :1, :]
        pos_embed = torch.cat([cls_pos_embed, vit_pos_embed], dim=1)
        x = x + pos_embed
        
        # Apply ViT encoder with adapters
        x = self.vit.encoder.dropout(x)
        x = self.vit.encoder.layers(x)
        x = self.vit.encoder.ln(x)
        
        # Extract CLS token and classify
        cls_output = x[:, 0]
        logits = self.classifier(cls_output)
        
        return logits
    
    def get_trainable_params(self) -> Iterator[nn.Parameter]:
        """Get only trainable parameters for optimizer.
        
        Returns:
            Iterator over trainable parameters of the model.
        """
        return filter(lambda p: p.requires_grad, self.parameters())
    
    def print_trainable_params(self) -> None:
        """Print statistics about trainable parameters.
        
        Displays the total number of parameters, trainable parameters,
        and percentage of trainable parameters in the model.
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Trainable percentage: {100 * trainable_params / total_params:.2f}%")


# Wrapper to integrate FPS sampling with the model
class AdaptPointFormerWithSampling(nn.Module):
    """Wrapper that handles FPS sampling before the model."""
    
    def __init__(
        self,
        num_classes: int = 15,
        num_points: int = 1024,
        vit_name: str = 'vit_b_16',
        adapter_dim: int = 64,
        pretrained: bool = True,
        dropout_rate: float = 0.1,
        fps_sampling_func: Optional[Callable[[torch.Tensor, int, int], torch.Tensor]] = None,
        n_samples: int = 512,
        k_neighbors: int = 16,
    ) -> None:
        """Initialize the AdaptPointFormerWithSampling model.
        
        Args:
            num_classes: Number of output classes for classification.
            num_points: Number of points in the input point cloud.
            vit_name: Name of the Vision Transformer backbone to use.
            adapter_dim: Dimension of the bottleneck in adapter layers.
            pretrained: Whether to use pretrained weights for the ViT.
            dropout_rate: Dropout rate for regularization.
            fps_sampling_func: Function to perform FPS sampling with k-NN grouping.
            n_samples: Number of samples after FPS sampling.
            k_neighbors: Number of nearest neighbors in point grouping.
        """
        super().__init__()
        self.model = AdaptPointFormer(
            num_classes=num_classes,
            num_points=num_points,
            vit_name=vit_name,
            adapter_dim=adapter_dim,
            pretrained=pretrained,
            dropout_rate=dropout_rate
        )

        self.fps_sampling_func = fps_sampling_func
        self.n_samples = n_samples
        self.k_neighbors = k_neighbors
    
    def get_trainable_params(self) -> Iterator[nn.Parameter]:
        """Get only trainable parameters for optimizer.
        
        Returns:
            Iterator over trainable parameters of the underlying model.
        """
        return filter(lambda p: p.requires_grad, self.model.parameters())
    
    def print_trainable_params(self) -> None:
        """Print statistics about trainable parameters.
        
        Displays the total number of parameters, trainable parameters,
        and percentage of trainable parameters in the underlying model.
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Trainable percentage: {100 * trainable_params / total_params:.2f}%")

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """Forward pass with integrated FPS sampling.
        
        Args:
            points: Tensor of shape (B, N, 3) containing raw point clouds.
            
        Returns:
            Classification logits of shape (B, num_classes).
        """
        # Apply FPS sampling with k-NN
        grouped_points = self.fps_sampling_func(
            points, 
            n_samples=self.n_samples, 
            k=self.k_neighbors
        )
        
        # Forward through the model
        return self.model(grouped_points)