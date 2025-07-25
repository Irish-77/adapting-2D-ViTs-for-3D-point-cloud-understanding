import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import get_vit
from timm.models.layers import DropPath, Mlp

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
        - ChatGPT helped with the vectorized implementation
    """
    
    @staticmethod
    def part1by2_vectorized(n: torch.Tensor) -> torch.Tensor:
        """Vectorized version of part1by2 that operates on tensors.
        
        Args:
            n: Integer tensor to process, representing coordinate values.
            
        Returns:
            Tensor with bits separated by 2 positions.
        """
        n = n & 0x000003ff
        n = (n ^ (n << 16)) & 0xff0000ff
        n = (n ^ (n << 8)) & 0x0300f00f
        n = (n ^ (n << 4)) & 0x030c30c3
        n = (n ^ (n << 2)) & 0x09249249
        return n
    
    @staticmethod
    def encode_morton3_vectorized(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Vectorized version of encode_morton3 that operates on tensors.
        
        Args:
            x: X-coordinate tensor.
            y: Y-coordinate tensor.
            z: Z-coordinate tensor.
            
        Returns:
            Morton codes tensor.
        """
        return (MortonEncoder.part1by2_vectorized(z) << 2) + \
               (MortonEncoder.part1by2_vectorized(y) << 1) + \
                MortonEncoder.part1by2_vectorized(x)
    
    @staticmethod
    def points_to_morton(points: torch.Tensor, resolution: int = 1024) -> torch.Tensor:
        """Convert points to Morton order and return sorting indices.
        
        This method processes a batch of 3D point clouds and converts each point
        to its corresponding Morton code. The steps are:
        
        1. Normalize the points to fit within [0, resolution-1]
        2. Convert to integer coordinates
        3. Compute Morton code for each point using encode_morton3_vectorized
        4. Return indices that would sort the points by their Morton codes
        
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
        
        # Extract x, y, z components
        x = points_discrete[..., 0]  # (B, N)
        y = points_discrete[..., 1]  # (B, N)
        z = points_discrete[..., 2]  # (B, N)
        
        # Compute Morton codes for all points at once
        morton_codes = MortonEncoder.encode_morton3_vectorized(x, y, z)  # (B, N)
        
        # Get sorting indices
        indices = torch.argsort(morton_codes, dim=1)
        return indices

class AttentionLayer(nn.Module):
    """Multi-head self-attention layer
    
    This implements the standard multi-head attention mechanism.
    Implementation must match original one so that model weights can be loaded.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
    ) -> None:
        """
        Args:
            dim: Input dimension of features
            num_heads: Number of attention heads
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Model components
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the attention layer.

        Args:
            x: Input tensor of shape (batch_size, num_points, in_channels)

        Returns:
            x: Output tensor of shape (batch_size, num_points, in_channels)
        """
        B, N, C = x.shape
        # Reshape to (B, N, 3, num_heads, head_dim)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        # Extract q, k, v
        q, k, v = qkv[0], qkv[1], qkv[2]
        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Apply softmax to get attention weights
        attn = attn.softmax(dim=-1)
        
        # Weighted sum of values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        # Apply final projection
        x = self.proj(x)

        return x

class AdapterLayer(nn.Module):
    """Bottleneck adapter module for efficient fine-tuning"""
    
    def __init__(
        self,
        model_dimension: int = 768,
        bottleneck: int = 64,
        dropout: float = 0.0,
    ) -> None:
        """
        Args:
            model_dimension: Dimension of the input features
            bottleneck: Dimension of the bottleneck projection
            dropout: Dropout probability applied in the bottleneck
        """
        super().__init__()

        self.n_embd = model_dimension
        self.down_size = bottleneck
        self.dropout = dropout

        # Model components
        self.adapter_norm = nn.LayerNorm(self.n_embd)
        self.scale = nn.Parameter(torch.ones(1))
        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.relu = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)

        # Initialize weights
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
            nn.init.zeros_(self.up_proj.weight)
            nn.init.zeros_(self.down_proj.bias)
            nn.init.zeros_(self.up_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the adapter layer.

        Explanation:
        1. Apply layer normalization to the input.
        2. Down-project the input to a lower-dimensional space.
        3. Apply ReLU activation.
        4. Apply dropout.
        5. Up-project back to the original dimension.
        6. Scale the output and add residual connection.

        Paper has nice visualization of this process:
        https://arxiv.org/html/2407.13200v2/x4.png

        Args:
            x: Input tensor of shape (batch_size, num_points, in_channels)
        Returns:
            Output tensor of shape (batch_size, num_points, in_channels)
        """

        residual = x
        
        # Step 1
        x = self.adapter_norm(x)
        # Step 2
        down = self.down_proj(x)
        # Step 3
        down = self.relu(down)
        # Step 4
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        # Step 5
        up = self.up_proj(down)
        # Step 6
        up = up * self.scale
        # Add residual connection
        output = up + residual
        return output


class APFViTLayer(nn.Module):
    """Vision Transformer Layer with attention, MLP, and adapter components"""

    def __init__(
        self,
        dim: int = 768,
        num_heads: int = 12,
        drop_path: float = 0.0,
        dropout: float = 0.1,
    ) -> None:
        """
        Args:
            dim: Hidden dimension of the layer
            num_heads: Number of attention heads
            drop_path: Drop path rate for stochastic depth regularization
            dropout: Dropout probability for the adapter
        """
        super().__init__()

        # Model components
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()        
        self.mlp = Mlp(
            in_features = dim,
            hidden_features = dim * 4,
        )
        self.attention = AttentionLayer(
            dim=dim,
            num_heads=num_heads,
        )
        self.adapter = AdapterLayer(model_dimension=dim, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the ViT layer.
        
        Args:
            x: Input tensor of shape (batch_size, num_points, in_channels)
        Returns:
            x: Output tensor of shape (batch_size, num_points, in_channels)
        """
        # Attention
        x_attn = self.norm1(x)
        x_attn = self.attention(x_attn)
        x_attn = self.drop_path(x_attn)
        x = x + x_attn
        residual = x
        
        # Apply the adaptation layer
        adapt_x = self.adapter(x)

        # Apply the MLP
        x_mlp = self.norm2(x)
        x_mlp = self.mlp(x_mlp)
        x_mlp = self.drop_path(x_mlp)
        
        x = x_mlp + adapt_x + residual
        return x


class SimpleAdapterLayer(nn.Module):
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
    """ViT block with adapter layers for parameter-efficient fine-tuning.
    Simpler implementation used by renderer.
    """
    
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
        
        self.adapter1 = SimpleAdapterLayer(hidden_dim, adapter_dim)
        self.adapter2 = SimpleAdapterLayer(hidden_dim, adapter_dim)
        
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