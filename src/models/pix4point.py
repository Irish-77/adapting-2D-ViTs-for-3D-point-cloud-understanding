import math
import timm
import torch
import torch.nn as nn
from typing import cast, List, Optional, Tuple, Iterator


def farthest_point_sampling(points: torch.Tensor, n_samples: int) -> torch.Tensor:
    """
    Perform farthest point sampling (FPS) on a set of points.

    Args:
        points: Input point cloud tensor of shape (B, N, D) where B is the batch size,
               N is the number of points and D is the dimension of each point
        n_samples: Number of points to sample

    Returns:
        torch.Tensor: Indices of sampled points with shape (B, max(n_samples, N))
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


def group_knn(
    pnts: torch.Tensor,
    cntrds: torch.Tensor,
    feats: torch.Tensor,
    k: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Group k-nearest neighbours for each centroid point.
    Args:
        pnts: Tensor of Shape (B, N, 3) containing Points
        cntrds: Tensor of Shape (B, n_points, 3) containing Centroids
        feats: Tensor of Shape (B, N, 3?) containing Features Descriptions of th Points
        k: Int of Maximum Number of Features to Gather
    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - Tensor of Neighbour Point (B, n_samples, k, 3)
            - Tensor  of Neighbour Point Features (B, n_samples, k, D)
    """
    device = pnts.device
    B, N, D = pnts.size()
    n_samples = cntrds.size(1)

    # Apply k-Nearest Neighbours
    def knn(support: torch.Tensor, query: torch.Tensor, k: int) -> torch.Tensor:
        """k-Nearest Neighbours
        Args:
            support: Tensor of shape (B, n_samples, C) containing Centroids
            query: Tensor of shape (B, N, C) containing Points
        Returns:
            Neighbour Indice Ints (B, n_samples, k).
        """
        dist = torch.cdist(support, query)  # (B, n_samples, k)
        idx = dist.topk(k=k, dim=-1, largest=False, sorted=True).indices # (B, n_samples, k)
        return idx.int()

    knn_idx = knn(cntrds, pnts, k)  # (B, n_samples, k)

    # Gather Points and Features
    batch_idx = (
        torch.arange(B, device=device)
        .view(B, 1, 1)
        .expand(-1, n_samples, k)
    ) # (B, n_samples, k)
    grouped_pnts = pnts[batch_idx, knn_idx]
    grouped_feats = feats[batch_idx, knn_idx]

    return grouped_pnts, grouped_feats


class P3Embed(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        sample_ratio: float = 0.25,
        scale: int = 4,
        k: int = 32,
        layers: int = 4,
        embed_dim: int = 256,
        **kwargs,
    ):
        super().__init__()
        self.sample_ratio = sample_ratio
        self.k = k

        self.sample_fn = farthest_point_sampling
        self.grouper = group_knn

        stages = int(math.log(1 / sample_ratio, scale))
        embed_dim = int(embed_dim // 2 ** (stages - 1))
        self.convs = nn.ModuleList()
        self.channel_list = [in_channels]
        for _ in range(int(stages)):
            channels = (
                [in_channels + 3]
                + [embed_dim] * (layers // 2)
                + [embed_dim * 2] * (layers // 2 - 1)
                + [embed_dim]
            )

            conv1, conv2 = nn.Sequential(), nn.Sequential()
            conv1_layers = []
            for i in range(layers // 2):
                last_layer = (i == (layers // 2 - 1))
                conv1_layers += [ nn.Conv2d(channels[i], channels[i+1], 1, bias=last_layer) ]
                if last_layer:
                    conv1_layers += [
                        nn.BatchNorm2d(channels[i+1]),
                        nn.ReLU()
                    ]
            conv1 = nn.Sequential(*conv1_layers)

            channels[layers // 2] *= 2
            conv2_layers = []
            for i in range(layers // 2, layers):
                conv2_layers += [
                    nn.Conv2d(channels[i], channels[i+1], 1, bias=False),
                    nn.BatchNorm2d(channels[i+1]),
                    nn.ReLU()
                ]

            conv2 = nn.Sequential(*conv2_layers)

            self.convs.append(nn.ModuleList([conv1, conv2]))
            self.channel_list.append(embed_dim)
            in_channels = embed_dim
            embed_dim *= 2

        self.pool = lambda x: torch.max(x, dim=-1, keepdim=True)[0]
        self.out_channels = self.channel_list[-1]

    def forward(
        self, p: torch.Tensor, f: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        B, N, _ = p.shape[:3]
        out_p, out_f = [p], [f]
        for convs in self.convs:
            convs = cast(nn.ModuleList, convs)
            cur_pnt, cur_feat = out_p[-1], out_f[-1].transpose(1,2)
            N = int(N // 4)
            cntrd_idx = self.sample_fn(cur_pnt, N).long() # (B, N)
            cntrd_pnt = torch.gather(cur_pnt, 1, cntrd_idx.unsqueeze(-1).expand(-1, -1, 3)) # (B, N, 3)

            dp, fj = self.grouper(cur_pnt, cntrd_pnt, cur_feat, self.k)
            dp = dp.permute(0, 3, 1, 2).contiguous() # (B, 3, n_smaples, k)
            fj = fj.permute(0, 3, 1, 2).contiguous()  # (B, D, n_samples, k)

            fj = torch.cat([dp, fj], dim=1) # # (B, D+3, n_smaples, k)
            fj = convs[0](fj)
            fj = torch.cat(
                [self.pool(fj).expand(-1, -1, -1, self.k), fj], dim=1
            )

            out_f.append(self.pool(convs[1](fj)).squeeze(-1))
            out_p.append(cntrd_pnt)

        return out_p, out_f


class PointViT(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 384,
        pretrained_model: str = "vit_small_patch16_384.augreg_in21k_ft_in1k",
        k_neighbors: int = 16,
        global_features: str = 'max,cls'
    ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.global_features = global_features.split(',')

        self.patch_embed = P3Embed(
            in_channels=in_channels,
            k=k_neighbors
        )
        self.proj = nn.Linear(self.patch_embed.out_channels, self.embed_dim)
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128, bias=True),
            nn.GELU(),
            nn.Linear(128, self.embed_dim)
        )

        self.vit = timm.create_model(pretrained_model, pretrained=True)
        self.vit_blocks: nn.ModuleList = cast(nn.ModuleList, self.vit.blocks)
        self.norm: nn.Module =cast(nn.Module, self.vit.norm)

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, embed_dim))

    def forward(
        self, p: torch.Tensor, x: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x is None:
            x = p.clone().transpose(1, 2).contiguous()
        feats = x

        p_list, x_list = self.patch_embed(p, feats) # (B, 512, 3), (B, 512, 3)
        center_p = p_list[-1] # (16, 512, 3)

        x = self.proj(x_list[-1].transpose(1, 2)) # (B, 512, 384)
        pos_embed = self.pos_embed(center_p) # (16, 512, 384)

        pos_embed = [self.cls_pos.expand(feats.shape[0], -1, -1), pos_embed]
        tokens = [self.cls_token.expand(feats.shape[0], -1, -1), x]

        pos_embed = torch.cat(pos_embed, dim=1)
        feats = torch.cat(tokens, dim=1)

        for blk in self.vit_blocks:
            feats = blk(feats + pos_embed)
        feats = self.norm(feats)

        return p_list, x_list, feats

    def forward_cls_feat(self, p: torch.Tensor, x: Optional[torch.Tensor] = None) -> torch.Tensor:
        _, _, x = self.forward(p, x)
        token_features = x[:, 1:, :]
        cls_feats = []
        for token_type in self.global_features:
            if 'cls' in token_type:
                cls_feats.append(x[:, 0, :])
            if 'max' in token_type:
                cls_feats.append(torch.max(token_features, dim=1, keepdim=False)[0])
        global_features = torch.cat(cls_feats, dim=1)

        return global_features

    def get_trainable_params(self) -> Iterator[nn.Parameter]:
        """Get only trainable parameters for optimizer.

        Returns:
            Iterator over trainable parameters of the underlying model.
        """
        return filter(lambda p: p.requires_grad, self.parameters())

    def print_trainable_params(self) -> None:
        """Print statistics about trainable parameters.

        Displays the total number of parameters, trainable parameters,
        and percentage of trainable parameters in the underlying model.
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Trainable percentage: {100 * trainable_params / total_params:.2f}%")


class ClsHead(nn.Module):
    def __init__(
        self,
        in_channels: int = 768,
        num_classes: int = 15,
        mlps: List[int] = [256, 256],
        dropout: float = 0.5,
        point_dim: int = 2
    ):
        super().__init__()

        self.point_dim = point_dim
        mlps = [in_channels] + mlps + [num_classes]

        layers = []
        for i in range(len(mlps) - 2):
            layers += [
                nn.Linear(mlps[i], mlps[i + 1], True),
                nn.BatchNorm1d(mlps[i + 1]),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]

        layers += [
            nn.Linear(mlps[-2], mlps[-1], True)
        ]
        self.head = nn.Sequential(*layers)


    def forward(self, end_points: torch.Tensor) -> torch.Tensor:
        logits = self.head(end_points)
        return logits


class Pix4Point(nn.Module):
    def __init__(
        self,
        num_classes: int = 15,
        embed_dim: int = 768,
        pretrained_model: str = 'vit_small_patch16_384.augreg_in21k_ft_in1k',
        k_neighbors: int = 16
    ) -> None:
        """Initialize the Pix4Point model.

        Args:
            num_classes: Number of output classes for classification.
            vit_name: Name of the Vision Transformer backbone to use.
            pretrained: Whether to use pretrained weights for the ViT.
            dropout_rate: Dropout rate for regularization.
            n_samples: Number of samples after FPS sampling.
            k_neighbors: Number of nearest neighbors in point grouping.
            embed_dim: Number of embedding dimension
        """
        super().__init__()
        self.model = PointViT(
            pretrained_model=pretrained_model,
            embed_dim=embed_dim,
            k_neighbors=k_neighbors
        )

        self.cls_head = ClsHead(
            in_channels=2*embed_dim,
            num_classes=num_classes
        )


    def get_trainable_params(self) -> Iterator[nn.Parameter]:
        """Get only trainable parameters for optimizer.

        Returns:
            Iterator over trainable parameters of the underlying model.
        """
        return filter(lambda p: p.requires_grad, self.model.parameters())

    def print_trainable_params(self) -> None:
        """Print Statistics about Trainable Parameters.

        Displays the total number of parameters, trainable parameters,
        and percentage of trainable parameters in the underlying model.
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Trainable percentage: {100 * trainable_params / total_params:.2f}%")

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """Forward Pass

        Args:
            points: Tensor of shape (B, N, 3) containing raw point clouds.

        Returns:
            Classification logits of shape (B, num_classes).
        """

        # Forward through the Model
        global_features = self.model.forward_cls_feat(points)
        logits = self.cls_head(global_features)

        return logits
