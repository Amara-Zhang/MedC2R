import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange

from .position_encoding import PositionEmbeddingLearned3d


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(
                nn.Linear(inner_dim, dim),
                nn.Dropout(dropout),
            )
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads),
            qkv,
        )

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            Attention(
                                dim,
                                heads=heads,
                                dim_head=dim_head,
                                dropout=dropout,
                            ),
                        ),
                        PreNorm(
                            dim,
                            FeedForward(dim, mlp_dim, dropout=dropout),
                        ),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ViT(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        image_patch_size,
        frames,
        frame_patch_size,
        dim,
        depth,
        heads,
        mlp_dim,
        pool="cls",
        channels=3,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(image_patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, (
            "Image dimensions must be divisible by the patch size."
        )
        assert frames % frame_patch_size == 0, (
            "Frames must be divisible by frame patch size."
        )

        self.patch_height = patch_height
        self.patch_width = patch_width
        self.frame_patch_size = frame_patch_size

        patch_dim = channels * patch_height * patch_width * frame_patch_size
        assert pool in {"cls", "mean"}, (
            "pool type must be either cls (cls token) or mean (mean pooling)"
        )

        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) (f pf) -> b (h w f) (p1 p2 pf c)",
                p1=patch_height,
                p2=patch_width,
                pf=frame_patch_size,
            ),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = PositionEmbeddingLearned3d(
            dim // 3,
            image_height // patch_height,
            image_width // patch_width,
            frames // frame_patch_size,
        )
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
        )

    def forward(self, video):
        B, C, H, W, D = video.shape
        x = self.to_patch_embedding(video)

        pos = self.pos_embedding(
            B,
            H // self.patch_height,
            W // self.patch_width,
            D // self.frame_patch_size,
            x,
        )
        x = x + pos
        x = self.dropout(x)
        x = self.transformer(x)

        return x, pos


class BoundaryAwareAttention(nn.Module):
    def __init__(
        self,
        dim,
        H,
        W,
        D,
        heads=8,
        dim_head=64,
        dropout=0.0,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.grid_H = H
        self.grid_W = W
        self.grid_D = D
        self.num_patches = H * W * D

        self.boundary_conv = nn.Sequential(
            nn.Conv3d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 1, 3, padding=1),
            nn.Sigmoid(),
        )

        self.to_out = (
            nn.Sequential(
                nn.Linear(inner_dim, dim),
                nn.Dropout(dropout),
            )
            if project_out
            else nn.Identity()
        )

    def forward(self, x, mask=None):
        B, N, _ = x.shape
        assert N == self.num_patches, (
            f"Number of tokens N({N}) != expected num_patches ({self.num_patches})"
        )

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads),
            qkv,
        )

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        if mask is not None:
            if mask.dim() == 3:
                mask = mask.squeeze(1)
            mask = mask.to(device=dots.device, dtype=dots.dtype)

            boundary_bias = self.compute_boundary_bias(mask, B, N)
            dots = dots + boundary_bias.unsqueeze(1)

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)

    def compute_boundary_bias(self, patch_mask, B, N):
        H, W, D = self.grid_H, self.grid_W, self.grid_D
        assert N == H * W * D, "N must equal H*W*D"

        boundary_bias = torch.zeros(
            B,
            N,
            N,
            device=patch_mask.device,
            dtype=patch_mask.dtype,
        )

        for b in range(B):
            mask_3d = patch_mask[b].view(1, 1, H, W, D)

            dilated = F.max_pool3d(
                mask_3d,
                kernel_size=3,
                stride=1,
                padding=1,
            )
            eroded = -F.max_pool3d(
                -mask_3d,
                kernel_size=3,
                stride=1,
                padding=1,
            )

            boundary = torch.clamp(dilated - eroded, 0, 1)
            boundary_flat = boundary.view(-1)
            boundary_bias[b] = (
                boundary_flat.unsqueeze(1) @ boundary_flat.unsqueeze(0)
            )

        return boundary_bias * 2.0


class BoundaryAwareTransformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        H,
        W,
        D,
        dropout=0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            attn = BoundaryAwareAttention(
                dim,
                H=H,
                W=W,
                D=D,
                heads=heads,
                dim_head=dim_head,
                dropout=dropout,
            )
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(dim, attn),
                        PreNorm(
                            dim,
                            FeedForward(dim, mlp_dim, dropout=dropout),
                        ),
                    ]
                )
            )

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask) + x
            x = ff(x) + x
        return x


class BoundaryAwareViT(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        frames,
        image_patch_size,
        frame_patch_size,
        dim,
        depth,
        heads,
        mlp_dim,
        channels=3,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()

        image_height, image_width = image_size, image_size
        patch_h, patch_w = image_patch_size, image_patch_size
        patch_d = frame_patch_size

        assert (
            image_height % patch_h == 0
            and image_width % patch_w == 0
            and frames % patch_d == 0
        ), "image/frame sizes must be divisible by patch sizes."

        self.patch_h = patch_h
        self.patch_w = patch_w
        self.patch_d = patch_d
        self.channels = channels

        self.grid_H = image_height // patch_h
        self.grid_W = image_width // patch_w
        self.grid_D = frames // patch_d
        self.num_patches = self.grid_H * self.grid_W * self.grid_D

        patch_dim = channels * patch_h * patch_w * patch_d
        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) (f p3) -> b (h w f) (p1 p2 p3 c)",
                p1=patch_h,
                p2=patch_w,
                p3=patch_d,
            ),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.num_patches, dim)
        )
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = BoundaryAwareTransformer(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            H=self.grid_H,
            W=self.grid_W,
            D=self.grid_D,
            dropout=dropout,
        )

        self.to_latent = nn.Identity()

    def mask_to_patch_mask(self, mask):
        patches = rearrange(
            mask,
            "b c (h p1) (w p2) (f p3) -> b (h w f) (p1 p2 p3 c)",
            p1=self.patch_h,
            p2=self.patch_w,
            p3=self.patch_d,
        )
        patch_vals = patches.mean(dim=-1)
        patch_mask = (patch_vals > 0.5).float()
        return patch_mask

    def forward(self, x, mask=None):
        x_patches = self.to_patch_embedding(x)

        patch_mask = None
        if mask is not None:
            if mask.dim() == 2 and mask.shape[1] == self.num_patches:
                patch_mask = mask
            else:
                patch_mask = self.mask_to_patch_mask(mask)

        x_patches = x_patches + self.pos_embedding
        x_patches = self.dropout(x_patches)
        x_out = self.transformer(x_patches, mask=patch_mask)

        return x_out, self.pos_embedding
