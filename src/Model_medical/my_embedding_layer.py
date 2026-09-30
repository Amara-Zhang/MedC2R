import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
# repository-specific imports
from .helpers import PerceiverResampler
from .utils import get_visual_encoder
from .vit_3d import ViT, BoundaryAwareViT
from .transformer_decoder import TransformerDecoder, TransformerDecoderLayer
from .cross_attention import TwoWayTransformer
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class MyEmbedding(nn.Module):

    def __init__(self,
                 pretrained_visual_encoder=None,
                 pretrained_adapter=None,
                 num_embeddings=32000,
                 embedding_dim=4096,
                 perceiver_num=32,
                 vis_dim=768,
                 patch_size=32,
                 frame_patch_size=4,
                 seg_channel=256,
                 region_token_len=33):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(
            torch.randn((num_embeddings, embedding_dim)), requires_grad=True
        )
        self.image_token_weight = nn.Parameter(torch.randn((2, embedding_dim)), requires_grad=True)
        self.region_token_weight = nn.Parameter(torch.randn((2, embedding_dim)), requires_grad=True)
        self.patch_size = patch_size
        self.frame_patch_size = frame_patch_size
        self.seg_channel = seg_channel
        self.region_token_len = region_token_len
        self.vision_encoder = ViT(
            image_size=512,
            frames=512,
            image_patch_size=patch_size,
            frame_patch_size=frame_patch_size,
            dim=vis_dim,
            depth=12,
            heads=8,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1
        )
        self.mask_encoder = BoundaryAwareViT(
            image_size=256,
            frames=64,
            image_patch_size=patch_size,
            frame_patch_size=16,
            dim=255,
            depth=3,
            heads=8,
            mlp_dim=512,
            channels=1,
            dropout=0.1,
            emb_dropout=0.1
        )
        if pretrained_visual_encoder is not None:
            vit3d_ckpt = torch.load(pretrained_visual_encoder, map_location='cpu')
            self.vision_encoder.load_state_dict(vit3d_ckpt, strict=True)
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        self.vis_dim = vis_dim
        self.perceiver = PerceiverResampler(dim=self.vis_dim, num_latents=perceiver_num)
        if pretrained_adapter is not None:
            state_dict = torch.load(pretrained_adapter, map_location='cpu')
            self.perceiver.load_state_dict(state_dict['perceiver'])
        self.fc = nn.Linear(self.vis_dim, self.embedding_dim)
        self.mask_fc = nn.Linear(255, self.embedding_dim)
        self.region_d = 512
        self.region_num_heads = 8
        self.region_proj = nn.Linear(self.vis_dim, self.region_d)
        self.region_unproj = nn.Linear(self.region_d, self.embedding_dim)
        encoder_layer = TransformerEncoderLayer(d_model=self.region_d,
                                                nhead=self.region_num_heads,
                                                dim_feedforward=self.region_d * 4,
                                                dropout=0.1,
                                                batch_first=True)
        self.region_transformer = TransformerEncoder(encoder_layer, num_layers=2)
        # region-global cross attention: region (queries) attend to image patches (keys/vals)
        self.region_image_attn = nn.MultiheadAttention(embed_dim=self.region_d,
                                                       num_heads=self.region_num_heads,
                                                       batch_first=True)
        self.mask_geom_proj = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, self.region_d)
        )
        self.gate_proj = nn.Sequential(
            nn.Linear(self.region_d * 2, self.region_d),
            nn.Sigmoid()
        )
        self.region_expand_mlp = nn.Sequential(
            nn.Linear(self.region_d, self.region_d),
            nn.ReLU(),
            nn.Linear(self.region_d, self.region_d)
        )

    def _pool_to_region_tokens(self, tokens, target_len):
        B, T, D = tokens.shape
        if T == target_len:
            return tokens
        out = F.adaptive_avg_pool1d(tokens.transpose(1, 2), target_len).transpose(1, 2)
        return out

    def forward(self, vision_x, mask_x, text_input, region2areas):
        sample = next(iter(vision_x.values()))
        B, S, C, H, W, D = sample.shape
        vision_temp = vision_x['image']  # [B, S, c, h, w, d]
        vision_temp = rearrange(vision_temp, "b S c h w d -> (b S) c h w d")
        vision_temp, pos_embedding = self.vision_encoder(vision_temp)
        vision_temp = rearrange(vision_temp, "(b s) v d -> b s v d", b=B, s=S)
        vision_temp = vision_temp.unsqueeze(2)
        vision_temp = self.perceiver(vision_temp)
        n = vision_temp.shape[2]
        vision_temp = rearrange(vision_temp, "b s n d -> (b s n) d")
        vision_temp = rearrange(vision_temp, "(b T) d -> b T d", b=B, T=n * S)
        image_embedding = vision_temp  # [B, N_i, vis_dim] patch-level global features
        del vision_x['image']
        region_embeddings = vision_x
        mask_embeddings = {}
        region_tokens_list = []  # list of [B,1,region_d]
        mask_geom_list = []      # list of [B,1,region_d]
        region_keys = list(region_embeddings.keys())
        for key in region_keys:
            vision_temp = region_embeddings[key]
            vision_temp = rearrange(vision_temp, "b S c h w d -> (b S) c h w d")
            vision_temp, _ = self.vision_encoder(vision_temp)  # -> (B*S, num_patches, vis_dim)
            vision_temp = rearrange(vision_temp, "(b s) v d -> b s v d", b=B, s=S)
            vision_temp = vision_temp.unsqueeze(2)
            vision_temp = self.perceiver(vision_temp)
            n = vision_temp.shape[2]
            vision_temp = rearrange(vision_temp, "b s n d -> (b s n) d")
            vision_temp = rearrange(vision_temp, "(b T) d -> b T d", b=B, T=n * S)  # [B, T_patch, vis_dim]
            region_embeddings[key] = vision_temp  # [B, T_patch, vis_dim]
            mask_out, _ = self.mask_encoder(x=mask_x[key], mask=mask_x[key])  # [B*S, T_mask, 255]
            mask_out = rearrange(mask_out, "(b s) t d -> b s t d", b=B, s=S)
            mask_out = rearrange(mask_out, "b s t d -> b (s t) d")  # [B, T_mask_combined, 255]
            mask_flat = mask_x[key].view(B, -1)
            area = mask_flat.sum(dim=1, keepdim=True)  # [B,1]
            mean_int = mask_flat.mean(dim=1, keepdim=True)  # [B,1]
            total_pixels = mask_flat.shape[1]
            area_ratio = area / (total_pixels + 1e-6)  # [B,1]
            placeholder = torch.zeros_like(area_ratio)  # [B,1]
            geom = torch.cat([area_ratio, mean_int, area / (total_pixels + 1e-6), placeholder], dim=1)  # [B,4]
            mask_geom_vec = self.mask_geom_proj(geom)  # [B, region_d]
            mask_geom_vec = mask_geom_vec.unsqueeze(1)  # [B,1,region_d]
            region_lowd = self.region_proj(vision_temp)  # [B, T_patch, region_d]
            region_token = region_lowd.mean(dim=1, keepdim=True)  # [B,1,region_d]
            region_tokens_list.append(region_token)  # list of [B,1,region_d]
            mask_geom_list.append(mask_geom_vec)    # list of [B,1,region_d]
        #     mask_geom_list: list of [B,1,region_d] for each region
        if len(region_tokens_list) > 0:
            all_region_tokens = torch.cat(region_tokens_list, dim=1)  # [B, R, region_d]
            all_mask_geom = torch.cat(mask_geom_list, dim=1)         # [B, R, region_d]
            num_regions = all_region_tokens.shape[1]
            # -------------------- region-global cross-attention --------------------
            image_lowd = self.region_proj(image_embedding)  # [B, N_i, region_d]
            attn_out, _ = self.region_image_attn(query=all_region_tokens, key=image_lowd, value=image_lowd)
            all_region_tokens = all_region_tokens + attn_out  # [B, R, region_d]
            all_region_tokens = self.region_transformer(all_region_tokens)  # [B, R, region_d]
            concat_rm = torch.cat([all_region_tokens, all_mask_geom], dim=-1)  # [B, R, 2*region_d]
            gates = self.gate_proj(concat_rm)
            fused_regions = gates * all_region_tokens + (1.0 - gates) * all_mask_geom  # [B, R, region_d]
            fused_regions = self.region_expand_mlp(fused_regions)  # [B, R, region_d]
            Bf, Rf, Dlow = fused_regions.shape
            # repeat -> [B, R, region_token_len, region_d]
            expanded = fused_regions.unsqueeze(2).repeat(1, 1, self.region_token_len, 1)
            # reshape -> [B, R*region_token_len, region_d]
            expanded = expanded.view(Bf, Rf * self.region_token_len, Dlow)
            expanded_highd = self.region_unproj(expanded)  # [B, R*region_len, embedding_dim]
            mask_highd = self.region_unproj(all_mask_geom)  # [B, R, embedding_dim]
            start = 0
            for i, key in enumerate(region_keys):
                end = start + self.region_token_len
                # region_part: [B, region_token_len, embedding_dim]
                region_part = expanded_highd[:, start:end, :]
                # mask token for region i: [B, 1, embedding_dim]
                mask_token_i = mask_highd[:, i:i+1, :]
                region_embeddings[key] = torch.cat([region_part, mask_token_i], dim=1)  # [B, region_token_len+1, embedding_dim]
                start = end
        else:
            pass
        image_embedding = self.fc(image_embedding)  # [B, N_i, embedding_dim]
        max_region = len(region_keys)
        slot_per_region = self.region_token_len + 1  # e.g. 33 + 1 mask = 34
        vision_region_embedding = torch.zeros((B, slot_per_region * max_region, self.embedding_dim),
                                              device=image_embedding.device)
        for i in range(B):
            for j in range(len(region2areas[i])):
                region = region2areas[i][j]
                vision_region_embedding[i, j * slot_per_region:(j + 1) * slot_per_region, :] = region_embeddings[region][i, :, :]
        embedding_weight = torch.cat([self.weight, self.image_token_weight, self.region_token_weight], dim=0)  # [num_embeddings+2, embedding_dim]
        embedding_weight = embedding_weight.unsqueeze(0).repeat(B, 1, 1)  # [B, num_embeddings+2, embedding_dim]
        embedding_weight = torch.cat([embedding_weight, image_embedding, vision_region_embedding], dim=1)  # [B, total_seq_len, embedding_dim]
        text_input_onehot = F.one_hot(text_input, embedding_weight.shape[1]).to(embedding_weight.dtype).to(text_input.device)  # [B, N, total_seq_len]
        out_put = torch.matmul(text_input_onehot.float(), embedding_weight)  # [B, N, embedding_dim]
        return out_put
