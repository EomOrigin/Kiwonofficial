import torch
import torch.nn as nn
import timm
from utils.region_attention import RegionAttention

class ViTWithRegionBias(nn.Module):
    """
    ViT_small_patch16_224 기반 + landmark 기반 patch-weight 적용
    고정/학습 가중치 선택 가능
    """
    def __init__(self, num_classes=2, default_weight=1.0, enhanced_weight=1.2,
                 patch_size=16, learnable=True):
        super().__init__()
        # 1) Pretrained ViT 로드
        self.vit = timm.create_model('vit_small_patch16_224', pretrained=True)
        in_feats = self.vit.head.in_features
        self.vit.head = nn.Linear(in_feats, num_classes)

        # 2) RegionAttention 모듈 연결
        self.region_attn = RegionAttention(
            image_size=(224, 224),
            patch_size=patch_size,
            default_weight=default_weight,
            enhanced_weight=enhanced_weight,
            learnable=learnable
        )

    def forward(self, x, landmarks_list):
        """
        x: (B, 3, 224, 224)
        landmarks_list: 배치마다 [(x, y), ...] 형태의 리스트
        """
        B = x.size(0)

        # --- Patch Embedding ---
        patch_tokens = self.vit.patch_embed(x)              # (B, N, D)
        cls_token    = self.vit.cls_token.expand(B, -1, -1) # (B, 1, D)
        x = torch.cat((cls_token, patch_tokens), dim=1)     # (B, N+1, D)
        x = x + self.vit.pos_embed.to(x.device).to(x.dtype)
        x = self.vit.pos_drop(x)

        # --- Region Weight 적용 ---
        cls_tok, pats = x[:, :1], x[:, 1:]                  # (B,1,D), (B,N,D)
        weight_maps = [self.region_attn(lm) for lm in landmarks_list]
        w = torch.stack(weight_maps, dim=0).to(pats.device).to(pats.dtype)  # (B, N)
        w = w.unsqueeze(2)                                  # (B, N, 1)
        pats = pats * w

        # --- Transformer & Head ---
        x = torch.cat((cls_tok, pats), dim=1)
        for blk in self.vit.blocks:
            x = blk(x)
        x = self.vit.norm(x)

        return self.vit.head(x[:, 0])   # (B, num_classes)
