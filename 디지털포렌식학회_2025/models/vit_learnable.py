# models/vit_learnable.py

import torch
import torch.nn as nn
import timm
from utils.region_attention_learnable import RegionAttention
from utils.landmark_indices import EYE_IDXS, NOSE_IDXS, MOUTH_IDXS

class ViTWithRegionBias(nn.Module):
    """
    ViT_small_patch16_224 기반 + 부위별 분리된 RegionAttention 통합
    """
    def __init__(self,
                 num_classes=2,
                 default_weight=1.0,
                 enhanced_weight=1.2,
                 patch_size=16,
                 learnable=True,
                 scalar_learnable=False):
        super().__init__()
        # 1) Pretrained ViT
        self.vit = timm.create_model('vit_small_patch16_224', pretrained=True)
        in_feats = self.vit.head.in_features
        self.vit.head = nn.Linear(in_feats, num_classes)

        # 2) RegionAttention (Multi-region)
        self.region_attn = RegionAttention(
            image_size=(224, 224),
            patch_size=patch_size,
            default_weight=default_weight,
            enhanced_weight=enhanced_weight,
            learnable=learnable,
            scalar_learnable=scalar_learnable
        )

    def forward(self, x, landmarks_list):
        B = x.size(0)
        # Patch embedding + CLS token
        patch_tokens = self.vit.patch_embed(x)              # (B, N, D)
        cls_token    = self.vit.cls_token.expand(B, -1, -1) # (B,1,D)
        x_tokens     = torch.cat((cls_token, patch_tokens), dim=1)
        x_tokens     = x_tokens + self.vit.pos_embed.to(x_tokens.device).to(x_tokens.dtype)
        x_tokens     = self.vit.pos_drop(x_tokens)

        cls_tok, pats = x_tokens[:, :1], x_tokens[:, 1:]    # (B,1,D), (B,N,D)

        # Compute per-sample weight maps
        weight_vecs = []
        for lm in landmarks_list:
            # lm: list of (x,y) tuples in FACEMESH order
            eye_lms   = [(x, y) for idx, (x, y) in enumerate(lm) if idx in EYE_IDXS]
            nose_lms  = [(x, y) for idx, (x, y) in enumerate(lm) if idx in NOSE_IDXS]
            mouth_lms = [(x, y) for idx, (x, y) in enumerate(lm) if idx in MOUTH_IDXS]

            wm = self.region_attn(eye_lms, nose_lms, mouth_lms)  # (N,)
            weight_vecs.append(wm)

        w = torch.stack(weight_vecs, dim=0).unsqueeze(2)  # (B, N, 1)
        w = w.to(pats.device).to(pats.dtype)
        pats = pats * w

        # Transformer blocks + classification head
        x_tokens = torch.cat((cls_tok, pats), dim=1)
        for blk in self.vit.blocks:
            x_tokens = blk(x_tokens)
        x_tokens = self.vit.norm(x_tokens)

        return self.vit.head(x_tokens[:, 0])
