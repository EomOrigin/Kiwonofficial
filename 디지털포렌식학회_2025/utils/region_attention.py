import torch
import torch.nn as nn

class RegionAttention(nn.Module):
    """
    얼굴 landmark 좌표 기반으로 ViT 패치 크기 단위로
    개별 학습 가능한 바이어스를 주는 모듈
    """
    def __init__(self, image_size=(224, 224), patch_size=16,
                 default_weight=1.0, learnable=True):
        super().__init__()
        H, W = image_size
        self.grid_rows = H // patch_size
        self.grid_cols = W // patch_size
        self.default_weight = default_weight
        N = self.grid_rows * self.grid_cols

        if learnable:
            self.enhanced_weight = nn.Parameter(torch.ones(N, dtype=torch.float32))
        else:
            self.register_buffer('enhanced_weight', torch.ones(N, dtype=torch.float32))

        self.patch_h = H / self.grid_rows
        self.patch_w = W / self.grid_cols

    def forward(self, landmarks):
        device = self.enhanced_weight.device
        mask = torch.zeros((self.grid_rows, self.grid_cols),
                           device=device, dtype=torch.float32)

        for x, y in landmarks:
            r = min(int(y // self.patch_h), self.grid_rows - 1)
            c = min(int(x // self.patch_w), self.grid_cols - 1)
            mask[r, c] = 1.0

        mask_flat = mask.flatten()
        weights   = self.default_weight + (self.enhanced_weight - self.default_weight) * mask_flat
        return weights