# utils/region_attention_learnable.py

import torch
import torch.nn as nn

class RegionAttention(nn.Module):
    """
    얼굴 landmark 좌표 기반으로 ViT 패치 단위에서 눈, 코, 입별로
    학습 가능한 가중치를 분리하여 적용하는 모듈

    Modes:
      - fixed_scalar:     각 region에 동일한 고정 스칼라
      - learnable_scalar: 각 region마다 하나의 스칼라 학습
      - learnable_vector: 각 region마다 패치 수만큼 가중치 벡터 학습
    """
    def __init__(self,
                 image_size=(224, 224),
                 patch_size=16,
                 default_weight=1.0,
                 enhanced_weight=1.2,
                 learnable=True,
                 scalar_learnable=False):
        super().__init__()
        H, W = image_size
        R, C = H // patch_size, W // patch_size
        N = R * C

        self.default_weight = default_weight
        self.grid_rows, self.grid_cols = R, C
        self.patch_h = H / R
        self.patch_w = W / C

        # Initialize per-region weights
        if not learnable and not scalar_learnable:
            # 고정 스칼라
            self.register_buffer("eye_weight", torch.tensor([enhanced_weight]))
            self.register_buffer("nose_weight", torch.tensor([enhanced_weight]))
            self.register_buffer("mouth_weight", torch.tensor([enhanced_weight]))
            self.mode = "fixed_scalar"
        elif scalar_learnable:
            # 학습 가능한 스칼라
            self.eye_weight   = nn.Parameter(torch.tensor(enhanced_weight, dtype=torch.float32))
            self.nose_weight  = nn.Parameter(torch.tensor(enhanced_weight, dtype=torch.float32))
            self.mouth_weight = nn.Parameter(torch.tensor(enhanced_weight, dtype=torch.float32))
            self.mode = "learnable_scalar"
        else:
            # 학습 가능한 벡터
            self.eye_weight   = nn.Parameter(torch.full((N,), enhanced_weight, dtype=torch.float32))
            self.nose_weight  = nn.Parameter(torch.full((N,), enhanced_weight, dtype=torch.float32))
            self.mouth_weight = nn.Parameter(torch.full((N,), enhanced_weight, dtype=torch.float32))
            self.mode = "learnable_vector"

    def forward(self, eye_landmarks, nose_landmarks, mouth_landmarks):
        device = self.eye_weight.device
        mask_eye   = torch.zeros((self.grid_rows, self.grid_cols), device=device)
        # nonzero = mask_eye.sum().item()
        # print(f"    ▶ mask_eye nonzero count: {nonzero}")
        mask_nose  = torch.zeros_like(mask_eye)
        mask_mouth = torch.zeros_like(mask_eye)

        # Populate masks
        for x, y in eye_landmarks:
            r = min(int(y // self.patch_h), self.grid_rows - 1)
            c = min(int(x // self.patch_w), self.grid_cols - 1)
            mask_eye[r, c] = 1.0
        for x, y in nose_landmarks:
            r = min(int(y // self.patch_h), self.grid_rows - 1)
            c = min(int(x // self.patch_w), self.grid_cols - 1)
            mask_nose[r, c] = 1.0
        for x, y in mouth_landmarks:
            r = min(int(y // self.patch_h), self.grid_rows - 1)
            c = min(int(x // self.patch_w), self.grid_cols - 1)
            mask_mouth[r, c] = 1.0

        flat_eye   = mask_eye.flatten()
        flat_nose  = mask_nose.flatten()
        flat_mouth = mask_mouth.flatten()

        # Compute per-region contributions
        w_eye   = (self.eye_weight   - self.default_weight) * flat_eye
        w_nose  = (self.nose_weight  - self.default_weight) * flat_nose
        w_mouth = (self.mouth_weight - self.default_weight) * flat_mouth

        # Final weight map
        weight_map = self.default_weight + w_eye + w_nose + w_mouth  # shape (N,)
        return weight_map