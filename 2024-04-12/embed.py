import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

import torch
import torch.nn as nn

class FixedEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim):
        super(FixedEmbedding, self).__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        # Conv2d를 사용하여 이미지를 패치로 변환하고 임베딩 차원으로 매핑합니다.
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        # 위치 인코딩 초기화: 각 패치에 고정된 위치 인코딩을 할당합니다.
        self.position_embeddings = nn.Parameter(self.create_positional_embeddings(self.num_patches, embed_dim), requires_grad=False)

    def create_positional_embeddings(self, num_patches, embed_dim):
        # 위치 인코딩 계산: 패치 인덱스에 따른 사인, 코사인 함수 적용
        position = torch.arange(0, num_patches, dtype=torch.float).unsqueeze(1)
        div_term = torch.pow(10000, torch.arange(0, embed_dim, 2).float() / embed_dim)
        pe = torch.zeros(num_patches, embed_dim)
        pe[:, 0::2] = torch.sin(position / div_term)
        pe[:, 1::2] = torch.cos(position / div_term)
        return pe

    def forward(self, x):
        # 패치 추출 및 위치 인코딩 적용
        x = self.proj(x)  # [B, E, H', W']
        x = x.flatten(2)  # [B, E, N]
        x = x.transpose(1, 2)  # [B, N, E]
        x = x + self.position_embeddings  # Add fixed positional embeddings
        return x


class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        assert img_size % patch_size == 0,\
            f"img_size({img_size} is not divisable by patch_size({patch_size}))"

        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x

class LearnableEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_chans=3, embed_dim=768):
        
        super().__init__()
        self.num_tokens = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        self.project = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.num_tokens += 1
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, self.embed_dim))

        # init cls token and pos_embed -> refer timm vision transformer
        # https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py#L391
        nn.init.normal_(self.cls_token, std=1e-6)
        trunc_normal_(self.pos_embed, std=.02)

    def forward(self, x):
        B, C, H, W = x.shape
        embedding = self.project(x)
        z = embedding.view(B, self.embed_dim, -1).permute(0, 2, 1)  # BCHW -> BNC

        # concat cls token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        z = torch.cat([cls_tokens, z], dim=1)

        # add position embedding
        z = z + self.pos_embed
        return z
    
if __name__ == '__main__':
    img = torch.randn(1, 3, 224, 224)
    fixed_embedding = FixedEmbedding(in_chans=3, embed_dim=786, img_size=224, patch_size=16)
    fixed_embedded = fixed_embedding(img)

    patch_embedding = PatchEmbedding(in_chans=3, embed_dim=786, img_size=224, patch_size=16)
    patch_emdedded = patch_embedding(img)

    learn_embedding = LearnableEmbedding(in_chans=3, embed_dim=786, img_size=224, patch_size=16)
    learn_emdedded = learn_embedding(img)
    
    print(f'fixed: {fixed_embedded.size()}, patch: {patch_emdedded.size()}, learnable: {learn_emdedded.size()}')