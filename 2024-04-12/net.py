import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

class CNBlock(nn.Module):
    '''
    Implementation:
        1. Depthwise Convolution
        2. Permute to (N, H, W, C)
        3. LayerNorm (channels_last)
        4. Linear -> GELU -> Linear
        5. Permute back

    Args:
        in_channels (int): Input Channel
    '''
    def __init__(self, in_channels):
        super().__init__()

        # Depthwise Convolution
        self.dwconv = nn.Conv2d(in_channels, in_channels, kernel_size=7, padding=3, groups=in_channels)

        # Layer Normalization
        self.norm = LayerNorm(in_channels, eps=1e-6)
        
        # Inverted BottleNeck
        self.inv_bottleneck = nn.Sequential(
            nn.Linear(in_channels, in_channels * 4),
            nn.GELU(),
            nn.Linear(in_channels * 4, in_channels)
        )

    def forward(self, x):
        shortcut = x

        # Depthwise Convolution
        x = self.dwconv(x)

        # Layer Normalization을 위해 "채널"을 끝으로 보냄 (N, C, H, W) -> (N, H, W, C)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)

        # Inverted BottleNeck
        x = self.inv_bottleneck(x)
        
        # 채널을 다시 배치 다음으로 돌려준 뒤, Residual Connection
        x = x.permute(0, 3, 1, 2)
        x = shortcut + x
    
        return x
    
class ConvNeXt_T(nn.Module):
    # Configuration: (num_blocks, hidden_size)
    default_cfgs = [(3, 96), (3, 192), (9, 384), (3,768)] 

    def __init__(self, in_channels=3, num_classes=100, cfgs=default_cfgs, pretrained=False):
        super().__init__()

        # stem
        self.downsample = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_channels, cfgs[0][1], kernel_size=4, stride=4),
            LayerNorm(cfgs[0][1], eps=1e-6, data_format="channel_first")
        )
        self.downsample.append(stem)

        # down sample layer (stage 사이의 별도의 downsample layer)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(cfgs[i][1], eps=1e-6, data_format="channel_first"),
                    nn.Conv2d(cfgs[i][1], cfgs[i+1][1], kernel_size=2, stride=2),
            )
            self.downsample.append(downsample_layer)

        # 4개의 Feature Resolution Stage
        self.stage = nn.ModuleList()
        for num_blk, hidden_size in cfgs:
            cn_blk = []
            for _ in range(num_blk):
                cn_blk.append(CNBlock(in_channels=hidden_size))
            self.stage.append(nn.Sequential(*cn_blk))        

        # Global Average Pooling : (N, C, H, W) -> (N, C)
        self.norm = nn.LayerNorm(cfgs[-1][1], eps=1e-6) 

        # Classification
        self.head = nn.Linear(cfgs[-1][1], num_classes)

        # 가중치 초기화 및 pretrained 가져오기
        self.apply(self._init_weights)
        if pretrained:
            self._load_pretrained_model()


    def forward(self, x):
        # downsample & stage
        for i in range(4):
            x = self.downsample[i](x)
            x = self.stage[i](x)
        
        # Global Average Pooling
        x = self.norm(x.mean([-2, -1]))

        # Classification
        x = self.head(x)

        return x
    
    def _init_weights(self, m):
        '''
        weight: 평균 0, 표준편차 0.02 값인 정규분포로 샘플링하여 초기화
                이때, 특정 임곗값 밖의 값들을 잘라내고 다시 샘플링
        bias: 0으로 초기화
        '''
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)        
    
    def _load_pretrained_model(self):
        # pretrained_model
        url = "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth"
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        
        # 모델 키 확인 및 로드
        if 'model' in checkpoint:
            checkpoint_model = checkpoint['model']
        else:
            checkpoint_model = checkpoint  # 체크포인트가 바로 state_dict 인 경

        # fine-tuning을 위해 마지막 head 부분 제거
        state_dict = self.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                del checkpoint_model[k]

        self.load_state_dict(checkpoint_model, strict=False)
        

class LayerNorm(nn.Module):
    '''
    nn.LayerNorm의 경우 마지막 인자에 대해서만 정규화 수행
    - (N, H, W, C)인 경우에는 기존 nn.LayerNorm을 통해 채널에 대한 정규화 수행
    - (N, C, H, W)인 경우에는 각 픽셀별 정규화 수행 필요
    '''
    def __init__(self, normalized_shape, eps=1e-6, data_format="channel_last"):
        super().__init__()

        self.w = nn.Parameter(torch.ones(normalized_shape))
        self.b = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

        self.data_format = data_format
        if self.data_format not in ["channel_last", "channel_first"]:
            raise NotImplementedError
        
        # F.layer_norm의 입력이 tuple이기 때문
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channel_last":
            return F.layer_norm(x, self.normalized_shape, self.w, self.b, self.eps)
        
        elif self.data_format == "channel_first":
            mean = x.mean(1, keepdim=True) # 평균
            var = (x - mean).pow(2).mean(1, keepdim=True) # 분산
            x = (x - mean) / torch.sqrt(var + self.eps) # 정규화
            x = self.w[:, None, None] * x + self.b[:, None, None]

            return x

if __name__ == "__main__":
    net = ConvNeXt_T(pretrained=True)

    random_input = torch.randn(16, 3, 32, 32)
    random_output = net(random_input)
    print(f"output_shape: {random_output.shape}")

