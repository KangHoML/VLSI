import thop
import torch
import torch.nn as nn
from torchsummary import summary

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)

class Seperable(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        padding = (kernel_size - 1) // 2
            
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )

        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class CustomBlock(nn.Module):
    def __init__(self, num_conv, in_channels, out_channels, kernel_size):
        super().__init__()

        layers = []
        
        for i in range(num_conv):
            if i == 0:
                layers.append(Seperable(in_channels, out_channels, kernel_size=kernel_size, stride=2))
            else:
                layers.append(Seperable(out_channels, out_channels, kernel_size=kernel_size, stride=1))

        self.blk = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.blk(x)

# Based on MobileNet(alpha = 1)
class CustomNet(nn.Module):
    cfg = ((2, 128), (2, 256), (6, 512), (1, 1024), (1, 1024))
    
    def __init__(self, kernel_size=3, num_classes=10):
        super().__init__()

        self.conv = nn.Sequential(
            ConvBlock(3, 32, kernel_size=kernel_size, stride=2),
            Seperable(32, 64, kernel_size=kernel_size, stride=1)
        )
        
        custom_blk = []
        in_channels = 64
        for (num_conv, out_channels) in self.cfg:
            custom_blk.append(CustomBlock(num_conv, in_channels, out_channels, kernel_size=kernel_size))
            in_channels = out_channels
        self.features = nn.Sequential(*custom_blk)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = self.features(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x
    
if __name__ == "__main__":
    net = CustomNet(kernel_size=3)
    random_input = torch.randn(16, 3, 32, 32)
    random_output = net(random_input)
    flops, params = thop.profile(net, inputs=(random_input, ))

    print(f"Output Size: {random_output.shape}")
    print(f"Computation (GFLOPs): {flops}, Params (Millions): {params}")
    