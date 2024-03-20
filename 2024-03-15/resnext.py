import torch
import torch.nn as nn

class BottleNeck(nn.Module):
    def __init__(self, in_channels, out_channels, stride, cardinality):
        super().__init__()

        self.conv = nn.Sequential(
            nn.LazyConv2d(out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels * 2, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels*2)
        )

        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * 2, kernel_size=1, stride=stride),
            nn.BatchNorm2d(out_channels*2)
        )

        self.act = nn.ReLU()
    
    def forward(self, x):
        shortcut = x
        x = self.conv(x)

        if x.shape != shortcut.shape:
            x += self.downsample(shortcut)
        else:
            x += shortcut
        
        x = self.act(x)

        return x

class ResNeXt(nn.Module):
    cfgs = {
        'ResNeXt50': ((3, 128), (4, 256), (6, 512), (3, 1024)),
        'ResNeXt101': ((3, 128), (4, 256), (23, 512), (3, 1024)),
        'ResNeXt152': ((3, 128), (8, 256), (36, 512), (3, 1024))
    }

    def __init__(self, cardinality=32, num_classes=10, cfg='ResNeXt50'):
        super().__init__()
        self.cardinality = cardinality
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        res_blk = []
        
        in_channels = 64
        for num_blk, out_channels in self.cfgs[cfg]:
            if out_channels == 128:
                self.stride = 1
            else:
                self.stride = 2
    
            res_blk.append(BottleNeck(in_channels, out_channels, stride=self.stride, cardinality=self.cardinality))
            for _ in range(1, num_blk):
                res_blk.append(BottleNeck(out_channels, out_channels, stride=1, cardinality=self.cardinality))
            
            in_channels = out_channels * 2

        self.features = nn.Sequential(*res_blk)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pool(x)

        x = self.features(x)
        x = self.avg_pool(x)

        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x

if __name__ == "__main__":
    net = ResNeXt(cfg="ResNeXt50")
    random_input = torch.randn(16, 3, 32, 32)
    random_output = net(random_input)
    print(random_output.shape)

