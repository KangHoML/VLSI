import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        self.conv = nn.Sequential(
            nn.LazyConv2d(out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
            nn.BatchNorm2d(out_channels)
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

class BottleNeck(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.LazyConv2d(out_channels, kernel_size=1, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels*4, kernel_size=1),
            nn.BatchNorm2d(out_channels*4)
        )

        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels*4, kernel_size=1, stride=stride),
            nn.BatchNorm2d(out_channels*4)
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

class ResNet(nn.Module):
    cfgs = {
        'ResNet18': ((2, 64), (2, 128), (2, 256), (2, 512)),
        'ResNet34': ((3, 64), (4, 128), (6, 256), (3, 512)),
        'ResNet50': ((3, 64), (4, 128), (6, 256), (3, 512)),
        'ResNet101': ((3, 64), (4, 128), (23, 256), (3, 512)),
        'ResNet152': ((3, 64), (8, 128), (36, 256), (3, 512))
    }

    def __init__(self, num_classes=10, cfg="ResNet18"):
        super().__init__()

        if cfg == 'ResNet18' or cfg == 'ResNet34':
            self.block = BasicBlock
        else:
            self.block = BottleNeck

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        res_blk = []

        in_channels = 64
        for (num_blk, out_channels) in self.cfgs[cfg]:
            if out_channels == 64:
                self.stride = 1
            else:
                self.stride = 2
            
            res_blk.append(self.block(in_channels, out_channels, stride=self.stride))
            for _ in range(1, num_blk):
                res_blk.append(self.block(out_channels, out_channels, stride=1))
            
            if self.block == BasicBlock:
                in_channels = out_channels
            else:
                in_channels = out_channels * 4
        
        self.features = nn.Sequential(*res_blk)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(in_channels, num_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.features(x)
        x = self.avgpool(x)

        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x

if __name__ == "__main__":
    net = ResNet(cfg="ResNet50")
    random_input = torch.randn(16, 3, 32, 32)
    random_output = net(random_input)
    print(random_output.shape)