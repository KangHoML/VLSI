import torch
import torch.nn as nn

class VGGBlock(nn.Module):
    def __init__(self, num_conv, out_channels, bn=False):
        super().__init__()

        layers = []
        
        for _ in range(num_conv):
            layers.append(nn.LazyConv2d(out_channels, kernel_size=3, padding=1))
            if bn:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU())

        layers.append(
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.blk = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.blk(x)

class VGG(nn.Module):
    cfgs = {
        'VGG11': ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512)),
        'VGG13': ((2, 64), (2, 128), (2, 256), (2, 512), (2, 512)),
        'VGG16': ((2, 64), (2, 128), (3, 256), (3, 512), (3, 512)),
        'VGG19': ((2, 64), (2, 128), (4, 256), (4, 512), (4, 512))
    }

    def __init__(self, num_classes=10, dropout=0.5, cfg='VGG16', bn=False):
        super().__init__()

        vgg_blks = []
        for (num_conv, out_channels) in self.cfgs[cfg]:
            vgg_blks.append(VGGBlock(num_conv, out_channels, bn))
        self.features = nn.Sequential(*vgg_blks)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(p=dropout),

            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            
            nn.Linear(4096, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        return x
    
if __name__ == "__main__":
    net = VGG()
    random_input = torch.randn(16, 3, 32, 32)
    random_output = net(random_input)
    print(random_output.shape)