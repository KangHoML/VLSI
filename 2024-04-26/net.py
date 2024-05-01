import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, mode):
        super().__init__()

        if mode == 'encode':
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.resize = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, output_padding=stride-1)
            self.conv2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.resize = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=1, stride=stride, output_padding=stride-1)
        
        self.conv = nn.Sequential(
            self.conv1,
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            self.conv2,
            nn.BatchNorm2d(out_channels)
        )

        self.shortcut = nn.Sequential(
            self.resize,
            nn.BatchNorm2d(out_channels)
        )

        self.act = nn.ReLU()

    def forward(self, x):
        identity = x
        x = self.conv(x)

        if x.shape != identity.shape:
            x += self.shortcut(identity)
        else:
            x += identity
        
        x = self.act(x)

        return x

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.cfgs = [(2, 32), (2, 64), (2, 128)]
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        res_blk = []
        in_channels = 32
        for (num_blk, out_channels) in self.cfgs:
            res_blk.append(BasicBlock(in_channels, out_channels, stride=2, mode="encode"))
            
            for _ in range(1, num_blk):
                res_blk.append(BasicBlock(out_channels, out_channels, stride=1, mode="encode"))
            
            in_channels = out_channels
        self.encode = nn.Sequential(*res_blk)

    def forward(self, x):
        x = self.conv(x)
        x = self.encode(x)

        return x

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.cfgs = [(2, 128), (2, 64), (2, 32)]

        res_blk = []
        in_channels = 128
        for (num_blk, out_channels) in self.cfgs:
            res_blk.append(BasicBlock(in_channels, out_channels, stride=2, mode="decode"))
            
            for _ in range(1, num_blk):
                res_blk.append(BasicBlock(out_channels, out_channels, stride=1, mode="decode"))
            
            in_channels = out_channels
        self.decode = nn.Sequential(*res_blk)

        self.de_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 3, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.decode(x)
        x = self.de_conv(x)

        return x
    
class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        
    def forward(self, x):
        latent = self.encoder(x)
        output = self.decoder(latent)

        return latent, output
    
if __name__ == "__main__":
    net = AutoEncoder()

    random_input = torch.randn(16, 3, 224, 224)
    latent, random_output = net(random_input)
    print(f"latent shape: {latent.shape}, output shape: {random_output.shape}")