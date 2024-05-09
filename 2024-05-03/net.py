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
        
        self.cfgs = [(2, 16), (2, 64), (2,256)]
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        res_blk = []
        in_channels = 16
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
        
        self.cfgs = [(2,256), (2, 64), (2, 16)]

        res_blk = []
        in_channels = 256
        for (num_blk, out_channels) in self.cfgs:
            res_blk.append(BasicBlock(in_channels, out_channels, stride=2, mode="decode"))
            
            for _ in range(1, num_blk):
                res_blk.append(BasicBlock(out_channels, out_channels, stride=1, mode="decode"))
            
            in_channels = out_channels
        self.decode = nn.Sequential(*res_blk)

        self.de_conv = nn.Sequential(
            nn.ConvTranspose2d(16, 3, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.decode(x)
        x = self.de_conv(x)

        return x
    
class CNNAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        
    def forward(self, x):
        latent = self.encoder(x)
        output = self.decoder(latent)

        return latent, output

class PatchEmbed(nn.Module):
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

class TransformerAutoEncoder(nn.Module):
    def __init__(self, img_size=64, patch_size=16, in_chans=3, embed_dim=768, num_heads=12, num_layers=12):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, (img_size // patch_size) ** 2, embed_dim))
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads
            ),
            num_layers=num_layers
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=embed_dim,
                nhead=num_heads
            ),
            num_layers=num_layers
        )
        self.proj = nn.Linear(embed_dim, in_chans * patch_size * patch_size)

    def forward(self, x):
        img_size = 64
        in_chans = 3
        x = self.patch_embed(x)
        x = x + self.pos_embed
        latent = self.encoder(x)
        x = self.decoder(x, latent)
        x = self.proj(x)
        x = x.reshape(x.shape[0], in_chans, img_size, img_size)
        return latent, x
    
if __name__ == "__main__":
    net = CNNAutoEncoder()
    random_input = torch.randn(1, 3, 64, 64)
    latent, random_output = net(random_input)
    print(f"latent shape: {latent.shape}, output shape: {random_output.shape}")
    net = TransformerAutoEncoder()
    random_input = torch.randn(1, 3, 64, 64)
    latent, random_output = net(random_input)
    print(f"latent shape: {latent.shape}, output shape: {random_output.shape}")