import torch
import torch.nn as nn

def get_act_func(act_func):
    if act_func == "Tanh":
        return nn.Tanh()
    elif act_func == "ReLU":
        return nn.ReLU()
    elif act_func == "ELU":
        return nn.ELU()
    elif act_func == "SILU":
        return nn.SiLU()
    else:
        raise NotImplementedError(act_func)

class CNNBlock(nn.Module):
    def __init__(self, num_conv, out_feat, act_func):
        super().__init__()

        conv_blk = []

        for _ in range(num_conv):
            conv_blk.append(
                nn.LazyConv2d(out_feat, kernel_size=3, stride=1, padding=1)
            )

            conv_blk.append(
                nn.BatchNorm2d(out_feat)
            )

            conv_blk.append(
                get_act_func(act_func)
            )

        self.block = nn.Sequential(*conv_blk)
    
    def forward(self, x):
        return self.block(x)

cfg = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))

class MNISTNetwork(nn.Module):
    def __init__(self, act_func):
        super().__init__()

        vgg_blk = []
        for idx, (num_conv, out_feat) in enumerate(cfg):
            vgg_blk.append(CNNBlock(num_conv, out_feat, act_func))
            if idx != 4:
                vgg_blk.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.feature = nn.Sequential(*vgg_blk)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(4096),
            nn.LazyLinear(10)
        )
    
    def forward(self, x):
        feature = self.feature(x)
        pred = self.classifier(feature)
        return pred
    
if __name__ == "__main__":
    net = MNISTNetwork(act_func="ReLU")
    random_input = torch.randn(16, 3, 28, 28)
    random_output = net(random_input)
    print(random_output.shape)