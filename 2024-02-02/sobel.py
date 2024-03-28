import torch
import argparse
import torch.nn as nn
import torchvision.transforms as T

from PIL import Image
from torch.optim import Adam
from torch.nn import MSELoss
from torch.nn import functional as F

class CustomFilter(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_x = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.conv_y = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
    
    def forward(self, x):
        edge_x = self.conv_x(x)
        edge_y = self.conv_y(x)
        edge = torch.sqrt(edge_x ** 2 + edge_y ** 2)
        return edge
    
def sobel_filter(input):
    kernel_x = torch.tensor([[-1., 0., 1.],
                             [-2., 0., 2.],
                             [-1., 0., 1.]], dtype=torch.float32).reshape((1, 1, 3, 3))
    kernel_y = torch.tensor([[-1., -2., -1.],
                            [0., 0., 0.],
                            [1., 2., 1.]], dtype=torch.float32).reshape((1, 1, 3, 3))
    
    target_x = F.conv2d(input, kernel_x, padding=1)
    target_y = F.conv2d(input, kernel_y, padding=1)

    return torch.sqrt(target_x ** 2 + target_y ** 2)

parser = argparse.ArgumentParser()
parser.add_argument("--img_path", type=str, default="./result/Lenna.png")
parser.add_argument("--learning_rate", type=float, default=0.1)
parser.add_argument("--epoch", type=int, default=100)

def train(args):
    input = torch.randn(1, 1, 8, 8)
    target = sobel_filter(input)

    filter = CustomFilter()
    criterion = MSELoss()
    optimizer = Adam(filter.parameters(), lr=args.learning_rate)

    for epoch in range(args.epoch):
        optimizer.zero_grad()
        output = filter(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

    torch.save(filter.state_dict(), "./result/sobel.pth")
    print("filter_x weights:", filter.conv_x.weight.data)
    print("filter_y weights:", filter.conv_y.weight.data)

def test(args):
    img = Image.open(args.img_path).convert("L")
    img_tensor = T.ToTensor()(img).unsqueeze(0)

    filter = CustomFilter()
    filter.load_state_dict(torch.load("./result/sobel.pth"))
    edge_detected = filter(img_tensor)
    edge_detected_img = edge_detected.squeeze()
    
    # Normalization
    edge_detected_img = (edge_detected_img - edge_detected_img.min()) \
                        / (edge_detected_img.max() - edge_detected_img.min()) * 255
    edge_detected_img = edge_detected_img.byte()
    
    result_img = T.ToPILImage()(edge_detected_img)
    result_img.save("./result/detected.png")

if __name__ == "__main__":
    args = parser.parse_args()
    train(args)
    test(args)



