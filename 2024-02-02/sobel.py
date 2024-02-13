import torch
import torch.nn as nn
import torchvision.transforms as T

from PIL import Image

class SobelFilter(nn.Module):
    def __init__(self):
        super().__init__()

        kernel_x = torch.tensor([[-1., 0., 1.],
                                [-2., 0., 2.],
                                [-1., 0., 1.]], dtype=torch.float32).reshape((1, 1, 3, 3))
        
        kernel_y = torch.tensor([[-1., -2., -1.],
                                [0., 0., 0.],
                                [1., 2., 1.]], dtype=torch.float32).reshape((1, 1, 3, 3))
        
        self.conv_x = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        self.conv_y = nn.Conv2d(1, 1, kernel_size=3, padding=1)

        with torch.no_grad():
            self.conv_x.weight = nn.Parameter(kernel_x, requires_grad=False)
            self.conv_y.weight = nn.Parameter(kernel_y, requires_grad=False)

    def forward(self, x):
        edge_x = self.conv_x(x)
        edge_y = self.conv_y(x)

        edge = torch.sqrt(edge_x ** 2 + edge_y ** 2)
        return edge
    
def edge_detector(img_path, save_path):
    img = Image.open(img_path).convert("L")
    img_tensor = T.ToTensor()(img).unsqueeze(0)

    filter = SobelFilter()
    edge_detected = filter(img_tensor)
    edge_detected_img = edge_detected.squeeze()

    edge_detected_img = (edge_detected_img - edge_detected_img.min()) \
                        / (edge_detected_img.max() - edge_detected_img.min()) * 255
    edge_detected_img = edge_detected_img.byte()

    result_img = T.ToPILImage()(edge_detected_img)
    result_img.save(save_path)


if __name__ == "__main__":
    img_path = './result/Lenna.png'
    save_path = './result/Lenna_detected.png'
    edge_detector(img_path, save_path)