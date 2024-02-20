import thop
import torch
import torch.nn as nn
from torch.nn.functional import avg_pool1d

'''
Large Networ : To be used as a teacher
'''
class TeacherNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        feature_map = x # be used when match the feature map
        
        x = torch.flatten(x, start_dim=1)
        flattened_output = x

        x = self.classifier(x)
        hidden_representation = avg_pool1d(flattened_output, 2) # be used when match the representation
        return x, hidden_representation, feature_map
    
'''
Tiny Network : Network that needs to train & To be used as a student
               Number of layers and hidden units are smaller than TearcherNet
'''
class StudentNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.regressor = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        
        hidden_representation = torch.flatten(x, start_dim=1) # be used when match the representation
        feature_map = self.regressor(x) # be used when match the feature map

        x = self.classifier(hidden_representation)
        return x, hidden_representation, feature_map

if __name__ == "__main__":
    teacher_net, student_net = TeacherNet(), StudentNet()
    random_input = torch.randn(16, 3, 32, 32)
    
    teacher_output, _, _ = teacher_net(random_input)
    student_output, _, _ = student_net(random_input)
    t_flops, t_params = thop.profile(teacher_net, inputs=(random_input, ))
    s_flops, s_params = thop.profile(student_net, inputs=(random_input, ))
    
    print('Teacher Network')
    print(f'    Output Shape: {teacher_output.shape}')
    print(f'    Computation (GFLOPs): {t_flops}, Params (Millions): {t_params}')
    print('Student Network')
    print(f'    Output Shape: {student_output.shape}')
    print(f'    Computation (GFLOPs): {s_flops}, Params (Millions): {s_params}')
    

