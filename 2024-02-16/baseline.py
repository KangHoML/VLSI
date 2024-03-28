import os
import torch
import argparse

from torch.utils.data import DataLoader
from data import CIFAR10Dataset
from net import StudentNet, TeacherNet
from train import train, test

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="../../datasets/CIFAR-10")
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--learning_rate", type=float, default=1e-3)
parser.add_argument("--epoch", type=int, default=10)

def prepare_data(args):
    train_dataset, val_dataset = CIFAR10Dataset(args.data_path, True), \
                                 CIFAR10Dataset(args.data_path, False)
    train_loader, val_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4), \
                               DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    return train_loader, val_loader

# without Teahcer Network
def baseline(args):
    torch.manual_seed(42)
    teacher_net, student_net = TeacherNet(), StudentNet()
    train_loader, val_loader = prepare_data(args)
    print('Results wo. Knowledge Distillation')

    # Train & Test teacher network
    print('Teacher Network')
    weights = "./result/teacher_net.pth"
    if os.path.exists(weights):
        teacher_net.load_state_dict(torch.load(weights))
    else:
        train(args, teacher_net, train_loader)
    test(teacher_net, val_loader)
    
    # Train & Test student network
    print('\nStudent Network')
    train(args, student_net, train_loader)
    test(student_net, val_loader)
    print("Norm of 1st layer of Student Network:", torch.norm(student_net.features[0].weight).item())

if __name__ == '__main__':
    os.makedirs("result", exist_ok=True)

    args = parser.parse_args()
    baseline(args)