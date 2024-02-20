import os
import torch
import argparse

from net import StudentNet, TeacherNet
from train import train, test, train_kd
from baseline import prepare_data

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="../../datasets/CIFAR-10")
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--learning_rate", type=float, default=1e-3)
parser.add_argument("--epoch", type=int, default=10)
parser.add_argument("--temperature", type=int, default=2)
parser.add_argument("--kd_weight", type=float, default=0.25)
parser.add_argument("--ce_weight", type=float, default=0.75)

# with Teacher Network
def match_output_logits(args):
    torch.manual_seed(42)
    teacher_net, student_net = TeacherNet(), StudentNet()
    train_loader, val_loader = prepare_data(args)
    print('\nResults wt. Knowledge Distillation')

    # Train & Test teacher network
    print('Teacher Network')
    train(args, teacher_net, train_loader)
    test(teacher_net, val_loader)
    
    # Train & Test student network
    print('\nStudent Network')
    train_kd(args, teacher_net, student_net, train_loader)
    test(student_net, val_loader)
    print("Norm of 1st layer of Student Network:", torch.norm(student_net.features[0].weight).item())


if __name__ == '__main__':
    os.makedirs("result", exist_ok=True)

    args = parser.parse_args()
    match_output_logits(args)