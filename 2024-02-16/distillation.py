import os
import torch
import argparse

from net import StudentNet, TeacherNet
from train import train, test, train_kd
from baseline import prepare_data

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="../../datasets/CIFAR-10")
parser.add_argument("--match", type=str, default="logit", choices=["logit", "representation", "feature_map"])
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--learning_rate", type=float, default=1e-3)
parser.add_argument("--epoch", type=int, default=10)
parser.add_argument("--temperature", type=int, default=2)
parser.add_argument("--kd_weight", type=float, default=0.25)
parser.add_argument("--ce_weight", type=float, default=0.75)

'''
Knowledge Distillation
    way 1. match the output logits (class probability distribution)
    way 2. match the hidden representation (output of flatten) 
    way 3. match the feature map (output of pooling)
'''
def knowledge_distillation(args):
    torch.manual_seed(42)
    teacher_net, student_net = TeacherNet(), StudentNet()
    train_loader, val_loader = prepare_data(args)
    print('\nResults wt. Knowledge Distillation')

    # Train & Test teacher network
    weights = "./result/teacher_net.pth"
    if os.path.exists(weights):
        teacher_net.load_state_dict(torch.load(weights))
    else:
        train(args, teacher_net, train_loader)
    
    # Train & Test student network
    print(f'\nStudent Network with {args.match}')
    train_kd(args, teacher_net, student_net, train_loader)
    test(student_net, val_loader)
    print("Norm of 1st layer of Student Network:", torch.norm(student_net.features[0].weight).item())
    
if __name__ == '__main__':
    os.makedirs("result", exist_ok=True)

    args = parser.parse_args()
    knowledge_distillation(args)