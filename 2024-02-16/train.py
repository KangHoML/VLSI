import torch

from torch.nn import CrossEntropyLoss
from torch.nn.functional import softmax, log_softmax
from torch.optim import Adam
from tqdm import tqdm

def train(args, net, train_loader):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    net = net.to(device)
    criterion = CrossEntropyLoss()
    optimizer = Adam(net.parameters(), lr=args.learning_rate)

    for epoch in range(args.epoch):
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        net.train()

        for inputs, labels in tqdm(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            outputs, _, _ = net(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_accuracy = 100 * train_correct / train_total

        print(f"Epoch [{epoch+1}/{args.epoch}]")
        print(f"  Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")    

def test(net, val_loader):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    net = net.to(device)
    criterion = CrossEntropyLoss()

    val_loss = 0.0
    val_correct = 0
    val_total = 0

    net.eval()

    with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs, _, _ = net(inputs)
                loss = criterion(outputs, labels)

                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                val_loss += loss.item()

    val_loss /= len(val_loader)
    val_accuracy = 100 * val_correct / val_total
    print(f"  Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

def train_kd(args, teacher_net, student_net, train_loader):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    student_net, teacher_net = student_net.to(device), teacher_net.to(device)
    ce_loss = CrossEntropyLoss()
    optimizer = Adam(student_net.parameters(), lr=args.learning_rate)

    teacher_net.eval()
    student_net.train()

    for epoch in range(args.epoch):
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in tqdm(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            # do not change the teacher's weights (don't need to train)
            with torch.no_grad():
                teacher_logits, _, _ = teacher_net(inputs) # used in distillation loss
            student_logits, _, _ = student_net(inputs)

            # generate the logits to class probabilities by using softmax
            # smoothing the logits by dividing temperature
            soft_targets = softmax(teacher_logits / args.temperature, dim=-1)
            soft_prob = log_softmax(student_logits / args.temperature, dim=-1)

            # calculate the distillation loss
            # scaled by temperature ** 2 is suggested by the paper
            distillation_loss = -torch.sum(soft_targets * soft_prob) / soft_prob.size()[0] * \
                                (args.temperature ** 2)
            # calculate the classification loss by using cross entropy loss
            classification_loss = ce_loss(student_logits, labels)

            # total loss: weighted sum of two losses
            loss = distillation_loss * args.kd_weight + classification_loss * args.ce_weight

            loss.backward()
            optimizer.step()

            _, predicted = torch.max(student_logits.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_accuracy = 100 * train_correct / train_total

        print(f"Epoch [{epoch+1}/{args.epoch}]")
        print(f"  Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")