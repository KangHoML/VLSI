import os
import torch
import argparse
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from tqdm import tqdm

from data import TextDataset, split_dataset
from lstm import Seq2Seq

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="../../datasets/fra_eng.txt")
parser.add_argument("--num_sample", type=int, default=33000)
parser.add_argument("--embed_size", type=int, default=256)
parser.add_argument("--hidden_size", type=int, default=256)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--epoch", type=int, default=20)

def plot_loss(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses)+1), train_losses, label ='Train_Loss', marker ='o')
    plt.plot(range(1, len(val_losses)+1), val_losses, label ='Validation_Loss', marker ='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    title = "LSTM with Attention"
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.savefig(f'./result/{title}.png')

def train(args):
    os.makedirs("result", exist_ok=True)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Device : {device}")

    text_dataset = TextDataset(args.data_path, args.num_sample, args.max_seq_len)
    train_dataset, val_dataset = split_dataset(text_dataset)
    train_loader, val_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True), \
                               DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    src_vocab_size, trg_vocab_size = len(text_dataset.src_vocab), len(text_dataset.trg_vocab)
    net = Seq2Seq(src_vocab_size, trg_vocab_size, args.embed_size, args.hidden_size).to(device)
    criterion = CrossEntropyLoss(ignore_index=0)
    optimizer = Adam(net.parameters(), lr=args.learning_rate)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    for epoch in range(args.epoch):
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        net.train()

        for data in tqdm(train_loader):
            encoder_inputs = data["encoder_input"].to(device)
            decoder_inputs = data["decoder_input"].to(device)
            decoder_targets = data["decoder_target"].to(device)

            output = net(encoder_inputs, decoder_inputs)
            loss = criterion(output.view(-1, output.size(-1)), decoder_targets.view(-1))

            loss.backward()
            optimizer.step()

            # calculate accuracy
            mask = decoder_targets != 0
            train_correct += ((output.argmax(dim=-1) == decoder_targets) * mask).sum().item()
            train_total += mask.sum().item()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_accuracy = 100 * train_correct / train_total
        train_losses.append(train_loss)

        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        net.eval()

        with torch.no_grad():
            for data in tqdm(val_loader):
                encoder_inputs = data["encoder_input"].to(device)
                decoder_inputs = data["decoder_input"].to(device)
                decoder_targets = data["decoder_target"].to(device)

                output = net(encoder_inputs, decoder_inputs)
                loss = criterion(output.view(-1, output.size(-1)), decoder_targets.view(-1))

                # calculate accuracy
                mask = decoder_targets != 0
                val_correct += ((output.argmax(dim=-1) == decoder_targets) * mask).sum().item()
                val_total += mask.sum().item()

                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_accuracy = 100 * val_correct / val_total
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(net.state_dict(), './result/seq2seq.pth')

        print(f"Epoch [{epoch+1}/{args.epoch}]")
        print(f"  Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
        
    plot_loss(train_losses, val_losses)

if __name__ == "__main__":
    args = parser.parse_args()
    train(args)