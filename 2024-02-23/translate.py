import torch

from data import TextDataset
from lstm import Seq2Seq
from transformer import Transformer

def seq_to_src(dataset, input_seq):
    sent = ''
    for encoded_word in input_seq:
        if encoded_word != 0 :
            sent = sent + dataset.index_to_src[encoded_word] + ' '
    return sent

def seq_to_trg(dataset, input_seq):
    sent = ''
    for encoded_word in input_seq:
        if (encoded_word != 0 and encoded_word != dataset.trg_vocab['<sos>'] and 
            encoded_word != dataset.trg_vocab['<eos>']):
            sent = sent + dataset.index_to_trg[encoded_word] + ' '
    return sent


def decode_seq(dataset, net, input_seq, max_seq_len, device):
    input_seq = torch.tensor(input_seq, dtype=torch.long).to(device)
    hidden, cell = net.encoder(input_seq)

    decoder_input = torch.tensor([dataset.trg_vocab['<sos>']],
                                  dtype=torch.long).unsqueeze(0).to(device)
    decoded_token = []

    for _ in range(max_seq_len):
        output, hidden, cell = net.decoder(decoder_input, hidden, cell)
        output_token = output.argmax(dim=-1).item()

        if output_token == dataset.trg_vocab['<eos>']:
            break

        decoded_token.apeend(output_token)
        decoder_input = torch.tensor([output_token],
                                     dtype=torch.long).unsqueeze(0).to(device)
    
    decoded_seq = ' '.join(dataset.index_to_src[token] for token in decoded_token)
    return decoded_seq

def translate(dataset, net, device):
    net = net.to(device)
    for seq_idx in [3, 50, 100, 300, 1001]:
        data = dataset[seq_idx]
        encoder_input_seq = data["encoder_input"]
        decoder_input_seq = data["decoder_input"]
        translated = decode_seq(dataset, net, encoder_input_seq, max_seq_len=16, device=device)

        print(f"    Input(Eng)       : {seq_to_src(dataset, encoder_input_seq)}")
        print(f"    Target(Fra)      : {seq_to_trg(dataset, decoder_input_seq)}")
        print(f"    Translated(Fra)  : {translated}")
        print("-" * 50)

if __name__ == "__main__":
    data_path = "../../dat,asets/fra_eng.txt"
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dataset = TextDataset(data_path, num_sample=33000, max_len=16)
    src_vocab_size, trg_vocab_size = len(dataset.src_vocab), len(dataset.trg_vocab)

    print("Model: Seq2Seq")
    net = Seq2Seq(src_vocab_size, trg_vocab_size, 256)
    net.load_state_dict(torch.load('./result/Seq2Seq.pth'))
    translate(dataset, net, device)

    print("Model: Transformer")
    net = Transformer(512, 6, src_vocab_size, trg_vocab_size, 0, 0,
                      max_seq_len=16, device=device)
    net.load_state_dict(torch.load("./result/Transformer.pth"))
    translate(dataset, net, device)    
