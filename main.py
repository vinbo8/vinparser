import math
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from conllu import ConllParser


MAX_SENT = 200
LSTM_DIM = 150
EMBEDDING_DIM = 300
REDUCE_DIM = 30


class Network(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embeddings = torch.nn.Embedding(vocab_size, embedding_dim)
        self.lstm = torch.nn.LSTM(EMBEDDING_DIM, LSTM_DIM, 1, batch_first=True)
        self.mlp = torch.nn.Linear(LSTM_DIM, REDUCE_DIM)

    def forward(self, input):
        embeds = self.embeddings(input)
        print(embeds.shape)
        output, (h_n, c_n) = self.lstm(embeds)
        print(output.shape)
        reduced_out = self.mlp(output)
        print(reduced_out.shape)
        return embeds


def pad(l, max_len):
    tensor = torch.LongTensor(l)
    diff = max_len - tensor.shape[0]
    l, r = math.floor(diff / 2), math.ceil(diff / 2)
    return F.pad(tensor, (l, r))


def main():
    c = ConllParser()
    with open('data/en-ud-train.conllu', 'r') as f:
        c.build(f)

    # vocab and indexes
    vocab = set(' '.join(block.raw() for block in c).split())
    vocab_size = len(vocab)
    word_to_idx = {word: i + 1 for i, word in enumerate(vocab)}
    word_to_idx['PAD'] = 0

    # sentences
    sentence_list = [block.raw().split() for block in c]

    sent_idxs = [pad([word_to_idx[word] for word in sent], MAX_SENT) for sent in sentence_list]

    input = torch.stack(sent_idxs)[:10]
    print(input.shape)

    n = Network(vocab_size, EMBEDDING_DIM)
    n(input)


if __name__ == '__main__':
    main()
