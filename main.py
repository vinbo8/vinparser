import sys
import math
import torch
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable
from conllu import ConllParser


MAX_SENT = 300
LSTM_DIM = 150
EMBEDDING_DIM = 300
REDUCE_DIM = 30
BATCH_SIZE = 10
EPOCHS = 10
LEARNING_RATE = 1e-5

class Network(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embeddings = torch.nn.Embedding(vocab_size, embedding_dim)
        self.lstm = torch.nn.LSTM(EMBEDDING_DIM, LSTM_DIM, 1, batch_first=True)
        self.mlp_head = torch.nn.Linear(LSTM_DIM, REDUCE_DIM)
        self.mlp_dep = torch.nn.Linear(LSTM_DIM, REDUCE_DIM)
        self.biaffine_weight = torch.nn.Parameter(torch.rand(BATCH_SIZE, REDUCE_DIM + 1, REDUCE_DIM))
        self.softmax = torch.nn.Softmax(dim=2)

    def forward(self, forms):
        embeds = self.embeddings(forms)
        output, (h_n, c_n) = self.lstm(embeds)
        reduced_head = self.mlp_head(output)
        reduced_dep = self.mlp_dep(output)

        bias = Variable(torch.ones(BATCH_SIZE, MAX_SENT, 1))
        reduced_dep = torch.cat([reduced_dep, bias], 2)

        # ROW IS DEP, COL IS HEAD
        y_pred = self.softmax(reduced_dep @ self.biaffine_weight @ reduced_head.transpose(1, 2))
        return y_pred



def rel_pad(l, max_len):
    tensor = torch.LongTensor(l)
    diff = max_len - tensor.shape[0]
    return F.pad(tensor, (0, 0, 0, diff), value=-1)


def form_pad(l, max_len):
    tensor = torch.LongTensor(l)
    diff = max_len - tensor.shape[0]
    l, r = math.floor(diff / 2), math.ceil(diff / 2)
    return F.pad(tensor, (1, diff - 1))


def main():
    c = ConllParser()
    with open('data/sv-ud-train.conllu', 'r') as f:
        c.build(f)

    # vocab and indexes
    vocab = set(' '.join(block.raw() for block in c).split())
    vocab_size = len(vocab)
    word_to_idx = {word: i + 1 for i, word in enumerate(vocab)}
    word_to_idx['PAD'] = 0

    # sentences
    sentence_list = [block.raw().split() for block in c]
    deprel_list = [rel_pad(block.rels(), MAX_SENT) for block in c]

    sent_idxs = [form_pad([word_to_idx[word] for word in sent], MAX_SENT) for sent in sentence_list]
    forms = torch.stack(sent_idxs)[:200].data
    rels = torch.stack(deprel_list)[:200]

    # build out_set
    labels = torch.zeros(forms.shape[0], MAX_SENT, 1)
    # rewrite better for parallel GPU
    # this is very ugly, I should consider capital punishment
    for batch_no, _ in enumerate(rels):
        for rel in rels[batch_no]:
            if rel[1].data[0] == -1:
                continue
            labels[batch_no, rel[1].data[0]] = rel[0].data[0]

    labels = torch.squeeze(labels.type(torch.LongTensor))
    train_data = list(zip(forms, labels))
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    parser = Network(vocab_size, EMBEDDING_DIM)

    # training
    parser.train()
    criterion = torch.nn.NLLLoss()
    optimiser = torch.optim.Adam(parser.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        for i, data in enumerate(train_loader):
            forms, labels = data
            X = Variable(forms)
            y = Variable(labels, requires_grad=False)
            y_pred = parser(X)
            train_loss = criterion(y_pred, y)
            optimiser.zero_grad()
            train_loss.backward()
            optimiser.step()

        print(epoch, train_loss.data[0])


if __name__ == '__main__':
    main()
