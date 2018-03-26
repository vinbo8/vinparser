import os
import math
import argparse
import torch
import torch.utils.data
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from Helpers import build_data

LSTM_DIM = 200
LSTM_DEPTH = 3
EMBEDDING_DIM = 100
REDUCE_DIM = 500
BATCH_SIZE = 10
EPOCHS = 5
LEARNING_RATE = 2e-3
DEBUG_SIZE = 100


class Biaffine(torch.nn.Module):

    def __init__(self, in1_features, in2_features):
        super(Biaffine, self).__init__()
        self.in1_features = in1_features
        self.in2_features = in2_features

        self.weight = torch.nn.Parameter(torch.Tensor(BATCH_SIZE, in1_features, in2_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(0))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input1, input2):
        batch_size, len1, dim1 = input1.size()
        ones = torch.ones(batch_size, len1, 1)
        input1 = torch.cat((input1, Variable(ones)), dim=2)

        biaffine = input1 @ self.weight @ input2.transpose(1, 2)
        return biaffine

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + 'in1_features=' + str(self.in1_features) \
            + ', in2_features=' + str(self.in2_features) \
            + ', out_features=' + str(self.out_features) + ')'


class Network(torch.nn.Module):
    def __init__(self, vocab_size, tag_vocab):
        super().__init__()
        self.embeddings_forms = torch.nn.Embedding(vocab_size, EMBEDDING_DIM)
        self.embeddings_tags = torch.nn.Embedding(tag_vocab, EMBEDDING_DIM)
        self.lstm = torch.nn.LSTM(2 * EMBEDDING_DIM, LSTM_DIM, LSTM_DEPTH,
                                  batch_first=True, bidirectional=True, dropout=0.33)
        self.mlp_head = torch.nn.Linear(2 * LSTM_DIM, REDUCE_DIM)
        self.mlp_dep = torch.nn.Linear(2 * LSTM_DIM, REDUCE_DIM)
        self.biaffine_weight = torch.nn.Parameter(torch.rand(BATCH_SIZE, REDUCE_DIM + 1, REDUCE_DIM), requires_grad=True)
        self.softmax = torch.nn.LogSoftmax(dim=2)
        # self.criterion = torch.nn.NLLLoss(reduce=False)
        self.criterion = torch.nn.NLLLoss()
        self.relu = torch.nn.ReLU()
        self.optimiser = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.9))
        self.dropout = torch.nn.Dropout(p=0.33)
        self.bilinear = torch.nn.Bilinear(REDUCE_DIM, REDUCE_DIM, 500, bias=False)
        self.compose = torch.nn.Linear(REDUCE_DIM, REDUCE_DIM)
        self.final = torch.nn.Linear(REDUCE_DIM, REDUCE_DIM)
        self.ripoff = Biaffine(REDUCE_DIM + 1, REDUCE_DIM)

    def forward(self, forms, tags, sizes, pack):
        # for debug:
        # sizes[0, :, 0] = 1
        # sizes = torch.stack([sizes for i in range(EMBEDDING_DIM)], dim=2)
        # sizes = torch.stack([sizes for i in range(2 * LSTM_DIM)], dim=2)

        MAX_SENT = forms.size(1)
        form_embeds = self.dropout(self.embeddings_forms(forms))
        assert form_embeds.shape == torch.Size([BATCH_SIZE, MAX_SENT, EMBEDDING_DIM])

        tag_embeds = self.dropout(self.embeddings_tags(tags))
        assert tag_embeds.shape == torch.Size([BATCH_SIZE, MAX_SENT, EMBEDDING_DIM])

        embeds = torch.cat([form_embeds, tag_embeds], dim=2)

        embeds = torch.nn.utils.rnn.pack_padded_sequence(embeds, pack, batch_first=True)
        output, _ = self.lstm(embeds)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        reduced_head = self.dropout(self.relu(self.mlp_head(output)))
        reduced_dep = self.dropout(self.relu(self.mlp_dep(output)))

        y_pred = self.ripoff(reduced_head, reduced_dep)
        return y_pred

    def train_(self, epoch, train_loader):
        self.train()
        for i, (forms, tags, labels, sizes) in enumerate(train_loader):
            forms, tags, labels, sizes = [torch.stack(list(i)) for i in zip(
                *sorted(zip(forms, tags, labels, sizes), key=lambda x: x[3].nonzero().size(0), reverse=True))]
            trunc = max([i.nonzero().size(0) + 1 for i in sizes])
            X1 = Variable(forms[:, :trunc])
            X2 = Variable(tags[:, :trunc])
            y = Variable(labels[:, :trunc], requires_grad=False)
            y[:, 0] = 0
            mask = Variable(sizes[:, :trunc])
            pack = [i.nonzero().size(0) + 1 for i in sizes]
            # squeezing
            y_pred = self(X1, X2, mask, pack)
            dims = y.size()
            a = y_pred.view(BATCH_SIZE * dims[1], dims[1])
            b = y.contiguous().view(BATCH_SIZE * dims[1])
            train_loss = F.cross_entropy(a, b, ignore_index=-1)
            self.zero_grad()
            train_loss.backward()
            self.optimiser.step()

            print("Epoch: {}\t{}/{}\tloss: {}".format(epoch, (i + 1) * len(forms), len(train_loader.dataset), train_loss.data[0]))

    def evaluate_(self, test_loader):
        correct = 0
        total_deps = 0
        self.eval()
        for i, (forms, tags, labels, sizes) in enumerate(test_loader):
            forms, tags, labels, sizes = [torch.stack(list(i)) for i in zip(*sorted(zip(forms, tags, labels, sizes), key=lambda x: x[3].nonzero().size(0), reverse=True))]
            trunc = max([i.nonzero().size(0) + 1 for i in sizes])
            X1 = Variable(forms[:, :trunc])
            X2 = Variable(tags[:, :trunc])
            y = Variable(labels[:, :trunc], requires_grad=False)
            mask = Variable(sizes[:, :trunc])
            pack = [i.nonzero().size(0) + 1 for i in sizes]
            y_pred = self(X1, X2, mask, pack)
            temp = y_pred.max(2)[1]
            try:
                correct += ((y == y_pred.max(2)[1]) * mask.type(torch.ByteTensor)).nonzero().size(0)
            except RuntimeError:
                print("fail")
                correct += 0
            total_deps += mask.nonzero().size(0)

        print("Accuracy = {}/{} = {}".format(correct, total_deps, (correct / total_deps)))


if __name__ == '__main__':
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    conll, train_loader = build_data('sv-ud-train.conllu', BATCH_SIZE)
    _, test_loader = build_data('sv-ud-test.conllu', BATCH_SIZE, conll)

    parser = Network(conll.vocab_size, conll.pos_size)

    # training
    print("Training")
    for epoch in range(EPOCHS):
        parser.train_(epoch, train_loader)

    # test
    print("Eval")
    parser.evaluate_(test_loader)
