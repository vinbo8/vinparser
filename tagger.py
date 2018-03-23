import sys
import argparse
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from parser import build_data

EMBED_DIM = 100
LSTM_DIM = 400
LSTM_LAYERS = 2
MLP_DIM = 400
LEARNING_RATE = 2e-3
EPOCHS = 10


class Tagger(torch.nn.Module):
    def __init__(self, vocab_size, tag_vocab):
        super().__init__()
        self.embeds = torch.nn.Embedding(vocab_size, EMBED_DIM)
        self.lstm = torch.nn.LSTM(EMBED_DIM, LSTM_DIM, LSTM_LAYERS, batch_first=True)
        self.relu = torch.nn.ReLU()
        self.mlp = torch.nn.Linear(LSTM_DIM, MLP_DIM)
        self.out = torch.nn.Linear(MLP_DIM, tag_vocab)
        self.softmax = torch.nn.LogSoftmax(dim=2)
        self.criterion = torch.nn.NLLLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE)

    def forward(self, forms):
        form_embeds = self.embeds(forms)
        lstm_out, _ = self.lstm(form_embeds)
        mlp_out = self.relu(self.mlp(lstm_out))
        debug = self.out(mlp_out)
        y_pred = self.softmax(debug).transpose(1, 2)
        return y_pred

    def train_(self, epoch, train_loader):
        self.train()
        for i, (forms, tags, labels, sizes) in enumerate(train_loader):
            forms, tags, labels, sizes = [torch.stack(list(i)) for i in zip(
                *sorted(zip(forms, tags, labels, sizes), key=lambda x: x[3].nonzero().size(0), reverse=True))]

            trunc = max([i.nonzero().size(0) + 1 for i in sizes])
            X = Variable(forms[:, :trunc])
            y = Variable(tags[:, :trunc])
            y_pred = self(X)
            train_loss = self.criterion(y_pred, y)
            self.zero_grad()
            train_loss.backward()
            self.optimizer.step()

            print("Epoch: {}\t{}/{}\tloss: {}".format(
                epoch, (i + 1) * len(forms), len(train_loader.dataset), train_loss.data[0]))

    def evaluate_(self, test_loader):
        correct, total = 0, 0
        self.eval()
        for i, (forms, tags, labels, sizes) in enumerate(test_loader):
            forms, tags, labels, sizes = [torch.stack(list(i)) for i in zip(
                *sorted(zip(forms, tags, labels, sizes), key=lambda x: x[3].nonzero().size(0), reverse=True))]

            trunc = max([i.nonzero().size(0) + 1 for i in sizes])
            X = Variable(forms[:, :trunc])
            y = Variable(tags[:, :trunc])
            sizes = Variable(sizes[:, :trunc])
            y_pred = self(X)
            temp = y_pred.max(1)[1]
            try:
                correct += ((y == temp) * sizes.type(torch.ByteTensor)).nonzero().size(0)
            except RuntimeError:
                print("fail")

            total += sizes.nonzero().size(0)

        print("Accuracy = {}/{} = {}".format(correct, total, (correct / total)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    conll, train_loader = build_data('sv-ud-train.conllu')
    # _, test_loader = build_data('sv-ud-test.conllu')

    tagger = Tagger(conll.vocab_size, conll.pos_size)
    _, test_loader = build_data('sv-ud-test.conllu', conll)

    # training
    print("Training")
    for epoch in range(EPOCHS):
        tagger.train_(epoch, train_loader)

    # test
    print("Eval")
    tagger.evaluate_(test_loader)


if __name__ == '__main__':
    main()