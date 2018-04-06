import sys
import configparser
import argparse
import torch
import Loader
import torch.nn.functional as F
from torch.autograd import Variable
from Parser import build_data
from Helpers import process_batch

BATCH_SIZE = 50
EMBED_DIM = 100
LSTM_DIM = 40
LSTM_LAYERS = 1
MLP_DIM = 400
LEARNING_RATE = 2e-3
EPOCHS = 5


class Tagger(torch.nn.Module):
    def __init__(self, sizes, args):
        super().__init__()
        self.embeds = torch.nn.Embedding(sizes['vocab'], EMBED_DIM)
        self.lstm = torch.nn.LSTM(EMBED_DIM, LSTM_DIM, LSTM_LAYERS, batch_first=True, bidirectional=True, dropout=0.5)
        self.relu = torch.nn.ReLU()
        self.mlp = torch.nn.Linear(2 * LSTM_DIM, MLP_DIM)
        self.out = torch.nn.Linear(MLP_DIM, sizes['postags'])
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.9))
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, forms, pack):
        # embeds + dropout
        form_embeds = self.dropout(self.embeds(forms))

        # pack/unpack for LSTM
        packed = torch.nn.utils.rnn.pack_padded_sequence(form_embeds, pack.tolist(), batch_first=True)
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)

        # LSTM => dense ReLU
        mlp_out = self.dropout(self.relu(self.mlp(lstm_out)))

        # reduce to dim no_of_tags
        return self.out(mlp_out)

    def train_(self, epoch, train_loader):
        self.train()
        train_loader.init_epoch()

        for i, batch in enumerate(train_loader):
            (x_forms, pack), x_tags, y_heads, y_deprels = batch.form, batch.upos, batch.head, batch.deprel

            mask = torch.zeros(pack.size()[0], max(pack)).type(torch.LongTensor)
            for n, size in enumerate(pack):
                mask[n, 0:size] = 1

            y_pred = self(x_forms, pack)

            # reshape for cross-entropy
            batch_size, longest_sentence_in_batch = x_forms.size()

            # predictions: (B x S x T) => (B * S, T)
            # heads: (B x S) => (B * S)
            y_pred = y_pred.view(batch_size * longest_sentence_in_batch, -1)
            x_tags = x_tags.contiguous().view(batch_size * longest_sentence_in_batch)

            train_loss = self.criterion(y_pred, x_tags)

            self.zero_grad()
            train_loss.backward()
            self.optimizer.step()

            print("Epoch: {}\t{}/{}\tloss: {}".format(
                epoch, (i + 1) * len(x_forms), len(train_loader.dataset), train_loss.data[0]))

    def evaluate_(self, test_loader):
        correct, total = 0, 0
        self.eval()
        for i, batch in enumerate(test_loader):
            (x_forms, pack), x_tags, y_heads, y_deprels = batch.form, batch.upos, batch.head, batch.deprel

            mask = torch.zeros(pack.size()[0], max(pack)).type(torch.LongTensor)
            for n, size in enumerate(pack):
                mask[n, 0:size] = 1

            # get tags
            y_pred = self(x_forms, pack).max(2)[1]

            mask = Variable(mask.type(torch.ByteTensor))

            correct += ((x_tags == y_pred) * mask).nonzero().size(0)

            total += mask.nonzero().size(0)

        print("Accuracy = {}/{} = {}".format(correct, total, (correct / total)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--train', default='./data/sv-ud-train.conllu')
    parser.add_argument('--dev', default='./data/sv-ud-dev.conllu')
    parser.add_argument('--test', default='./data/sv-ud-test.conllu')
    args = parser.parse_args()

    (train_loader, dev_loader, test_loader), sizes = Loader.get_iterators(args, BATCH_SIZE)

    tagger = Tagger(sizes, args)
    if args.cuda:
        tagger.cuda()

    # training
    print("Training")
    for epoch in range(EPOCHS):
        tagger.train_(epoch, train_loader)
        if not args.cuda:
            tagger.evaluate_(dev_loader)

    # test
    print("Eval")
    tagger.evaluate_(test_loader)


if __name__ == '__main__':
    main()