import sys
import configparser
import argparse
import torch
import Loader
import torch.nn.functional as F
from torch.autograd import Variable
from Runner import build_data
from Helpers import process_batch

parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true')
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--config', default='./config.ini')
<<<<<<< HEAD
parser.add_argument('--train', default='./data/en-ud-train.conllu.sem')
parser.add_argument('--dev', default='./data/en-ud-dev.conllu.sem')
parser.add_argument('--test', default='./data/en-ud-test.conllu.sem')
parser.add_argument('--embedd', default='')
=======
parser.add_argument('--train', action='append')
parser.add_argument('--dev', action='append')
parser.add_argument('--test', action='append')
parser.add_argument('--embed', action='append')
>>>>>>> c876d1d610a42508597dea5f1d6365510af9b6ca
args = parser.parse_args()

config = configparser.ConfigParser()
config.read(args.config)

BATCH_SIZE = int(config['tagger']['BATCH_SIZE'])
EMBED_DIM = int(config['tagger']['EMBED_DIM'])
LSTM_DIM = int(config['tagger']['LSTM_DIM'])
LSTM_LAYERS = int(config['tagger']['LSTM_LAYERS'])
MLP_DIM = int(config['tagger']['MLP_DIM'])
LEARNING_RATE = float(config['tagger']['LEARNING_RATE'])
EPOCHS = int(config['tagger']['EPOCHS'])


class Tagger(torch.nn.Module):
    def __init__(self, sizes, vocab, args):
        super().__init__()

        self.embeds = torch.nn.Embedding(sizes['vocab'], EMBED_DIM)
        self.embeds.weight.data.copy_(vocab.vectors)
        self.lstm = torch.nn.LSTM(EMBED_DIM, LSTM_DIM, LSTM_LAYERS, batch_first=True, bidirectional=True, dropout=0.5)
        self.relu = torch.nn.ReLU()
        self.mlp = torch.nn.Linear(2 * LSTM_DIM, MLP_DIM)
        self.out = torch.nn.Linear(MLP_DIM, sizes['semtags'])
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
            (x_forms, pack), x_tags, y_heads, y_deprels = batch.form, batch.sem, batch.head, batch.deprel

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
            (x_forms, pack), x_tags, y_heads, y_deprels = batch.form, batch.sem, batch.head, batch.deprel

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

    sets = (args.train[0], args.dev[0], args.test[0])
    (train_loader, dev_loader, test_loader), sizes, vocab = Loader.get_iterators(sets, args.embed[0], BATCH_SIZE, args.cuda)
    print(len(train_loader))

    tagger = Tagger(sizes, vocab, args)
    if args.cuda:
        tagger.cuda()

    # training
    print("Training")
    for epoch in range(EPOCHS):
        tagger.train_(epoch, train_loader)
        tagger.evaluate_(dev_loader)

    # test
    print("Eval")
    tagger.evaluate_(test_loader)


if __name__ == '__main__':
    main()
