import sys
import configparser
import argparse
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from Parser import build_data
from Helpers import process_batch

EMBED_DIM = 100
LSTM_DIM = 400
LSTM_LAYERS = 2
MLP_DIM = 400
LEARNING_RATE = 2e-3
EPOCHS = 5


class Tagger(torch.nn.Module):
    def __init__(self, vocab_size, tag_vocab):
        super().__init__()
        self.embeds = torch.nn.Embedding(vocab_size, EMBED_DIM)
        self.lstm = torch.nn.LSTM(EMBED_DIM, LSTM_DIM, LSTM_LAYERS, batch_first=True, bidirectional=True, dropout=0.5)
        self.relu = torch.nn.ReLU()
        self.mlp = torch.nn.Linear(2 * LSTM_DIM, MLP_DIM)
        self.out = torch.nn.Linear(MLP_DIM, tag_vocab)
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.9))
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, forms, pack):
        # embeds + dropout
        form_embeds = self.dropout(self.embeds(forms))

        # pack/unpack for LSTM
        packed = torch.nn.utils.rnn.pack_padded_sequence(form_embeds, pack, batch_first=True)
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)

        # LSTM => dense ReLU
        mlp_out = self.dropout(self.relu(self.mlp(lstm_out)))

        # reduce to dim no_of_tags
        return self.out(mlp_out)

    def train_(self, epoch, train_loader):
        self.train()
        for i, batch in enumerate(train_loader):
            forms, tags, mask, pack, y_heads, y_deprels = process_batch(batch)
            y_pred = self(forms, pack)

            # reshape for cross-entropy
            batch_size, longest_sentence_in_batch = forms.size()

            # predictions: (B x S x T) => (B * S, T)
            # heads: (B x S) => (B * S)
            y_pred = y_pred.view(batch_size * longest_sentence_in_batch, -1)
            tags = tags.contiguous().view(batch_size * longest_sentence_in_batch)

            train_loss = self.criterion(y_pred, tags)

            self.zero_grad()
            train_loss.backward()
            self.optimizer.step()

            print("Epoch: {}\t{}/{}\tloss: {}".format(
                epoch, (i + 1) * len(forms), len(train_loader.dataset), train_loss.data[0]))

    def evaluate_(self, test_loader):
        correct, total = 0, 0
        self.eval()
        for i, batch in enumerate(test_loader):
            forms, tags, mask, pack, y_heads, y_deprels = process_batch(batch)

            # get tags
            y_pred = self(forms, pack).max(2)[1]

            try:
                correct += ((tags == y_pred) * mask.type(torch.ByteTensor)).nonzero().size(0)
            except RuntimeError:
                pass

            total += mask.nonzero().size(0)

        print("Accuracy = {}/{} = {}".format(correct, total, (correct / total)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read('config.ini')
    conll, train_loader = build_data('sv-ud-train.conllu', int(config['tagger']['BATCH_SIZE']))

    tagger = Tagger(conll.vocab_size, conll.pos_size)
    _, test_loader = build_data('sv-ud-test.conllu', int(config['tagger']['BATCH_SIZE']), conll)

    # training
    print("Training")
    for epoch in range(EPOCHS):
        tagger.train_(epoch, train_loader)

    # test
    print("Eval")
    tagger.evaluate_(test_loader)


if __name__ == '__main__':
    main()