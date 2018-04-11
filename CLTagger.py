import sys
import configparser
import argparse
import torch
import Loader
import torch.nn.functional as F
from torch.autograd import Variable
from Parser import build_data
from Helpers import process_batch

parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true')
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--config', default='./config.ini')
parser.add_argument('--train', action='append')
parser.add_argument('--dev', action='append')
parser.add_argument('--test', action='append')
parser.add_argument('--embed', action='append')
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


class CLTagger(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.lstm_shared = torch.nn.LSTM(EMBED_DIM, LSTM_DIM, LSTM_LAYERS, batch_first=True, bidirectional=True, dropout=0.5)
        self.lstm_main = torch.nn.LSTM(LSTM_DIM * 2, LSTM_DIM, LSTM_LAYERS, batch_first=True, bidirectional=True, dropout=0.5)
        self.lstm_aux = torch.nn.LSTM(LSTM_DIM, LSTM_DIM, LSTM_LAYERS, batch_first=True, bidirectional=True, dropout=0.5)
        self.relu = torch.nn.ReLU()
        self.mlp_main = torch.nn.Linear(LSTM_DIM * 2, MLP_DIM)
        # self.mlp_aux = torch.nn.Linear(LSTM_DIM, MLP_DIM)
        self.criterion_main = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self.criterion_aux = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.9))
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward_main(self, forms, pack, sizes, vocab):
        # embeds + dropout
        embeds_main = torch.nn.Embedding(sizes['vocab'], EMBED_DIM)
        embeds_main.weight.data.copy_(vocab.vectors)
        form_embeds = self.dropout(embeds_main(forms))

        # pack/unpack for LSTM
        packed = torch.nn.utils.rnn.pack_padded_sequence(form_embeds, pack.tolist(), batch_first=True)
        lstm_out, _ = self.lstm_shared(packed)
        lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        # lstm_out_main, _ = self.lstm_main(lstm_out)
        # lstm_out_main, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out_main, batch_first=True)

        # LSTM => dense ReLU
        mlp_out = self.dropout(self.relu(self.mlp_main(lstm_out)))

        # reduce to dim no_of_tags
        out_main = torch.nn.Linear(MLP_DIM, sizes['postags'])
        return out_main(mlp_out)

    def forward_aux(self, forms, pack, sizes, vocab):
        # embeds + dropout
        embeds_aux = torch.nn.Embedding(sizes['vocab'], EMBED_DIM)
        embeds_aux.weight.data.copy_(vocab.vectors)
        form_embeds = self.dropout(embeds_aux(forms))

        # pack/unpack for LSTM
        packed = torch.nn.utils.rnn.pack_padded_sequence(form_embeds, pack.tolist(), batch_first=True)
        lstm_out, _ = self.lstm_shared(packed)
        lstm_out_aux, _ = self.lstm_aux(lstm_out)
        lstm_out_aux, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out_main, batch_first=True)

        # LSTM => dense ReLU
        # mlp_out = self.dropout(self.relu(self.mlp_aux(lstm_out_main)))

        # reduce to dim no_of_tags
        out_aux = torch.nn.Linear(lstm_out_aux, sizes['postags'])
        return out_aux(lstm_out_aux)

def train(model, epoch, train_loaders, type_task="main"):
    model.train()

    def get_loss(train_loader, type_task=type_task):
        train_loader[0][0].init_epoch()
        for i, batch in enumerate(train_loader[0][0]):
            (x_forms, pack), x_tags, y_heads, y_deprels = batch.form, batch.upos, batch.head, batch.deprel

            mask = torch.zeros(pack.size()[0], max(pack)).type(torch.LongTensor)
            for n, size in enumerate(pack):
                mask[n, 0:size] = 1

            if type_task == "aux":
                y_pred = model.forward_aux(x_forms, pack, train_loader[1], train_loader[2])
            else:
                y_pred = model.forward_main(x_forms, pack, train_loader[1], train_loader[2])                
            # reshape for cross-entropy
            batch_size, longest_sentence_in_batch = x_forms.size()

            # predictions: (B x S x T) => (B * S, T)
            # heads: (B x S) => (B * S)
            y_pred = y_pred.view(batch_size * longest_sentence_in_batch, -1)
            x_tags = x_tags.contiguous().view(batch_size * longest_sentence_in_batch)

            if type_task == "aux":
                train_loss = model.criterion_aux(y_pred, x_tags)
            else:
                train_loss = model.criterion_main(y_pred, x_tags)

            model.zero_grad()
            train_loss.backward()
            model.optimizer.step()

            print("Epoch: {}\t{}/{}\tloss: {}".format(
                epoch, (i + 1) * len(x_forms), len(train_loader[0][0].dataset), train_loss.data[0]))

    get_loss(train_loaders[0], type_task="main")
    get_loss(train_loaders[1], type_task="aux")

def evaluate_(model, test_loader, type_task="main"):
    correct, total = 0, 0
    model.eval()
    for i, batch in enumerate(test_loader):
        (x_forms, pack), x_tags, y_heads, y_deprels = batch.form, batch.upos, batch.head, batch.deprel
        
        mask = torch.zeros(pack.size()[0], max(pack)).type(torch.LongTensor)
        for n, size in enumerate(pack):
            mask[n, 0:size] = 1

            # get tags
        if type_task == "aux":
            y_pred = model.forward_aux(x_forms, pack).max(2)[1]
        else:
            y_pred = model.forward_main(x_forms, pack).max(2)[1]
            
        mask = Variable(mask.type(torch.ByteTensor))
        
        correct += ((x_tags == y_pred) * mask).nonzero().size(0)

        total += mask.nonzero().size(0)

    print("Accuracy = {}/{} = {}".format(correct, total, (correct / total)))

def main():

#    sets = (args.train[0], args.dev[0], args.test[0])
#    (train_loader, dev_loader, test_loader), sizes, vocab = Loader.get_iterators_cl(sets, args.embed[0], BATCH_SIZE, args.cuda)

    loaders = Loader.get_iterators_cl(args, BATCH_SIZE)

    tagger = CLTagger(args)
    if args.cuda:
        tagger.cuda()

    # training
    print("Training")
    for epoch in range(EPOCHS):
        train(tagger, epoch, loaders, type_task="main")
        evaluate(tagger, loaders[0][0][1], type_task="main")
        evaluate(tagger, loaders[1][0][1], type_task="aux")

    # test
    print("Eval")
    evaluate(tagger, loaders[0][0][2], type_task="main")

if __name__ == '__main__':
    main()
