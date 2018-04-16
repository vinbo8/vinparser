import os
import math
import random
import argparse
import configparser
import torch
import torch.utils.data
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from Helpers import build_data, process_batch
import Helpers
import Loader
from Modules import Biaffine, LongerBiaffine, LinearAttention, ShorterBiaffine, CharEmbedding

random.seed(1337)
np.random.seed(1337)

parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true')
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--config', default='./config.ini')
parser.add_argument('--train', default='./data/en-ud-train.conllu.sem')
parser.add_argument('--dev', default='./data/en-ud-dev.conllu.sem')
parser.add_argument('--test', default='./data/en-ud-test.conllu.sem')
parser.add_argument('--embed')

args = parser.parse_args()

config = configparser.ConfigParser()
config.read(args.config)

BATCH_SIZE = int(config['parser']['BATCH_SIZE'])
EMBED_DIM = int(config['parser']['EMBED_DIM'])
LSTM_DIM = int(config['parser']['LSTM_DIM'])
LSTM_LAYERS = int(config['parser']['LSTM_LAYERS'])
REDUCE_DIM_ARC = int(config['parser']['REDUCE_DIM_ARC'])
REDUCE_DIM_LABEL = int(config['parser']['REDUCE_DIM_LABEL'])
LEARNING_RATE = float(config['parser']['LEARNING_RATE'])
EPOCHS = int(config['parser']['EPOCHS'])


class Parser(torch.nn.Module):
    def __init__(self, sizes, vocab, args):
        super().__init__()

        self.use_cuda = args.cuda
        self.debug = args.debug

        # self.embeddings_chars = CharEmbedding(sizes, EMBED_DIM)
        self.embeddings_forms = torch.nn.Embedding(sizes['vocab'], EMBED_DIM)
        if args.embed:
            self.embeddings_forms.weight.data.copy_(vocab.vectors)
        self.embeddings_forms_rand = torch.nn.Embedding(sizes['vocab'], EMBED_DIM) 
        self.embeddings_tags = torch.nn.Embedding(sizes['postags'], EMBED_DIM)
        self.lstm = torch.nn.LSTM(500  + sizes['postags'], LSTM_DIM, LSTM_LAYERS + 1,
                                  batch_first=True, bidirectional=True, dropout=0.33)
        self.mlp_head = torch.nn.Linear(2 * LSTM_DIM, REDUCE_DIM_ARC)
        self.mlp_dep = torch.nn.Linear(2 * LSTM_DIM, REDUCE_DIM_ARC)
        self.mlp_deprel_head = torch.nn.Linear(2 * LSTM_DIM, REDUCE_DIM_LABEL)
        self.mlp_deprel_dep = torch.nn.Linear(2 * LSTM_DIM, REDUCE_DIM_LABEL)
        #pos
        self.mlp_tag = torch.nn.Linear(300, 150)
        self.out_tag = torch.nn.Linear(150, sizes['postags'])
        #sem
        self.mlp_semtag = torch.nn.Linear(500, 200)
        self.out_semtag = torch.nn.Linear(200, sizes['semtags'])

        self.lstm_tag = torch.nn.LSTM(EMBED_DIM * 2, 150, 1,
                                  batch_first=True, bidirectional=True, dropout=0.33)

        self.lstm_semtag = torch.nn.LSTM(EMBED_DIM * 5 +  sizes['postags'], 250, 1,
                                  batch_first=True, bidirectional=True, dropout=0.33)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=0.33)
        # self.biaffine = Biaffine(REDUCE_DIM_ARC + 1, REDUCE_DIM_ARC, BATCH_SIZE)
        self.biaffine = ShorterBiaffine(REDUCE_DIM_ARC)
        self.label_biaffine = LongerBiaffine(REDUCE_DIM_LABEL, REDUCE_DIM_LABEL, sizes['deprels'])
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self.optimiser = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.9))

        if self.use_cuda:
            self.biaffine.cuda()
            self.label_biaffine.cuda()

    def forward(self, forms, tags, semtags, pack):
        # embed and dropout forms and tags; concat
        # TODO: same mask embedding
        # char_embeds = self.embeddings_chars(chars, pack)
        form_embeds = self.dropout(self.embeddings_forms(forms))
        tag_embeds = self.dropout(self.embeddings_tags(tags))
        form_embeds_rand = self.dropout(self.embeddings_forms_rand(forms))
        form_embeds = torch.cat([form_embeds_rand ,form_embeds], dim = 2)

        # pack/unpack for LSTM_tag
        tagging_embeds = torch.nn.utils.rnn.pack_padded_sequence(form_embeds, pack.tolist(), batch_first=True)
        output_tag, _ = self.lstm_tag(tagging_embeds)
        output_tag, _ = torch.nn.utils.rnn.pad_packed_sequence(output_tag, batch_first=True)
        #pos 
        mlp_tag = self.dropout(self.relu(self.mlp_tag(output_tag)))
        y_pred_tag = self.out_tag(mlp_tag)
     
    #    output_tag  = torch.cat([output_tag, form_embeds,  y_pred_tag], dim = 2)

        # pack/unpack for LSTM_semtag
      #  semtagging_embeds = torch.nn.utils.rnn.pack_padded_sequence(output_tag, pack.tolist(), batch_first=True)
       # output_semtag, _ = self.lstm_semtag(semtagging_embeds)
       # output_semtag, _ = torch.nn.utils.rnn.pad_packed_sequence(output_semtag, batch_first=True)
        #sem
        #mlp_semtag = self.dropout(self.relu(self.mlp_semtag(output_semtag)))
        #y_pred_semtag = self.out_semtag(mlp_semtag)
        #print(output_tag.size(), output_semtag.size())
   

        embeds = torch.cat([form_embeds, output_tag, y_pred_tag], dim = 2)
        print(embeds.size())
   
        # pack/unpack for LSTM_parse
        embeds = torch.nn.utils.rnn.pack_padded_sequence(embeds, pack.tolist(), batch_first=True)
        
        output, _ = self.lstm(embeds)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        # predict heads
        reduced_head_head = self.dropout(self.relu(self.mlp_head(output)))
        reduced_head_dep = self.dropout(self.relu(self.mlp_dep(output)))
        y_pred_head = self.biaffine(reduced_head_head, reduced_head_dep)

        if self.debug:
            return y_pred_head, Variable(torch.rand(y_pred_head.size()))

        # predict deprels using heads
        reduced_deprel_head = self.dropout(self.relu(self.mlp_deprel_head(output)))
        reduced_deprel_dep = self.dropout(self.relu(self.mlp_deprel_dep(output)))
        predicted_labels = y_pred_head.max(2)[1]
        selected_heads = torch.stack([torch.index_select(reduced_deprel_head[n], 0, predicted_labels[n])
                                        for n, _ in enumerate(predicted_labels)])
        y_pred_label = self.label_biaffine(selected_heads, reduced_deprel_dep)
        y_pred_label = Helpers.extract_best_label_logits(predicted_labels, y_pred_label, pack)
        if self.use_cuda:
            y_pred_label = y_pred_label.cuda()

        return y_pred_head, y_pred_label, y_pred_tag#, y_pred_semtag, y_pred_tag

    def train_(self, epoch, train_loader):
        self.train()
        train_loader.init_epoch()

        for i, batch in enumerate(train_loader):
            (x_forms, pack), x_tags, y_heads, y_deprels, x_sem = batch.form, batch.upos, batch.head, batch.deprel, batch.sem

            mask = torch.zeros(pack.size()[0], max(pack)).type(torch.LongTensor)
            for n, size in enumerate(pack):
                mask[n, 0:size] = 1

            y_pred_head, y_pred_deprel, y_pred_tag  = self(x_forms, x_tags, x_sem, pack)

            # reshape for cross-entropy
            batch_size, longest_sentence_in_batch = y_heads.size()

            # predictions: (B x S x S) => (B * S x S)
            # heads: (B x S) => (B * S)
            y_pred_head = y_pred_head.view(batch_size * longest_sentence_in_batch, -1)
            y_heads = y_heads.contiguous().view(batch_size * longest_sentence_in_batch)

            # predictions: (B x S x D) => (B * S x D)
            # heads: (B x S) => (B * S)
            y_pred_deprel = y_pred_deprel.view(batch_size * longest_sentence_in_batch, -1)
            y_deprels = y_deprels.contiguous().view(batch_size * longest_sentence_in_batch)

            #sem
            #y_pred_semtag = y_pred_semtag.view(batch_size * longest_sentence_in_batch, -1)
            #x_sem = x_sem.contiguous().view(batch_size * longest_sentence_in_batch)
            
            #pos
            y_pred_tag = y_pred_tag.view(batch_size * longest_sentence_in_batch, -1)
            x_tags = x_tags.contiguous().view(batch_size * longest_sentence_in_batch)
 
            # sum losses
            train_loss = self.criterion(y_pred_head, y_heads)
            if not self.debug:
                train_loss += self.criterion(y_pred_deprel, y_deprels)
               # train_loss += 0.5 * self.criterion(y_pred_semtag, x_sem)
                train_loss += 0.5 * self.criterion(y_pred_tag, x_tags)

            self.zero_grad()
            train_loss.backward()
            self.optimiser.step()

            print("Epoch: {}\t{}/{}\tloss: {}".format(epoch, (i + 1) * len(x_forms), len(train_loader.dataset), train_loss.data[0]))

    def evaluate_(self, test_loader):
        las_correct, uas_correct, semtags_correct, tags_correct, total = 0, 0, 0, 0, 0
        self.eval()
        for i, batch in enumerate(test_loader):
            (x_forms, pack), x_tags, y_heads, y_deprels, x_sem = batch.form, batch.upos, batch.head, batch.deprel, batch.sem

            mask = torch.zeros(pack.size()[0], max(pack)).type(torch.LongTensor)
            for n, size in enumerate(pack):
                mask[n, 0:size] = 1

            # get labels
            # TODO: ensure well-formed tree
            y_pred_head, y_pred_deprel, y_pred_tag  = [i.max(2)[1] for i in self(x_forms, x_tags, x_sem, pack)]

            mask = mask.type(torch.ByteTensor)
            if self.use_cuda:
                mask = mask.cuda()

            mask = Variable(mask)
            heads_correct = ((y_heads == y_pred_head) * mask)
            deprels_correct = ((y_deprels == y_pred_deprel) * mask)
            #tags_correct = ((x_tags == y_pred_tag) * mask)

            # excepts should never trigger; leave them in just in case
            try:
                uas_correct += heads_correct.nonzero().size(0)
            except RuntimeError:
                pass
            try:
                las_correct += (heads_correct * deprels_correct).nonzero().size(0)
            except RuntimeError:
                pass
            #try:
             #   semtags_correct += ((x_sem == y_pred_semtag) * mask).nonzero().size(0)
            #except RuntimeError:
             #   pass
            try:
                tags_correct += ((x_tags == y_pred_tag) * mask).nonzero().size(0)
            except RuntimeError:
                pass
            total += mask.nonzero().size(0)

        print("UAS = {}/{} = {}\nLAS = {}/{} = {}\nTAG = {}/{} = {}\n".format(uas_correct, total, uas_correct / total,
                                                          las_correct, total, las_correct / total, 
                                                          tags_correct, total,  tags_correct / total)) #,
                                                          #semtags_correct, total,  semtags_correct / total))


if __name__ == '__main__':
    # args
    loaders = Loader.get_iterators(args, BATCH_SIZE)
   
    train_loader, dev_loader, test_loader, vocab =  loaders['train'], loaders['dev'], loaders['test'],  loaders['vocab'] 
    sizes = loaders['sizes']   

    parser = Parser(sizes, vocab, args)
    if args.cuda:
        parser.cuda()

    # training
    print("Training")
    for epoch in range(EPOCHS):
        parser.train_(epoch, train_loader)
        parser.evaluate_(dev_loader)

    # test
    print("Eval")
    parser.evaluate_(test_loader)
