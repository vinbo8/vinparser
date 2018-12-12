import os
import pprint
import numpy as np
import Helpers 
import torch
from torch import nn
from scripts import cle
from torch.autograd import Variable
from collections import Counter
import torch.nn.functional as F
from Modules import ShorterBiaffine, LongerBiaffine, ParallelLSTM

class Parser(nn.Module):
    def __init__(self, args, vocabs):
        super().__init__()

        self.args = args
        self.vocabs = vocabs
        
        self.embeddings_rand = nn.Embedding(len(vocabs['forms']), args.embed_dim)
        self.embeddings_forms = nn.Embedding(len(vocabs['forms']), args.embed_dim)
        
        # assume that embeddings exist
        self.embeddings_forms.weight.data.copy_(vocabs['forms'].vectors)

        # size should be embed_size + whatever the other embeddings have
        self.lstm = ParallelLSTM(args.embed_dim, args.lstm_dim, args.lstm_layers,
                                 batch_first=True, bidirectional=True, dropout=args.dropout)

        self.lstm = nn.DataParallel(self.lstm)

        self.mlp_arc_parent = nn.Linear(2 * args.lstm_dim, args.mlp_arc_dim)
        self.mlp_arc_child = nn.Linear(2 * args.lstm_dim, args.mlp_arc_dim)
        self.mlp_label_parent = nn.Linear(2 * args.lstm_dim, args.mlp_label_dim)
        self.mlp_label_child = nn.Linear(2 * args.lstm_dim, args.mlp_label_dim)

        self.arc_biaffine = ShorterBiaffine(args.mlp_arc_dim).to(args.device)
        self.label_biaffine = LongerBiaffine(args.mlp_label_dim, args.mlp_label_dim, len(vocabs['deprels']))

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=args.dropout)

        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)
        self.optimiser = torch.optim.Adam(self.parameters(), lr=args.lr, betas=(0.9, 0.9))

    def forward(self, batch, dev=False, downstream=False):
        if not downstream:
            (forms, form_pack), tags = batch.form, batch.upos
        else:
            forms, form_pack = batch[0], batch[1]
            
        embeds = self.dropout(self.embeddings_rand(forms))
        if self.args.src_embed_file:
            embeds += self.dropout(self.embeddings_forms(forms))

        # pack/unpack for LSTM
        output = self.lstm(embeds, forms, form_pack)

        # predict heads
        reduced_arc_parent = self.dropout(self.relu(self.mlp_arc_parent(output)))
        reduced_arc_child = self.dropout(self.relu(self.mlp_arc_child(output)))
        y_pred_head = self.arc_biaffine(reduced_arc_parent, reduced_arc_child)

        # predict deprels using heads
        reduced_label_parent = self.dropout(self.relu(self.mlp_label_parent(output)))
        reduced_label_child = self.dropout(self.relu(self.mlp_label_child(output)))

        predicted_labels = y_pred_head.max(2)[1]

        # fuck this for now
        # else:
            # predicted_labels = []
            # for batch in y_pred_head: 
                # heads_softmaxes = F.softmax(batch, dim=1).cpu()
                # # if self.args.use_cuda:
                    # # heads_softmaxes = heads_softmaxes.cpu()

                # predicted_labels.append(torch.from_numpy(cle.mst(heads_softmaxes.data.numpy())))

            # predicted_labels = Variable(torch.stack(predicted_labels))
            # if self.args.use_cuda:
                # predicted_labels = predicted_labels.cuda()

        selected_heads = torch.stack([torch.index_select(reduced_label_parent[n], 0, predicted_labels[n])
                                        for n, _ in enumerate(predicted_labels)])
        y_pred_label = self.label_biaffine(selected_heads, reduced_label_child)
        y_pred_label = Helpers.extract_best_label_logits(predicted_labels, y_pred_label, form_pack).to(self.args.device)

        return y_pred_head, y_pred_label

    '''
    1. the bare minimum that needs to be loaded is forms, upos, head, deprel (could change later); load those
    2. initialise everything else to none; load it if necessary based on command line args
    3. pass everything, whether it's been loaded or not, to the forward function; if it's unnecessary it won't use it
    '''
    def train_(self, epoch, train_loader):
        self.train()
        train_loader.init_epoch()

        for i, batch in enumerate(train_loader):
            y_heads, y_deprels, y_langids = batch.head, batch.deprel, batch.misc
            elements_per_batch = len(y_heads)
            y_pred_heads, y_pred_deprels = self(batch)

            batch_size, longest_sentence_in_batch = y_heads.size()

            # * predictions: (B x S x S) => (B * S x S)
            # * heads: (B x S) => (B * S)
            y_pred_heads = y_pred_heads.view(batch_size * longest_sentence_in_batch, -1)
            y_heads = y_heads.contiguous().view(batch_size * longest_sentence_in_batch)

            # * predictions: (B x S x D) => (B * S x D)
            # * heads: (B x S) => (B * S)
            y_pred_deprels = y_pred_deprels.view(batch_size * longest_sentence_in_batch, -1)
            y_deprels = y_deprels.contiguous().view(batch_size * longest_sentence_in_batch)

            # sum losses
            train_loss = self.criterion(y_pred_heads, y_heads) + self.criterion(y_pred_deprels, y_deprels)

            self.zero_grad()
            train_loss.backward()
            self.optimiser.step()

            print("Epoch: {}\t{}/{}\tloss: {}".format(epoch, (i + 1) * elements_per_batch, len(train_loader.dataset), train_loss.item()))

    def evaluate_(self, test_loader, print_conll=False):
        las_correct, uas_correct, total = 0, 0, 0
        self.eval()
        for i, batch in enumerate(test_loader):
            form_pack, y_heads, y_deprels = batch.form[1], batch.head, batch.deprel
            
            y_pred_heads, y_pred_deprels = self(batch)
            y_pred_heads = y_pred_heads.max(2)[1]
            y_pred_deprels = y_pred_deprels.max(2)[1]

            mask = torch.zeros(form_pack.size()[0], max(form_pack)).type(torch.LongTensor)
            for n, size in enumerate(form_pack): mask[n, 0:size] = 1
            mask = mask.type(torch.ByteTensor).to(self.args.device)
            mask = Variable(mask)
            mask[0, 0] = 0

            heads_correct = ((y_heads == y_pred_heads) * mask)
            deprels_correct = ((y_deprels == y_pred_deprels) * mask)

            # excepts should never trigger; leave them in just in case
            try:
                uas_correct += heads_correct.nonzero().size(0)
            except RuntimeError:
                pass

            try:
                las_correct += (heads_correct * deprels_correct) .nonzero().size(0)
            except RuntimeError:
                pass

            total += mask.nonzero().size(0)

            if print_conll:
                deprel_vocab = self.vocabs['deprels']
                deprels = [deprel_vocab.itos[i.data[0]] for i in y_pred_deprels.view(-1, 1)]

                heads_softmaxes = self(batch)[0][0]
                heads_softmaxes = F.softmax(heads_softmaxes, dim=1)
                heads_softmaxes = heads_softmaxes.cpu()

                json = cle.mst(heads_softmaxes.data.numpy())

                Helpers.write_to_conllu(self.args.test, self.args.outfile, json, deprels, i)

        print("UAS = {}/{} = {}\nLAS = {}/{} = {}".format(uas_correct, total, uas_correct / total,
                                                          las_correct, total, las_correct / total))


class MTMapper(nn.Module):
    def __init__(self, args, vocabs, parser):
        super().__init__()

        self.args = args
        self.vocabs = vocabs
        self.src_parser = parser

        # construct vocabs for trg_parser
        trg_vocabs = self.src_parser.vocabs
        trg_vocabs['forms'] = self.vocabs['trg']

        self.trg_parser = Parser(args, trg_vocabs).to(self.args.device)

        self.optimiser = torch.optim.Adam(self.parameters(), lr=args.lr)
        self.criterion = nn.MSELoss()

    def sort(self, batch):
        sent, sent_len = batch
        sent_len_sorted, idx_sort = np.sort(sent_len)[::-1], np.argsort(-sent_len)
        idx_unsort = np.argsort(idx_sort)
        idx_sort = idx_sort.to(self.args.device)

        return (sent.index_select(0, idx_sort), np.sort(sent_len)[::-1]), idx_unsort

    def unsort(self, sent, idx_unsort):
        idx_unsort = idx_unsort.to(self.args.device)
        return sent.index_select(0, idx_unsort)

    def forward(self, batch):
        src, src_unsort = self.sort(batch.src)
        trg, trg_unsort = self.sort(batch.trg)

        with torch.no_grad():
            parsed_src, _ = self.src_parser(src, downstream=True)

        parsed_trg, _ = self.trg_parser(trg, downstream=True)

        assert parsed_src.size() == parsed_trg.size()

        return self.unsort(parsed_src, src_unsort), self.unsort(parsed_trg, trg_unsort)

    def train_(self, epoch, train_iterator):
        self.train()
        train_iterator.init_epoch()
        sqrt_crit = lambda u, v: torch.sqrt(self.criterion(u, v))

        for i, batch in enumerate(train_iterator):
            pad_len = max(batch.src[0].size(1), batch.trg[0].size(1))

            SRC_PAD = pad_len - batch.src[0].size(1)
            TRG_PAD = pad_len - batch.trg[0].size(1)
            SRC_PAD_TOKEN = self.vocabs['src'].stoi['<pad>']
            TRG_PAD_TOKEN = self.vocabs['trg'].stoi['<pad>']

            batch.src = (F.pad(batch.src[0], (0, SRC_PAD), value=SRC_PAD_TOKEN), batch.src[1])
            batch.trg = (F.pad(batch.trg[0], (0, TRG_PAD), value=TRG_PAD_TOKEN), batch.trg[1])
            
            assert batch.src[0].size(1) == batch.trg[0].size(1)

            parsed_src, parsed_trg = self(batch)
            loss = sqrt_crit(parsed_src, parsed_trg)

            self.optimiser.zero_grad()
            loss.backward()
            self.optimiser.step()

            if i % self.args.print_every == 0:
                print("aligner epoch: {}\t{}/{}\tloss={}".format(epoch, i * len(batch), len(train_iterator.dataset), loss.item()))
