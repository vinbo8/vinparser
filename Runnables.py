import os
import torch
import pprint
import Helpers
from scripts import cle
from torch.autograd import Variable
from collections import Counter
import torch.nn.functional as F
from Modules import CharEmbedding, ShorterBiaffine, LongerBiaffine, LangModel


class Parser(torch.nn.Module):
    def __init__(self, args, sizes, vocab, embed_dim=100, lstm_dim=400, lstm_layers=3,
                 reduce_dim_arc=100, reduce_dim_label=100, learning_rate=1e-3):
        super().__init__()

        self.args = args
        self.vocab = vocab

        if self.args.use_chars:
            self.embeddings_chars = CharEmbedding(sizes['chars'], embed_dim, lstm_dim, lstm_layers)

        self.embeddings_rand = torch.nn.Embedding(sizes['forms'], embed_dim)
        self.embeddings_forms = torch.nn.Embedding(sizes['forms'], embed_dim)
        if self.args.embed:
            self.embeddings_forms.weight.data.copy_(vocab['forms'].vectors)
            if self.args.fix_embeds:
                self.embeddings_forms.weight.requires_grad = False

        self.embeddings_tags = torch.nn.Embedding(sizes['postags'], 100)
        self.embeddings_langids = torch.nn.Embedding(sizes['misc'], 100)

        # size should be embed_size + whatever the other embeddings have
        lstm_in_dim = 500 if self.args.use_misc else 400
        self.lstm = torch.nn.LSTM(lstm_in_dim, lstm_dim, lstm_layers,
                                  batch_first=True, bidirectional=True, dropout=0.33)
        self.mlp_head = torch.nn.Linear(2 * lstm_dim, reduce_dim_arc)
        self.mlp_dep = torch.nn.Linear(2 * lstm_dim, reduce_dim_arc)
        self.mlp_deprel_head = torch.nn.Linear(2 * lstm_dim, reduce_dim_label)
        # aha 
        self.mlp_deprel_weights = torch.nn.Linear(reduce_dim_label, reduce_dim_label, bias=False)
        self.mlp_deprel_dep = torch.nn.Linear(2 * lstm_dim, reduce_dim_label)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=0.33)
        self.biaffine = ShorterBiaffine(reduce_dim_arc)
        self.biaffine_for_weights = ShorterBiaffine(reduce_dim_arc)
        self.weight_biaffine = ShorterBiaffine(reduce_dim_arc)
        self.label_biaffine = LongerBiaffine(reduce_dim_label, reduce_dim_label, sizes['deprels'])

        # ======
        # for the pred_lang loss
        self.lang_pred_hidden = torch.nn.Linear(400, 100)
        self.lang_pred_out = torch.nn.Linear(100, sizes['misc'])
        # ======

        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self.weight_criterion = torch.nn.MSELoss()
        params = filter(lambda p: p.requires_grad, self.parameters())
        selective_params = [p[1] for p in filter(lambda p: p[0] == 'biaffine_for_weights.weight', self.named_parameters())]
        self.optimiser = torch.optim.Adam(params, lr=learning_rate, betas=(0.9, 0.9))
        self.selective_optimiser = torch.optim.Adam(selective_params, lr=learning_rate, betas=(0.9, 0.9))

        if self.args.use_cuda:
            self.biaffine.cuda()
            self.biaffine_for_weights.cuda()
            self.label_biaffine.cuda()

    def forward(self, batch, dev=False):
        chars, char_pack = None, None
        (forms, form_pack), tags, langids = batch.form, batch.upos, batch.misc

        composed_embeds = self.dropout(self.embeddings_rand(forms))
        if self.args.embed:
            composed_embeds += self.dropout(self.embeddings_forms(forms))
        if self.args.use_chars:
            (chars, _, char_pack) = batch.char
            composed_embeds += self.dropout(self.embeddings_chars(chars, char_pack))

        tag_embeds = self.dropout(self.embeddings_tags(tags))
        langid_embeds = self.dropout(self.embeddings_langids(langids))

        embeds = torch.cat([composed_embeds, tag_embeds], dim=2)
        if self.args.use_misc:
            embeds = torch.cat([embeds, langid_embeds], dim=2)
        # embeds = torch.cat([composed_embeds, tag_embeds, langid_embeds], dim=2)

        # pack/unpack for LSTM
        for_lstm = torch.nn.utils.rnn.pack_padded_sequence(embeds, form_pack.tolist(), batch_first=True)
        output, _ = self.lstm(for_lstm)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        # predict heads
        reduced_head_head = self.dropout(self.relu(self.mlp_head(output)))
        reduced_head_dep = self.dropout(self.relu(self.mlp_dep(output)))
        y_pred_head = self.biaffine(reduced_head_head, reduced_head_dep)
        y_pred_weights = self.biaffine_for_weights(reduced_head_head, reduced_head_head)

        # if not self.training:
        #     y_pred_head *= y_pred_weights

        # predict deprels using heads
        reduced_deprel_head = self.dropout(self.relu(self.mlp_deprel_head(output)))
        reduced_deprel_dep = self.dropout(self.relu(self.mlp_deprel_dep(output)))
        predicted_labels = []

        if self.training:
            predicted_labels = y_pred_head.max(2)[1]

        else:
            for batch in y_pred_head: 
                heads_softmaxes = F.softmax(batch, dim=1)
                if self.args.use_cuda:
                    heads_softmaxes = heads_softmaxes.cpu()

                predicted_labels.append(torch.from_numpy(cle.mst(heads_softmaxes.data.numpy())))

            predicted_labels = Variable(torch.stack(predicted_labels))
            if self.args.use_cuda:
                predicted_labels = predicted_labels.cuda()

        # batch_size, longest_word_in_batch, _ = y_pred_weights.size()
        # true_weights = Variable(torch.zeros(batch_size, longest_word_in_batch, longest_word_in_batch))
        # enum = torch.LongTensor([i for i in range(longest_word_in_batch)])
        # if self.args.use_cuda: 
        #     enum = enum.cuda()
        #     true_weights = true_weights.cuda()

        # for batch in range(batch_size):
        #     for n, i in enumerate(predicted_labels[batch].data):
        #         true_weights[batch, n, i] = 1

        selected_heads = torch.stack([torch.index_select(reduced_deprel_head[n], 0, predicted_labels[n])
                                        for n, _ in enumerate(predicted_labels)])
        y_pred_label = self.label_biaffine(selected_heads, reduced_deprel_dep)
        y_pred_label = Helpers.extract_best_label_logits(predicted_labels, y_pred_label, form_pack)
        if self.args.use_cuda:
            y_pred_label = y_pred_label.cuda()

        # lang pred bollix
        # langid = self.dropout(F.relu(self.lang_pred_hidden(embeds)))
        # y_pred_langid = self.lang_pred_out(langid)
        # if self.args.use_cuda:
            # y_pred_langid = y_pred_langid.cuda()

        return y_pred_head, y_pred_label, (None, None) # (y_pred_weights, true_weights)

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
            y_pred_heads, y_pred_deprels, (y_pred_weights, y_weights) = self(batch)

            # all_ones = Variable(torch.ones(y_pred_weights.size()[0:2]).type(torch.LongTensor))
            # if self.args.use_cuda:
                # all_ones = all_ones.cuda()
            # reshape for cross-entropy
            batch_size, longest_sentence_in_batch = y_heads.size()

            # * predictions: (B x S x S) => (B * S x S)
            # * heads: (B x S) => (B * S)
            y_pred_heads = y_pred_heads.view(batch_size * longest_sentence_in_batch, -1)
            y_heads = y_heads.contiguous().view(batch_size * longest_sentence_in_batch)

            # * predictions: (B x S x D) => (B * S x D)
            # * heads: (B x S) => (B * S)
            y_pred_deprels = y_pred_deprels.view(batch_size * longest_sentence_in_batch, -1)
            y_deprels = y_deprels.contiguous().view(batch_size * longest_sentence_in_batch)

            # langid
            # y_pred_langids = y_pred_langids.view(batch_size * longest_sentence_in_batch, -1)
            # y_langids = y_langids.contiguous().view(batch_size * longest_sentence_in_batch)

            # sum losses
            train_loss = self.criterion(y_pred_heads, y_heads) + self.criterion(y_pred_deprels, y_deprels) # + self.criterion(y_pred_langids, y_langids)
            # dev_loss = self.criterion(y_pred_weights.view(batch_size * longest_sentence_in_batch, -1), all_ones.view(batch_size * longest_sentence_in_batch))

            self.zero_grad()
            train_loss.backward()
            # dev_loss.backward()
            self.optimiser.step()
            # self.selective_optimiser.step()

            print("Epoch: {}\t{}/{}\tloss: {}".format(epoch, (i + 1) * elements_per_batch, len(train_loader.dataset), train_loss.data[0]))

    def evaluate_(self, test_loader, print_conll=False):
        las_correct, uas_correct, total = 0, 0, 0
        self.eval()
        for i, batch in enumerate(test_loader):
            form_pack, y_heads, y_deprels = batch.form[1], batch.head, batch.deprel
            
            y_pred_heads, y_pred_deprels, _ = self(batch)
            y_pred_heads = y_pred_heads.max(2)[1]
            y_pred_deprels = y_pred_deprels.max(2)[1]

            mask = torch.zeros(form_pack.size()[0], max(form_pack)).type(torch.LongTensor)
            for n, size in enumerate(form_pack): mask[n, 0:size] = 1
            mask = mask.type(torch.ByteTensor)
            mask = mask.cuda() if self.args.use_cuda else mask
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
                deprel_vocab = self.vocab['deprels']
                deprels = [deprel_vocab.itos[i.data[0]] for i in y_pred_deprels.view(-1, 1)]

                heads_softmaxes = self(batch)[0][0]
                heads_softmaxes = F.softmax(heads_softmaxes, dim=1)
                if self.args.use_cuda:
                    heads_softmaxes = heads_softmaxes.cpu()

                json = cle.mst(heads_softmaxes.data.numpy())

                Helpers.write_to_conllu(self.args.test, self.args.outfile, json, deprels, i)

        print("UAS = {}/{} = {}\nLAS = {}/{} = {}".format(uas_correct, total, uas_correct / total,
                                                          las_correct, total, las_correct / total))

    def hargle_(self, epoch, dev_loader):
        self.train()
        dev_loader.init_epoch()

        for i, batch in enumerate(dev_loader):
            _, _, (y_pred_weights, y_weights) = self(batch)

            # reshape for cross-entropy
            batch_size, longest_sentence_in_batch, _ = y_weights.size()
            dev_loss = self.weight_criterion(y_pred_weights, y_weights)

            self.zero_grad()
            dev_loss.backward()
            self.selective_optimiser.step()

            print("Epoch: {}\t{}/{}\tloss: {}".format(epoch, (i + 1) * batch_size, len(dev_loader.dataset), dev_loss.data[0]))


class LangID(torch.nn.Module):
    def __init__(self, args, sizes):
        super().__init__()
        self.word_embed = torch.nn.Embedding(sizes['word'], 300)
        self.lstm = torch.nn.LSTM(300, 300, batch_first=True, bidirectional=True)
        self.hidden = torch.nn.Linear(600, 300)
        self.out = torch.nn.Linear(300, sizes['lang'])

        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=2e-3)

    def forward(self, *input):
        words = input[0]
        embeddings = self.word_embed(words)
        lstm, _ = self.lstm(embeddings)
        l1 = F.relu(self.hidden(lstm))
        out = self.out(l1)

        return out

    def train_(self, epoch, train_loader):
        self.train()
        train_loader.init_epoch()

        for i, batch in enumerate(train_loader):
            (words, sizes), langs = batch.word, batch.lang
            sizes = sizes.tolist()

            mask = Helpers.get_mask(sizes)
            pred = self.forward(words, sizes)

            batch_size, max_len = langs.size()[0], langs.size()[1]
            langs = langs.view(-1)
            pred = pred.view(batch_size * max_len, -1)

            loss = self.criterion(pred, langs)

            self.zero_grad()
            loss.backward()
            self.optimizer.step()

            print("Epoch: {}\t{}/{}\tloss: {}".format(
                epoch, (i + 1) * len(words), len(train_loader.dataset), loss.data[0]))

    def evaluate_(self, loader):
        self.eval()
        correct, total = 0, 0
        for i, batch in enumerate(loader):
            (words, sizes), langs = batch.word, batch.lang
            sizes = sizes.tolist()

            pred = self.forward(words, sizes)

            batch_size, max_len = langs.size()[0], langs.size()[1]
            langs = langs.view(-1)
            pred = pred.view(batch_size * max_len, -1)

            correct += (pred.max(1)[1] == langs).nonzero().size(0)
            total += pred.size()[0]

        print("Accuracy: {}".format(correct / total))


class LangSwitch(torch.nn.Module):
    def __init__(self, args, sizes, vocab, chain=False, embeddings=None, embed_dim=100, lstm_dim=100, lstm_layers=3,
                 mlp_dim=100, learning_rate=1e-5):
        super().__init__()

        self.sizes, self.args, self.vocab = sizes, args, vocab
        self.form_embeds = torch.nn.Embedding(sizes['forms'], embed_dim)
        self.tag_embeds = torch.nn.Embedding(sizes['postags'], embed_dim)
        self.deprel_embeds = torch.nn.Embedding(sizes['deprels'], embed_dim)
        self.previous_langid_embeds = torch.nn.Embedding(sizes['misc'], embed_dim)
        self.vocab = vocab
        self.chain = chain
        if self.args.embed:
            self.embeds.weight.data.copy_(vocab[0].vectors)
        self.lstm = torch.nn.LSTM(2 * embed_dim, lstm_dim, lstm_layers, batch_first=True, bidirectional=True, dropout=0.5)
        self.relu = torch.nn.ReLU()
        self.mlp = torch.nn.Linear(2 * lstm_dim + embed_dim, mlp_dim)
        self.out = torch.nn.Linear(mlp_dim, sizes['misc'])

        self.conv_2 = torch.nn.Conv2d(1, 25, (2, 1))
        self.conv_3 = torch.nn.Conv2d(1, 25, (3, 1))
        self.conv_4 = torch.nn.Conv2d(1, 25, (4, 1))
        self.conv_5 = torch.nn.Conv2d(1, 25, (5, 1))

        self.conv_2_2 = torch.nn.Conv2d(25, 128, (2, 1))
        self.conv_3_2 = torch.nn.Conv2d(25, 128, (3, 1))
        self.conv_4_2 = torch.nn.Conv2d(25, 128, (4, 1))
        self.conv_5_2 = torch.nn.Conv2d(25, 128, (5, 1))

        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self.optimiser = torch.optim.Adam(self.parameters(), lr=learning_rate, betas=(0.9, 0.9))
        self.dropout = torch.nn.Dropout(p=0.2)

    def forward(self, batch):

        (forms, pack), tags, deprels, misc = batch.form, batch.upos, batch.deprel, batch.misc
        
        form_embeds = self.dropout(self.form_embeds(forms))
        tag_embeds = self.dropout(self.tag_embeds(tags))
        previous_langid = self.dropout(self.previous_langid_embeds(misc))

        embeds = torch.cat([form_embeds, tag_embeds, previous_langid], dim=2)
        embeds_2 = torch.cat([form_embeds, tag_embeds], dim=2)

        batch_size, longest_sent, embed_dim = embeds.size()
        # adaptive_pool_size = longest_sent // 3
        # embeds = embeds.view(batch_size, 1, longest_sent, embed_dim)

        # conv_2 = F.adaptive_max_pool2d(F.relu(self.conv_2(embeds)), (longest_sent, embed_dim)).squeeze(dim=3)
        # conv_3 = F.adaptive_max_pool2d(F.relu(self.conv_3(embeds)), (longest_sent, embed_dim)).squeeze(dim=3)
        # conv_4 = F.adaptive_max_pool2d(F.relu(self.conv_4(embeds)), (longest_sent, embed_dim)).squeeze(dim=3)
        # conv_5 = F.adaptive_max_pool2d(F.relu(self.conv_5(embeds)), (longest_sent, embed_dim)).squeeze(dim=3)

        # conv_2 = F.adaptive_max_pool2d(F.relu(self.conv_2_2(conv_2)), (longest_sent, 1)).squeeze(dim=3).transpose(1, 2)
        # conv_3 = F.adaptive_max_pool2d(F.relu(self.conv_3_2(conv_3)), (longest_sent, 1)).squeeze(dim=3).transpose(1, 2)
        # conv_4 = F.adaptive_max_pool2d(F.relu(self.conv_4_2(conv_4)), (longest_sent, 1)).squeeze(dim=3).transpose(1, 2)
        # conv_5 = F.adaptive_max_pool2d(F.relu(self.conv_5_2(conv_5)), (longest_sent, 1)).squeeze(dim=3).transpose(1, 2)
        # embeds = torch.cat([form_embeds, tag_embeds, previous_langid], dim=2)
        # packed = torch.nn.utils.rnn.pack_padded_sequence(embeds, pack.tolist(), batch_first=True)
        # lstm_out, _ = self.lstm(packed)
        # lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)

        packed = torch.nn.utils.rnn.pack_padded_sequence(embeds_2, pack.tolist(), batch_first=True)
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)

        hmm = torch.cat([lstm_out, previous_langid], dim=2)
        # for_mlp = torch.cat([conv_2, conv_3, conv_4, conv_5], dim=2)
        # mlp_out = self.dropout(self.relu(self.mlp(for_mlp)))

        y_pred = self.out(self.relu(self.mlp(hmm)))
        if self.args.use_cuda:
            y_pred = y_pred.cuda()

        return y_pred


    def train_(self, epoch, train_loader):
        self.train()
        train_loader.init_epoch()

        for i, batch in enumerate(train_loader):
            y_misc = batch.misc
            batch_size = y_misc.size()[0]

            pad_misc_tensor = Variable(torch.LongTensor(batch_size, 1))
            if self.args.use_cuda:
                pad_misc_tensor = pad_misc_tensor.cuda()

            pad_misc_tensor[:] = self.vocab['misc'].stoi['<pad>']
            shifted_y_misc = torch.cat([y_misc[:, 1:], pad_misc_tensor], dim=1) 

            y_pred_misc = self(batch)

            # reshape for cross-entropy
            batch_size, longest_sentence_in_batch = y_misc.size()

            # predictions: (B x S x S) => (B * S x S)
            # heads: (B x S) => (B * S)
            y_pred_misc = y_pred_misc.view(batch_size * longest_sentence_in_batch, -1)
            shifted_y_misc = shifted_y_misc.contiguous().view(batch_size * longest_sentence_in_batch)

            # sum losses
            train_loss = self.criterion(y_pred_misc, shifted_y_misc)

            self.zero_grad()
            train_loss.backward()
            self.optimiser.step()

            print("Epoch: {}\t{}/{}\tloss: {}".format(epoch, (i + 1) * batch_size, len(train_loader.dataset), train_loss.data[0]))

    def evaluate_(self, test_loader, print_conll=False):
        correct, total = 0, 0
        self.eval()

        for i, batch in enumerate(test_loader):
            form_pack, y_misc = batch.form[1], batch.misc
            batch_size = y_misc.size()[0]

            pad_misc_tensor = Variable(torch.LongTensor(batch_size, 1))
            if self.args.use_cuda:
                pad_misc_tensor = pad_misc_tensor.cuda()

            pad_misc_tensor[:] = self.vocab['misc'].stoi['<pad>']
            shifted_y_misc = torch.cat([y_misc[:, 1:], pad_misc_tensor], dim=1) 

            # get tags
            y_pred = self(batch).max(2)[1]

            mask = torch.zeros(form_pack.size()[0], max(form_pack)).type(torch.LongTensor)
            for n, size in enumerate(form_pack): mask[n, 0:size] = 1

            mask = Variable(mask.type(torch.ByteTensor))
            if self.args.use_cuda:
                mask = mask.cuda()

            try:
                correct += ((shifted_y_misc == y_pred) * mask).nonzero().size(0)
            except:
                pass

            total += mask.nonzero().size(0)

        print("Accuracy = {}/{} = {}".format(correct, total, (correct / total)))
