import os
import torch
import pprint
import Helpers
from scripts import cle
from torch.autograd import Variable
from collections import Counter
import torch.nn.functional as F
from Modules import CharEmbedding, ShorterBiaffine, LongerBiaffine, LangModel


class Analyser(torch.nn.Module):
    def __init__(self, sizes, args, vocab, chain=False, embeddings=None, embed_dim=100, lstm_dim=100, lstm_layers=3,
                 mlp_dim=100, learning_rate=1e-5):
        super().__init__()
        self.cuda = args.use_cuda

        self.morph_vocab = vocab[3]
        self.feat_vocab = []
        for i in self.morph_vocab.itos:
            if "=" not in i:
                self.feat_vocab.append(i)
            else:
                self.feat_vocab.extend([j.split("=")[0] for j in i.split("|")])
        # TODO: could use a Vocab object but don't care right now
        self.feat_vocab = set(self.feat_vocab)
        self.feat_vocab_itos = list(self.feat_vocab)
        self.feat_vocab_stoi = {i: n for (n, i) in enumerate(self.feat_vocab_itos)}

        # components
        self.embeds = torch.nn.Embedding(sizes['vocab'], embed_dim)
        self.lstm = torch.nn.LSTM(embed_dim, lstm_dim, lstm_layers, batch_first=True, bidirectional=True, dropout=0.5)
        self.mlp = torch.nn.Linear(2 * lstm_dim, mlp_dim)
        self.out = torch.nn.Linear(mlp_dim, len(self.feat_vocab))

        self.optimiser = torch.optim.Adam(self.parameters(), lr=learning_rate, betas=(0.9, 0.9))
        self.criterion = torch.nn.BCEWithLogitsLoss()

    def forward(self, x_forms, pack):
        embeds = F.dropout(self.embeds(x_forms), p=0.33, training=self.training)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embeds, pack.tolist(), batch_first=True)
        lstm_out, _ = self.lstm(packed) 
        # TODO: try adding pad_value to match the loss pad value
        lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        mlp_out = F.dropout(F.relu(self.mlp(lstm_out)), p=0.33, training=self.training)
        out_pred = self.out(mlp_out)
        if self.cuda:
            out_pred = out_pred.cuda()

        return out_pred
    
    def train_(self, epoch, train_loader):
        self.train()
        train_loader.init_epoch()

        for i, batch in enumerate(train_loader):
            (x_forms, pack), x_tags = batch.form, batch.upos
            new_batch_tensor = Helpers.extract_batch_bucket_vector(batch, self.morph_vocab, self.feat_vocab_itos, self.feat_vocab_stoi)
            predicted_tensor = self.forward(x_forms, pack)

            train_loss = self.criterion(predicted_tensor, new_batch_tensor.type(torch.FloatTensor))

            self.zero_grad()
            train_loss.backward()
            self.optimiser.step()

            print("Epoch: {}\t{}/{}\tloss: {}".format(
                epoch, (i + 1) * len(x_forms), len(train_loader.dataset), train_loss.data[0]))


class Tagger(torch.nn.Module):
    def __init__(self, sizes, args, vocab, chain=False, embeddings=None, embed_dim=100, lstm_dim=100, lstm_layers=3,
                 mlp_dim=100, learning_rate=1e-5):
        super().__init__()

        self.embeds = torch.nn.Embedding(sizes['vocab'], embed_dim)
        self.compress = torch.nn.Linear(300,100)
        self.use_cuda = args.use_cuda
        self.save = args.save
        self.vocab = vocab
        self.test_file = args.test
        self.chain = chain
        if args.embed:
            self.embeds.weight.data.copy_(vocab[0].vectors)
        self.lstm = torch.nn.LSTM(100, lstm_dim, lstm_layers, batch_first=True, bidirectional=True, dropout=0.5)
        self.relu = torch.nn.ReLU()
        self.mlp = torch.nn.Linear(2 * lstm_dim, mlp_dim)
        self.out = torch.nn.Linear(mlp_dim, sizes['postags'])
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, betas=(0.9, 0.9))
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, forms, pack):
        # embeds + dropout
        form_embeds = self.dropout(self.embeds(forms))
        form_embeds = self.relu(self.compress(form_embeds))

        # pack/unpack for LSTM
        packed = torch.nn.utils.rnn.pack_padded_sequence(form_embeds, pack.tolist(), batch_first=True)
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)

        # LSTM => dense ReLU
        mlp_out = self.dropout(self.relu(self.mlp(lstm_out)))

        # reduce to dim no_of_tags
        y_pred = self.out(mlp_out)
        if self.cuda:
            y_pred = y_pred.cuda()

        return y_pred

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
        
        if self.save:
            if not os.path.exists(self.save):
                os.makedirs(self.save)
            with open(os.path.join(self.save, 'tagger.pt'), "wb") as f:
                torch.save(self.state_dict(), f)

    def evaluate_(self, test_loader, print_conll=False):
        correct, total = 0, 0
        self.eval()

        tag_tensors = [i.upos for i in test_loader]
        for i, batch in enumerate(test_loader):
            (x_forms, pack), x_tags, y_heads, y_deprels = batch.form, batch.upos, batch.head, batch.deprel

            mask = torch.zeros(pack.size()[0], max(pack)).type(torch.LongTensor)
            for n, size in enumerate(pack):
                mask[n, 0:size] = 1

            # get tags
            y_pred = self(x_forms, pack).max(2)[1]

            mask = Variable(mask.type(torch.ByteTensor))
            if self.cuda:
                mask = mask.cuda()

            correct += ((x_tags == y_pred) * mask).nonzero().size(0)

            total += mask.nonzero().size(0)

            if print_conll:
                tag_vocab = self.vocab[2]
                tags = [tag_vocab.itos[i.data[0]] for i in y_pred.view(-1, 1)]
                Helpers.write_tags_to_conllu(self.test_file, tags, i)

        print("Accuracy = {}/{} = {}".format(correct, total, (correct / total)))
        if self.chain: return tag_tensors


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
        # predicted_labels = y_pred_head.max(2)[1]

        for batch in y_pred_head: 
            heads_softmaxes = F.softmax(batch, dim=1)
            if self.args.use_cuda:
                heads_softmaxes = heads_softmaxes.cpu()

            predicted_labels.append(cle.mst(heads_softmaxes.data.numpy()))


        # batch_size, longest_word_in_batch, _ = y_pred_weights.size()
        # true_weights = Variable(torch.zeros(batch_size, longest_word_in_batch, longest_word_in_batch))
        # enum = torch.LongTensor([i for i in range(longest_word_in_batch)])
        # if self.args.use_cuda: 
        #     enum = enum.cuda()
        #     true_weights = true_weights.cuda()

        # for batch in range(batch_size):
        #     for n, i in enumerate(predicted_labels[batch].data):
        #         true_weights[batch, n, i] = 1

        predicted_labels = torch.stack(predicted_labels) 
        selected_heads = torch.stack([torch.index_select(reduced_deprel_head[n], 0, predicted_labels[n])
                                        for n, _ in enumerate(predicted_labels)])
        y_pred_label = self.label_biaffine(selected_heads, reduced_deprel_dep)
        y_pred_label = Helpers.extract_best_label_logits(predicted_labels, y_pred_label, form_pack)
        if self.args.use_cuda:
            y_pred_label = y_pred_label.cuda()

        # lang pred bollix
        langid = self.dropout(F.relu(self.lang_pred_hidden(embeds)))
        y_pred_langid = self.lang_pred_out(langid)
        if self.args.use_cuda:
            y_pred_langid = y_pred_langid.cuda()

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

            # predictions: (B x S x S) => (B * S x S)
            # heads: (B x S) => (B * S)
            y_pred_heads = y_pred_heads.view(batch_size * longest_sentence_in_batch, -1)
            y_heads = y_heads.contiguous().view(batch_size * longest_sentence_in_batch)

            # predictions: (B x S x D) => (B * S x D)
            # heads: (B x S) => (B * S)
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
                heads_softmaxes = F.softmax(heads_softmaxes, dim=0)
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



class CLTagger(torch.nn.Module):
    def __init__(self, args, main_sizes, aux_sizes, main_embeds, aux_embeds, embed_dim=100, lstm_dim=100, lstm_layers=2,
                 mlp_dim=100, learning_rate=1e-5):

        super().__init__()

        #Load pretrained embeds
        self.embeds_main = torch.nn.Embedding(main_sizes['vocab'], embed_dim)
        self.embeds_aux = torch.nn.Embedding(aux_sizes['vocab'], embed_dim)

        if args.embed:
            self.embeds_main.weight.data.copy_(main_embeds.vectors)
            self.embeds_aux.weight.data.copy_(aux_embeds.vectors)

        #Pass through shared then individual LSTMs
        self.lstm_shared = torch.nn.LSTM(embed_dim, lstm_dim, lstm_layers, batch_first=True, bidirectional=True, dropout=0.5)
        self.lstm_main = torch.nn.LSTM(lstm_dim * 2, lstm_dim, lstm_layers, batch_first=True, bidirectional=True, dropout=0.5)
        self.lstm_aux = torch.nn.LSTM(lstm_dim * 2, lstm_dim, lstm_layers, batch_first=True, bidirectional=True, dropout=0.5)

        #Pass through individual MLPs
        self.relu = torch.nn.ReLU()
        self.mlp_main = torch.nn.Linear(lstm_dim * 2, mlp_dim)
        self.mlp_aux = torch.nn.Linear(lstm_dim * 2, mlp_dim)
        #Outs
        self.out_main = torch.nn.Linear(mlp_dim, main_sizes['postags'])
        self.out_aux = torch.nn.Linear(mlp_dim, aux_sizes['postags'])
        #Losses
        self.criterion_main = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self.criterion_aux = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, betas=(0.9, 0.9))
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, forms, pack, type_task):
        if type_task == "main":
            return self.forward_main(forms, pack)
        elif type_task == "aux":
            return self.forward_aux(forms, pack)
        else:
            raise TypeError

    def forward_main(self, forms, pack):
        # embeds + dropout
        form_embeds = self.dropout(self.embeds_main(forms))

        # pack/unpack for LSTM
        packed = torch.nn.utils.rnn.pack_padded_sequence(form_embeds, pack.tolist(), batch_first=True)
        lstm_out, _ = self.lstm_shared(packed)
        lstm_out_main, _ = self.lstm_main(lstm_out)
        lstm_out_main, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out_main, batch_first=True)

        # LSTM => dense ReLU
        mlp_out = self.dropout(self.relu(self.mlp_main(lstm_out_main)))

        # reduce to dim no_of_tags
        return self.out_main(mlp_out)

    def forward_aux(self, forms, pack):
        # embeds + dropout
        form_embeds = self.dropout(self.embeds_aux(forms))

        # pack/unpack for LSTM
        packed = torch.nn.utils.rnn.pack_padded_sequence(form_embeds, pack.tolist(), batch_first=True)
        lstm_out, _ = self.lstm_shared(packed)
        lstm_out_aux, _ = self.lstm_aux(lstm_out)
        lstm_out_aux, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out_aux, batch_first=True)

        # LSTM => dense ReLU
        mlp_out = self.dropout(self.relu(self.mlp_aux(lstm_out_aux)))

        # reduce to dim no_of_tags
        return self.out_aux(mlp_out)

    def train_(self, epoch, train_loader, type_task="main"):
        self.train()
        train_loader.init_epoch()

        for i, batch in enumerate(train_loader):
            (x_forms, pack), x_tags, y_heads, y_deprels = batch.form, batch.upos, batch.head, batch.deprel

            mask = torch.zeros(pack.size()[0], max(pack)).type(torch.LongTensor)
            for n, size in enumerate(pack):
                mask[n, 0:size] = 1

            y_pred = self(x_forms, pack, type_task)
            # reshape for cross-entropy
            batch_size, longest_sentence_in_batch = x_forms.size()

            # predictions: (B x S x T) => (B * S, T)
            # heads: (B x S) => (B * S)
            y_pred = y_pred.view(batch_size * longest_sentence_in_batch, -1)
            x_tags = x_tags.contiguous().view(batch_size * longest_sentence_in_batch)

            if type_task == "aux":
                train_loss = self.criterion_aux(y_pred, x_tags)
            else:
                train_loss = self.criterion_main(y_pred, x_tags)

            self.zero_grad()
            train_loss.backward()
            self.optimizer.step()

            print("Epoch: {}\t{}/{}\tloss: {}".format(
                epoch, (i + 1) * len(x_forms), len(train_loader.dataset), train_loss.data))

    def evaluate_(self, test_loader, type_task="main"):
        correct, total = 0, 0
        self.eval()
        for i, batch in enumerate(test_loader):
            (x_forms, pack), x_tags, y_heads, y_deprels = batch.form, batch.upos, batch.head, batch.deprel

            mask = torch.zeros(pack.size()[0], max(pack)).type(torch.LongTensor)
            for n, size in enumerate(pack):
                mask[n, 0:size] = 1

                # get tags
            y_pred = self(x_forms, pack, type_task).max(2)[1]
            mask = Variable(mask.type(torch.ByteTensor))

            correct += ((x_tags == y_pred) * mask).nonzero().size(0)

            total += mask.nonzero().size(0)

        print("Accuracy = {}/{} = {}".format(correct, total, (correct / total)))


class CSParser(torch.nn.Module):
    def __init__(self, sizes, args, vocab, embeddings=None, embed_dim=100, lstm_dim=400, lstm_layers=3,
                 reduce_dim_arc=100, reduce_dim_label=100, learning_rate=1e-3):
        super().__init__()

        self.use_cuda = args.use_cuda
        self.use_chars = args.use_chars
        self.random_bs = args.random_bs

        if self.use_chars:
            self.embeddings_chars = CharEmbedding(sizes['chars'], embed_dim, lstm_dim, lstm_layers)

        self.embeddings_forms = torch.nn.Embedding(sizes['vocab'], embed_dim)
        if args.embed: self.embeddings_forms.weight.data.copy_(embeddings)
        self.embeddings_tags = torch.nn.Embedding(sizes['postags'], embed_dim)
        self.lstm = torch.nn.LSTM(2 * embed_dim, lstm_dim, lstm_layers,
                                  batch_first=True, bidirectional=True, dropout=0.33)
        self.mlp_head = torch.nn.Linear(2 * lstm_dim, reduce_dim_arc)
        self.mlp_dep = torch.nn.Linear(2 * lstm_dim, reduce_dim_arc)
        self.mlp_deprel_head = torch.nn.Linear(2 * lstm_dim, reduce_dim_label)
        self.mlp_deprel_dep = torch.nn.Linear(2 * lstm_dim, reduce_dim_label)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=0.33)
        # self.biaffine = Biaffine(reduce_dim_arc + 1, reduce_dim_arc, BATCH_SIZE)
        self.biaffine = ShorterBiaffine(reduce_dim_arc)
        self.label_biaffine = LongerBiaffine(reduce_dim_label, reduce_dim_label, sizes['deprels'])
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self.optimiser = torch.optim.Adam(self.parameters(), lr=learning_rate, betas=(0.9, 0.9))

        # ======================
        # for the language model
        # ======================
        self.lang_model_criterion = torch.nn.NLLLoss()
        self.lstm_lm = torch.nn.LSTM(embed_dim, lstm_dim, 1, batch_first=True, bidirectional=False, dropout=0.33)
        self.dense1 = torch.nn.Linear(2 * lstm_dim, lstm_dim // 2)
        self.dense2 = torch.nn.Linear(lstm_dim // 2, sizes['vocab'])

        # ==============
        # for the langid
        # ==============
        self.lstm_langid = torch.nn.LSTM(embed_dim, lstm_dim, lstm_layers, batch_first=True, bidirectional=True, dropout=0.5)
        self.relu_langid = torch.nn.ReLU()
        self.mlp_langid = torch.nn.Linear(2 * lstm_dim, 300)
        self.out_langid = torch.nn.Linear(300, sizes['langs'])
        self.dropout_langid = torch.nn.Dropout(p=0.5)
        self.criterion_langid = torch.nn.CrossEntropyLoss(ignore_index=-1)

        if self.use_cuda:
            self.biaffine.cuda()
            self.label_biaffine.cuda()

    def forward_aux(self, forms):
        batch_size = forms.size()[0]
        embeds = self.embeddings_forms(forms)
        lstm = self.lstm_lm(embeds)[0].contiguous().view((batch_size, -1))
        # embeds = embeds.view((batch_size, -1))
        out = F.relu(self.dense1(lstm))
        out = self.dense2(out)
        return F.log_softmax(out, dim=1)

    def forward(self, forms, tags, pack, chars, char_pack):
        form_embeds = self.dropout(self.embeddings_forms(forms))
        tag_embeds = self.dropout(self.embeddings_tags(tags))
        composed_embeds = form_embeds

        # =========================
        # aux task - lang modelling
        # =========================
        if self.use_chars:
            composed_embeds += self.dropout(self.embeddings_chars(chars, char_pack))

        embeds = torch.cat([composed_embeds, tag_embeds], dim=2)

        # pack/unpack for LSTM
        embeds = torch.nn.utils.rnn.pack_padded_sequence(embeds, pack.tolist(), batch_first=True)
        output, _ = self.lstm(embeds)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        # ===============
        # lang pred
        # ===============
        langid_embeds = torch.nn.utils.rnn.pack_padded_sequence(composed_embeds, pack.tolist(), batch_first=True)
        langid_out, _ = self.lstm_langid(langid_embeds)
        langid_out, _ = torch.nn.utils.rnn.pad_packed_sequence(langid_out, batch_first=True)
        langid_out = self.out_langid(self.dropout_langid(F.relu(self.mlp_langid(langid_out))))

        # predict heads
        reduced_head_head = self.dropout(self.relu(self.mlp_head(output)))
        reduced_head_dep = self.dropout(self.relu(self.mlp_dep(output)))
        y_pred_head = self.biaffine(reduced_head_head, reduced_head_dep)

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

        return y_pred_head, y_pred_label, langid_out

    '''
    1. the bare minimum that needs to be loaded is forms, upos, head, deprel (could change later); load those
    2. initialise everything else to none; load it if necessary based on command line args
    3. pass everything, whether it's been loaded or not, to the forward function; if it's unnecessary it won't use it
    '''
    def train_(self, epoch, train_loader, task_type="main"):
        self.train()
        train_loader.init_epoch()

        for i, batch in enumerate(train_loader):
            if task_type == "aux":
                x_forms = batch.form[0][:, 1:3]
                y_forms = batch.form[0][:, 3]
                y_out = self.forward_aux(x_forms)
                loss = self.lang_model_criterion(y_out, y_forms)
                self.zero_grad()
                loss.backward()
                self.optimiser.step()

                print("Epoch: {}\t{}/{}\tloss: {}".format(epoch, (i + 1) * len(x_forms),
                                                          len(train_loader.dataset), loss.data[0]))

            else:
                chars, length_per_word_per_sent = None, None
                (x_forms, pack), x_tags, y_heads, y_deprels, y_langs = batch.form, batch.upos, \
                                                                       batch.head, batch.deprel, batch.misc

                # TODO: add something similar for semtags
                if self.use_chars:
                    (chars, _, length_per_word_per_sent) = batch.char

                y_pred_head, y_pred_deprel, langid_out = self(x_forms, x_tags, pack, chars, length_per_word_per_sent)

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

                # for langid
                y_pred_langid = langid_out.view(batch_size * longest_sentence_in_batch, -1)
                y_langs = y_langs.contiguous().view(batch_size * longest_sentence_in_batch)

                # sum losses
                train_loss = self.criterion(y_pred_head, y_heads) + self.criterion(y_pred_deprel, y_deprels) - \
                             self.criterion_langid(y_pred_langid, y_langs)

                self.zero_grad()
                train_loss.backward()
                self.optimiser.step()

                print("Epoch: {}\t{}/{}\tloss: {}".format(epoch, (i + 1) * len(x_forms), len(train_loader.dataset), train_loss.data[0]))

                if self.random_bs and 'mtl' in self.random_bs:
                    return self.embeddings_forms

    def evaluate_(self, test_loader):
        las_correct, uas_correct, total = 0, 0, 0
        self.eval()
        for i, batch in enumerate(test_loader):
            chars, length_per_word_per_sent = None, None
            (x_forms, pack), x_tags, y_heads, y_deprels, langtags = \
                batch.form, batch.upos, batch.head, batch.deprel, batch.misc

            # TODO: add something similar for semtags
            if self.use_chars:
                (chars, _, length_per_word_per_sent) = batch.char

            mask = torch.zeros(pack.size()[0], max(pack)).type(torch.LongTensor)
            for n, size in enumerate(pack):
                mask[n, 0:size] = 1

            # get labels
            # TODO: ensure well-formed tree
            if self.random_bs and self.random_bs[0] == 'weight':
                head, deprel, _ = self(x_forms, x_tags, pack, chars, length_per_word_per_sent)
                y_pred_head = (Helpers.softmax_weighter(batch.misc) * head).max(2)[1]
                y_pred_deprel = deprel.max(2)[1]
            else:
                y_pred_head, y_pred_deprel = [i.max(2)[1] for i in
                                            self(x_forms, x_tags, pack, chars, length_per_word_per_sent)[:2]]

            mask = mask.type(torch.ByteTensor)
            if self.use_cuda:
                mask = mask.cuda()

            mask = Variable(mask)
            heads_correct = ((y_heads == y_pred_head) * mask)
            deprels_correct = ((y_deprels == y_pred_deprel) * mask)

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

        print("UAS = {}/{} = {}\nLAS = {}/{} = {}".format(uas_correct, total, uas_correct / total,
                                                          las_correct, total, las_correct / total))


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
    def __init__(self, sizes, args, vocab, chain=False, embeddings=None, embed_dim=100, lstm_dim=100, lstm_layers=3,
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
        self.lstm = torch.nn.LSTM(1 * embed_dim, lstm_dim, lstm_layers, batch_first=True, bidirectional=True, dropout=0.5)
        self.relu = torch.nn.ReLU()
        self.mlp = torch.nn.Linear(512, mlp_dim)
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
        deprel_embeds = self.dropout(self.deprel_embeds(deprels))
        previous_langid = self.dropout(self.previous_langid_embeds(misc))

        embeds = torch.cat([form_embeds, tag_embeds, previous_langid], dim=2)

        batch_size, longest_sent, embed_dim = embeds.size()
        adaptive_pool_size = longest_sent // 3
        embeds = embeds.view(batch_size, 1, longest_sent, embed_dim)

        conv_2 = F.adaptive_max_pool2d(F.relu(self.conv_2(embeds)), (longest_sent, embed_dim)).squeeze(dim=3)
        conv_3 = F.adaptive_max_pool2d(F.relu(self.conv_3(embeds)), (longest_sent, embed_dim)).squeeze(dim=3)
        conv_4 = F.adaptive_max_pool2d(F.relu(self.conv_4(embeds)), (longest_sent, embed_dim)).squeeze(dim=3)
        conv_5 = F.adaptive_max_pool2d(F.relu(self.conv_5(embeds)), (longest_sent, embed_dim)).squeeze(dim=3)

        conv_2 = F.adaptive_max_pool2d(F.relu(self.conv_2_2(conv_2)), (longest_sent, 1)).squeeze(dim=3).transpose(1, 2)
        conv_3 = F.adaptive_max_pool2d(F.relu(self.conv_3_2(conv_3)), (longest_sent, 1)).squeeze(dim=3).transpose(1, 2)
        conv_4 = F.adaptive_max_pool2d(F.relu(self.conv_4_2(conv_4)), (longest_sent, 1)).squeeze(dim=3).transpose(1, 2)
        conv_5 = F.adaptive_max_pool2d(F.relu(self.conv_5_2(conv_5)), (longest_sent, 1)).squeeze(dim=3).transpose(1, 2)
        # embeds = torch.cat([form_embeds, tag_embeds, previous_langid], dim=2)
        # packed = torch.nn.utils.rnn.pack_padded_sequence(previous_langid, pack.tolist(), batch_first=True)
        # lstm_out, _ = self.lstm(packed)
        # lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)

        for_mlp = torch.cat([conv_2, conv_3, conv_4, conv_5], dim=2)
        mlp_out = self.dropout(self.relu(self.mlp(for_mlp)))

        y_pred = self.out(mlp_out)
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
