import math
import torch
from torch.autograd import Variable
import torch.nn.functional as F

BATCH_SIZE = 50


class CharEmbedding(torch.nn.Module):
    def __init__(self, sizes, embedding_dim, lstm_dim, lstm_depth):
        super().__init__()
        self.embedding_chars = torch.nn.Embedding(sizes['chars'], embedding_dim)
        self.lstm = torch.nn.LSTM(embedding_dim, lstm_dim, lstm_depth,
                                  batch_first=True, bidirectional=False, dropout=0.33)
        self.attention = LinearAttention(lstm_dim)

    def forward(self, forms, pack_sent):
        # input: B x S x W
        batch_size, max_words, max_chars = forms.size()
        forms = forms.contiguous().view(batch_size * max_words, -1)
        indexes = (forms == 0).sum(dim=1).type(torch.LongTensor)
        y, indexes = torch.sort(indexes, 0)
        temp = forms[indexes]

        restore = temp[np.argsort(indexes.data)]
        assert restore.data.tolist() == forms.data.tolist()
        forms.size()
        out = self.embedding_chars(forms)
        pack = (temp != 0).sum(dim=1)
        pack[pack == 0] = 1

        # embeds = torch.nn.utils.rnn.pack_padded_sequence(out, pack.data.tolist(), batch_first=True)
        embeds, (_, c) = self.lstm(out)
        # embeds = embeds.contiguous().view(batch_size, max_words, max_chars, -1)
        embeds = self.attention(embeds)
        c = c[:, -1, :]
        # embeds, _ = torch.nn.utils.rnn.pad_packed_sequence(embeds, batch_first=True)

        return embeds


class LinearAttention(torch.nn.Module):

    def __init__(self, lstm_features):
        super().__init__()
        self.lstm_features = lstm_features

        self.weight = torch.nn.Parameter(torch.rand(self.lstm_features, 1))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(0))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input1):
        soft = F.softmax(input1 @ self.weight, dim=0)
        return input1.transpose(0, 1) @ soft


class Biaffine(torch.nn.Module):

    def __init__(self, in1_features, in2_features):
        super(Biaffine, self).__init__()
        self.in1_features = in1_features
        self.in2_features = in2_features

        self.weight = torch.nn.Parameter(torch.rand(BATCH_SIZE, in1_features, in2_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(0))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input1, input2):
        is_cuda = next(self.parameters()).is_cuda
        batch_size, len1, dim1 = input1.size()
        ones = torch.ones(batch_size, len1, 1)
        if is_cuda:
            ones = ones.cuda()
        input1 = torch.cat((input1, Variable(ones)), dim=2)

        biaffine = input1 @ self.weight @ input2.transpose(1, 2)
        return biaffine

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + 'in1_features=' + str(self.in1_features) \
            + ', in2_features=' + str(self.in2_features) \
            + ', out_features=' + str(self.out_features) + ')'


class RowBiaffine(torch.nn.Module):
    def __init__(self, in1_features, in2_features, dep_labels):
        super().__init__()
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.dep_labels = dep_labels
        self.weight = torch.nn.Parameter(torch.rand(dep_labels, in1_features, in2_features))
        self.bias = torch.nn.Parameter(torch.rand(dep_labels))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(0))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input1, input2):
        batch_size, sent_len, dim = input1.size()
        '''
        S = []
        for batch in range(batch_size):
            s_i = []
            for word in range(sent_len):
                h_head = input1[batch, word].view(1, -1)
                h_dep = input2[batch, word]
                s_i.append(h_head @ self.weight @ h_dep)
            s_i = torch.stack(s_i)
            S.append(s_i)
        S = torch.stack(S)
        return S.squeeze(3)
        '''
        return (input1 @ self.weight).transpose(1, 2) @ input2

    def forward_(self, input1, input2):
        batch_size, sent_len, dim = input1.size()
        S = []
        for batch in range(batch_size):
            s_i = []
            for word in range(sent_len):
                h_head = input1[batch, word].view(1, -1)
                h_dep = input2[batch, word]
                s_i.append(h_head @ self.weight @ h_dep)
            s_i = torch.stack(s_i)
            S.append(s_i)
        S = torch.stack(S)
        return S.squeeze(3)


class LongerBiaffine(torch.nn.Module):
    def __init__(self, in1_features, in2_features, dep_labels):
        super().__init__()
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.dep_labels = dep_labels
        self.weight = torch.nn.Parameter(torch.rand(in1_features + 1, in2_features + 1, dep_labels))
        self.bias = torch.nn.Parameter(torch.rand(dep_labels))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(0))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input1, input2):
        is_cuda = next(self.parameters()).is_cuda
        batch_size, len1, dim1 = input1.size()
        batch_size, len2, dim2 = input2.size()
        ones = torch.ones(batch_size, len1, 1)
        if is_cuda:
            ones = ones.cuda()
        input1 = torch.cat((input1, Variable(ones)), dim=2)
        input2 = torch.cat((input2, Variable(ones)), dim=2)
        dim1 += 1
        dim2 += 1
        input1 = input1.view(batch_size * len1, dim1)
        weight = self.weight.transpose(1, 2).contiguous().view(dim1, self.dep_labels * dim2)
        affine = (input1 @ weight).view(batch_size, len1 * self.dep_labels, dim2)
        biaffine = (affine @ input2.transpose(1, 2)).view(batch_size, len1, self.dep_labels, len2).transpose(2, 3)
        biaffine += self.bias.expand_as(biaffine)
        return biaffine

