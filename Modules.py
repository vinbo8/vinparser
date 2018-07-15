import math
import torch
from torch.autograd import Variable
import torch.nn.functional as F


class DomainShiftFunction(torch.autograd.Function):
    def forward(self, input):
        return input

    def backward(self, grad_output):
        grad_input = grad_output.clone()
        print("Hargle")
        return 0.8 * grad_input


class DomainShifter(torch.nn.Module):
    def forward(self, input):
        return DomainShiftFunction()(input)


class LangModel(torch.nn.Module):
    # pass the whole parser to access param
    def __init__(self, sizes, args, embed_dim=300, context_size=2):
        super().__init__()
        self.use_cuda = args.use_cuda

        self.embeds = torch.nn.Embedding(sizes['vocab'], embed_dim)
        self.dense1 = torch.nn.Linear(context_size * embed_dim, embed_dim // 2)
        self.dense2 = torch.nn.Linear(embed_dim // 2, sizes['vocab'])

    def forward(self, inputs):
        embeds = self.embeds(inputs).view((1, -1))
        out = F.relu(self.dense1(embeds))
        out = self.dense2(out)
        return F.log_softmax(out, dim=1)


class CharEmbedding(torch.nn.Module):
    def __init__(self, char_size, embed_dim, lstm_dim, lstm_layers):
        super().__init__()
        self.embedding_chars = torch.nn.Embedding(char_size, 100)
        self.lstm = torch.nn.LSTM(100, int(150), 1,
                                  batch_first=True, bidirectional=False, dropout=0.33)
        self.attention = LinearAttention(int(150))
        self.mlp = torch.nn.Linear(300, 100, bias=False)

    def forward(self, forms, pack_sent):
        # input: B x S x W
        batch_size, max_words, max_chars = forms.size()
        forms = forms.contiguous().view(batch_size * max_words, -1)
        pack = pack_sent.contiguous().view(batch_size * max_words)

        out = self.embedding_chars(forms)

        embeds, (_, c) = self.lstm(out)
        # embeds = embeds.contiguous().view(batch_size, max_words, max_chars, -1)
        embeds = self.attention(embeds).squeeze(dim=2)
        c = c[-1]
        out = torch.cat([embeds, c], dim=1)
        embed_mat = self.mlp(out).view(batch_size, max_words, -1)

        # embeds, _ = torch.nn.utils.rnn.pad_packed_sequence(embeds, batch_first=True)
        return embed_mat


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
        return input1.transpose(1, 2) @ soft
        out = input1.transpose(1, 2) @ soft
        return out

class Biaffine(torch.nn.Module):

    def __init__(self, in1_features, in2_features, batch_size):
        super(Biaffine, self).__init__()
        self.in1_features = in1_features
        self.in2_features = in2_features

        self.weight = torch.nn.Parameter(torch.rand(batch_size, in1_features, in2_features))
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


class ShorterBiaffine(torch.nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.in_features = in_features,
        self.weight = torch.nn.Parameter(torch.rand(in_features + 1, in_features, 1))
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
        dim1 += 1

        input1 = input1.contiguous().view(batch_size * len1, dim1)
        W = self.weight.transpose(1, 2).contiguous().view(dim1, dim1 - 1)
        affine = (input1 @ W).view(batch_size, len1, dim1 - 1)
        biaffine = (affine @ input2.transpose(1, 2)).view(batch_size, len1, 1, len1).transpose(2, 3).squeeze(3)

        return biaffine


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

