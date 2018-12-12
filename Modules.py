import math
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn


class ParallelLSTM(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.lstm = nn.LSTM(*args, **kwargs)

    def forward(self, embeds, forms, form_pack):
        total_length = forms.size(1)
        for_lstm = nn.utils.rnn.pack_padded_sequence(embeds, form_pack.tolist(), batch_first=True)
        output, _ = self.lstm(for_lstm)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True, total_length=total_length)

        return output


class ShorterBiaffine(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.in_features = in_features,
        self.weight = nn.Parameter(torch.rand(in_features + 1, in_features, 1))
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


class LongerBiaffine(nn.Module):
    def __init__(self, in1_features, in2_features, dep_labels):
        super().__init__()
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.dep_labels = dep_labels
        self.weight = nn.Parameter(torch.rand(in1_features + 1, in2_features + 1, dep_labels))
        self.bias = nn.Parameter(torch.rand(dep_labels))
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

