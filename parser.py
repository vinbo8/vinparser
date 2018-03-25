import os
import math
import argparse
import torch
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable
from conllu import ConllParser

LSTM_DIM = 200
LSTM_DEPTH = 3
EMBEDDING_DIM = 100
REDUCE_DIM = 500
BATCH_SIZE = 10
EPOCHS = 5
LEARNING_RATE = 2e-3
DEBUG_SIZE = 100


class Biaffine(torch.nn.Module):
    """
    https://github.com/tdozat/Parser/blob/0739216129cd39d69997d28cbc4133b360ea3934/lib/linalg.py#L116  # NOQA
    """

    def __init__(self, in1_features, in2_features, out_features,
                 bias=(True, True, True)):
        super(Biaffine, self).__init__()
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features
        self._use_bias = bias

        shape = (in1_features + int(bias[0]),
                 in2_features + int(bias[1]),
                 out_features)
        self.weight = torch.nn.Parameter(torch.Tensor(*shape))
        if bias[2]:
            self.bias = torch.nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(0))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input1, input2):
        is_cuda = next(self.parameters()).is_cuda
        device_id = next(self.parameters()).get_device() if is_cuda else None
        out_size = self.out_features
        batch_size, len1, dim1 = input1.size()
        if self._use_bias[0]:
            ones = torch.ones(batch_size, len1, 1)
            if is_cuda:
                ones = ones.cuda(device_id)
            input1 = torch.cat((input1, Variable(ones)), dim=2)
            dim1 += 1
        len2, dim2 = input2.size()[1:]
        if self._use_bias[1]:
            ones = torch.ones(batch_size, len2, 1)
            if is_cuda:
                ones = ones.cuda(device_id)
            input2 = torch.cat((input2, Variable(ones)), dim=2)
            dim2 += 1
        input1_reshaped = input1.contiguous().view(batch_size * len1, dim1)
        W_reshaped = torch.transpose(self.weight, 1, 2) \
            .contiguous().view(dim1, out_size * dim2)
        affine = torch.mm(input1_reshaped, W_reshaped) \
            .view(batch_size, len1 * out_size, dim2)
        biaffine = torch.transpose(
            torch.bmm(affine, torch.transpose(input2, 1, 2))
            .view(batch_size, len1, out_size, len2), 2, 3)
        if self._use_bias[2]:
            biaffine += self.bias.expand_as(biaffine)
        return biaffine

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + 'in1_features=' + str(self.in1_features) \
            + ', in2_features=' + str(self.in2_features) \
            + ', out_features=' + str(self.out_features) + ')'


class BilinearFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, m1, m2, weight, bias=None):
        ctx.save_for_backward(m1, m2, weight, bias)
        output = m1 @ weight @ m2
        return output

    @staticmethod
    def backward(ctx, grad_output):
        m1, m2, weight, bias = ctx.saved_variables
        grad_m1 = grad_m2 = grad_weight = grad_bias = None

        grad_m1 = grad_output @ m2.transpose(1, 2) @ weight.transpose(1, 2)
        grad_m2 = weight.transpose(1, 2) @ m1.transpose(1, 2) @ grad_output
        grad_weight = m1.transpose(1, 2) @ grad_output @ m2.transpose(1, 2)

        return grad_m1, grad_m2, grad_weight, grad_bias


class BiasedBilinear(torch.nn.Module):
    def __init__(self, batch_size, dim_features):
        super().__init__()
        self.batch_size = batch_size
        self.dim_features = dim_features
        self.weight = torch.nn.Parameter(torch.Tensor(batch_size, dim_features, dim_features))
        self.reset_params()

    def reset_params(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, reduced_dep, reduced_head):
        return BilinearFunc.apply(reduced_dep, reduced_head.transpose(1, 2), self.weight)


class Network(torch.nn.Module):
    def __init__(self, vocab_size, tag_vocab):
        super().__init__()
        self.embeddings_forms = torch.nn.Embedding(vocab_size, EMBEDDING_DIM)
        self.embeddings_tags = torch.nn.Embedding(tag_vocab, EMBEDDING_DIM)
        self.lstm = torch.nn.LSTM(EMBEDDING_DIM, LSTM_DIM, LSTM_DEPTH,
                                  batch_first=True, bidirectional=True)
        self.lstm_2 = torch.nn.LSTM(EMBEDDING_DIM, LSTM_DIM, LSTM_DEPTH, batch_first=True, bidirectional=True)
        self.mlp_head = torch.nn.Linear(2 * LSTM_DIM, REDUCE_DIM)
        self.mlp_dep = torch.nn.Linear(2 * LSTM_DIM, REDUCE_DIM)
        self.biaffine_weight = torch.nn.Parameter(torch.rand(BATCH_SIZE, REDUCE_DIM + 1, REDUCE_DIM), requires_grad=True)
        self.softmax = torch.nn.LogSoftmax(dim=2)
        # self.criterion = torch.nn.NLLLoss(reduce=False)
        self.criterion = torch.nn.NLLLoss()
        self.relu = torch.nn.ReLU()
        self.optimiser = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.9))
        self.weird_thing = BiasedBilinear(BATCH_SIZE, REDUCE_DIM)
        self.dropout = torch.nn.Dropout(p=0.33)
        self.bilinear = torch.nn.Bilinear(REDUCE_DIM, REDUCE_DIM, 500, bias=False)
        self.compose = torch.nn.Linear(REDUCE_DIM, REDUCE_DIM)
        self.final = torch.nn.Linear(REDUCE_DIM, REDUCE_DIM)
        self.ripoff = Biaffine(REDUCE_DIM, REDUCE_DIM, 1, bias=(True, False, False))

    def forward(self, forms, tags, sizes, pack):
        # for debug:
        # sizes[0, :, 0] = 1
        # sizes = torch.stack([sizes for i in range(EMBEDDING_DIM)], dim=2)
        # sizes = torch.stack([sizes for i in range(2 * LSTM_DIM)], dim=2)

        MAX_SENT = forms.size(1)
        form_embeds = self.embeddings_forms(forms)
        assert form_embeds.shape == torch.Size([BATCH_SIZE, MAX_SENT, EMBEDDING_DIM])

        tag_embeds = self.dropout(self.embeddings_tags(tags))
        assert tag_embeds.shape == torch.Size([BATCH_SIZE, MAX_SENT, EMBEDDING_DIM])

        embeds = form_embeds

        embeds = torch.nn.utils.rnn.pack_padded_sequence(embeds, pack, batch_first=True)
        output, _ = self.lstm(embeds)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        reduced_head = self.relu(self.mlp_head(output))
        reduced_dep = self.relu(self.mlp_dep(output))
        # bias = Variable(torch.ones(BATCH_SIZE, output.size(1), 1))
        # reduced_dep = torch.cat([reduced_dep, bias], 2)
        # ROW IS DEP, COL IS HEAD
        # y_pred = reduced_dep @ self.biaffine_weight @ reduced_head.transpose(1, 2)
        # y_pred = self.weird_thing(reduced_dep, reduced_head)
        # y_pred = self.final(reduced_dep) @ self.compose(reduced_head).transpose(1, 2)
        y_pred = self.ripoff(reduced_head, reduced_dep).squeeze(3)
        # y_pred = self.bilinear(reduced_dep.transpose(1, 2), reduced_head.transpose(1, 2))

        # y_pred = self.softmax(y_pred)
        return y_pred

    def train_(self, epoch, train_loader):
        self.train()
        for i, (forms, tags, labels, sizes) in enumerate(train_loader):
            forms, tags, labels, sizes = [torch.stack(list(i)) for i in zip(*sorted(zip(forms, tags, labels, sizes), key=lambda x: x[3].nonzero().size(0), reverse=True))]
            trunc = max([i.nonzero().size(0) + 1 for i in sizes])
            X1 = Variable(forms[:, :trunc])
            X2 = Variable(tags[:, :trunc])
            y = Variable(labels[:, :trunc], requires_grad=False)
            y[:, 0] = 0
            mask = Variable(sizes[:, :trunc])
            pack = [i.nonzero().size(0) + 1 for i in sizes]
            # squeezing
            y_pred = self(X1, X2, mask, pack)
            dims = y.size()
            a = y_pred.view(BATCH_SIZE * dims[1], dims[1])
            b = y.contiguous().view(BATCH_SIZE * dims[1])
            train_loss = F.cross_entropy(a, b, ignore_index=-1)
            self.zero_grad()
            train_loss.backward()
            self.optimiser.step()

            print("Epoch: {}\t{}/{}\tloss: {}".format(epoch, (i + 1) * len(forms), len(train_loader.dataset), train_loss.data[0]))

    def evaluate_(self, test_loader):
        correct = 0
        total_deps = 0
        self.eval()
        for i, (forms, tags, labels, sizes) in enumerate(test_loader):
            forms, tags, labels, sizes = [torch.stack(list(i)) for i in zip(*sorted(zip(forms, tags, labels, sizes), key=lambda x: x[3].nonzero().size(0), reverse=True))]
            trunc = max([i.nonzero().size(0) + 1 for i in sizes])
            X1 = Variable(forms[:, :trunc])
            X2 = Variable(tags[:, :trunc])
            y = Variable(labels[:, :trunc], requires_grad=False)
            mask = Variable(sizes[:, :trunc])
            pack = [i.nonzero().size(0) + 1 for i in sizes]
            y_pred = self(X1, X2, mask, pack)
            temp = y_pred.max(2)[1]
            try:
                correct += ((y == y_pred.max(2)[1]) * mask.type(torch.ByteTensor)).nonzero().size(0)
            except RuntimeError:
                print("fail")
                correct += 0
            total_deps += mask.nonzero().size(0)

        print("Accuracy = {}/{} = {}".format(correct, total_deps, (correct / total_deps)))


def build_data(fname, train_conll=None):
    # build data
    with open(os.path.join('data', fname), 'r') as f:
        conll = ConllParser(f) if not train_conll else ConllParser(f, train_conll)

    # sentences
    print("Preparing %s.." % fname)
    forms, rels, tags = conll.get_tensors()
    assert forms.shape == torch.Size([len(conll), conll.longest_sent])
    assert tags.shape == torch.Size([len(conll), conll.longest_sent])

    # labels
    labels = -torch.ones(forms.shape[0], conll.longest_sent, 1)
    for batch_no, _ in enumerate(rels):
        for rel in rels[batch_no]:
            # if rel[1] == 0:
            #   con
            labels[batch_no, rel[1]] = rel[0]

    labels = torch.squeeze(labels.type(torch.LongTensor))
    assert labels.shape == torch.Size([len(conll), conll.longest_sent])

    # sizes
    sizes_int = torch.zeros(len(conll)).view(-1, 1).type(torch.LongTensor)
    sizes = torch.zeros(len(conll), conll.longest_sent)
    for n, form in enumerate(forms):
        sizes_int[n] = form[form != 0].shape[0]

    for n, size in enumerate(sizes_int):
        sizes[n, 1:size[0]] = 1

    assert sizes.shape == torch.Size([len(conll), conll.longest_sent])

    # build loader & model
    data = list(zip(forms, tags, labels, sizes))[:DEBUG_SIZE]
    loader = torch.utils.data.DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    return conll, loader


def main():
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    conll, train_loader = build_data('sv-ud-train.conllu')
    _, test_loader = build_data('sv-ud-test.conllu', conll)

    parser = Network(conll.vocab_size, conll.pos_size)
    # training
    print("Training")
    for epoch in range(EPOCHS):
        parser.train_(epoch, train_loader)

    # test
    print("Eval")
    parser.evaluate_(train_loader)


if __name__ == '__main__':
    main()
