import sys
import math
import torch
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable
from conllu import ConllParser

MAX_SENT = 500
LSTM_DIM = 40
LSTM_DEPTH = 3
EMBEDDING_DIM = 100
REDUCE_DIM = 500
BATCH_SIZE = 10
EPOCHS = 1
LEARNING_RATE = 2e-3
DEBUG_SIZE = 500


class Network(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embeddings = torch.nn.Embedding(vocab_size, embedding_dim)
        self.lstm = torch.nn.LSTM(EMBEDDING_DIM, LSTM_DIM, LSTM_DEPTH, batch_first=True)
        self.mlp_head = torch.nn.Linear(LSTM_DIM, REDUCE_DIM)
        self.mlp_dep = torch.nn.Linear(LSTM_DIM, REDUCE_DIM)
        self.biaffine_weight = torch.nn.Parameter(torch.rand(BATCH_SIZE, REDUCE_DIM + 1, REDUCE_DIM))
        self.softmax = torch.nn.Softmax(dim=2)

    def forward(self, forms):
        embeds = self.embeddings(forms)
        assert embeds.shape == torch.Size([BATCH_SIZE, MAX_SENT, EMBEDDING_DIM])

        output, (h_n, c_n) = self.lstm(embeds)
        assert output.shape == torch.Size([BATCH_SIZE, MAX_SENT, LSTM_DIM])

        reduced_head = F.relu(self.mlp_head(output))
        assert reduced_head.shape == torch.Size([BATCH_SIZE, MAX_SENT, REDUCE_DIM])

        reduced_dep = F.relu(self.mlp_dep(output))
        bias = Variable(torch.ones(BATCH_SIZE, MAX_SENT, 1))
        reduced_dep = torch.cat([reduced_dep, bias], 2)
        assert reduced_dep.shape == torch.Size([BATCH_SIZE, MAX_SENT, REDUCE_DIM + 1])

        # ROW IS DEP, COL IS HEAD
        y_pred = self.softmax(reduced_dep @ self.biaffine_weight @ reduced_head.transpose(1, 2))
        return y_pred


def rel_pad(l, max_len):
    tensor = torch.LongTensor(l)
    diff = max_len - tensor.shape[0]
    return F.pad(tensor, (0, 0, 0, diff), value=-1)


def form_pad(l, max_len):
    tensor = torch.LongTensor(l)
    diff = max_len - tensor.shape[0]
    l, r = math.floor(diff / 2), math.ceil(diff / 2)
    return F.pad(tensor, (1, diff - 1))


def main():
    # build data
    c = ConllParser()
    with open('data/sv-ud-train.conllu', 'r') as f:
        c.build(f)

    # vocab and indexes
    vocab = set(' '.join(block.raw() for block in c).split())
    vocab_size = len(vocab)
    word_to_idx = {word: i + 2 for i, word in enumerate(vocab)}
    word_to_idx['PAD'] = 0

    # sentences
    print("Preparing data..")
    sentence_list = [block.raw().split() for block in c]
    deprel_list = [rel_pad(block.rels(), MAX_SENT) for block in c]

    # forms
    sent_idxs = [form_pad([word_to_idx[word] for word in sent], MAX_SENT) for sent in sentence_list]
    forms = torch.stack(sent_idxs)[:DEBUG_SIZE].data
    assert forms.shape == torch.Size([DEBUG_SIZE, MAX_SENT])

    # labels
    # DEBUG_SIZE == TREEBANK_SIZE
    # ugly; rewrite loop?
    rels = torch.stack(deprel_list)[:DEBUG_SIZE]
    labels = torch.zeros(forms.shape[0], MAX_SENT, 1)
    for batch_no, _ in enumerate(rels):
        for rel in rels[batch_no]:
            if rel[1].data[0] == -1:
                continue
            labels[batch_no, rel[1].data[0]] = rel[0].data[0]

    labels = torch.squeeze(labels.type(torch.LongTensor))
    assert labels.shape == torch.Size([DEBUG_SIZE, MAX_SENT])

    # sizes
    sizes_int = torch.zeros(DEBUG_SIZE).view(-1, 1).type(torch.LongTensor)
    sizes = torch.zeros(DEBUG_SIZE, MAX_SENT)
    for n, form in enumerate(forms):
        sizes_int[n] = form[form != 0].shape[0] + 1

    for n, size in enumerate(sizes_int):
        sizes[n, 0:size[0]] = 1

    assert sizes.shape == torch.Size([DEBUG_SIZE, MAX_SENT])

    # build loader & model
    train_data = list(zip(forms, labels, sizes))[:DEBUG_SIZE]
    test_data = list(zip(forms, labels, sizes))[:DEBUG_SIZE]

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

    parser = Network(vocab_size, EMBEDDING_DIM)

    # training
    print("Training..")
    parser.train()
    criterion = torch.nn.NLLLoss(reduce=False)
    optimiser = torch.optim.Adam(parser.parameters(), lr=LEARNING_RATE)
    for epoch in range(EPOCHS):
        for i, data in enumerate(train_loader):
            forms, labels, sizes = data
            X = Variable(forms)
            y = Variable(labels, requires_grad=False)
            mask = Variable(sizes)
            y_pred = parser(X)
            train_loss = (criterion(y_pred, y) * mask).sum().sum() / mask.nonzero().size(0)
            optimiser.zero_grad()
            train_loss.backward()
            optimiser.step()

        print("Epoch: {}\tloss: {}".format(epoch, train_loss.data[0]))

    # test
    print("Eval..")
    correct = 0
    total_deps = 0
    parser.eval()
    for i, data in enumerate(test_loader):
        forms, labels, sizes = data
        X = Variable(forms)
        y = Variable(labels, requires_grad=False)
        y_pred = parser(X)
        total_deps = y.nonzero().size(0)
        correct = (y_pred.max(2)[1] * y).nonzero().size(0)

    print("Accuracy = {}/{} = {}".format(correct, total_deps, (correct / total_deps)))


if __name__ == '__main__':
    main()
