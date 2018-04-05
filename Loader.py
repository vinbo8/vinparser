import os
import codecs
import torch
from torchtext import data, datasets


ROOT_LINE = "0\t__ROOT\t_\t__ROOT\t_\t_\t0\t__ROOT\t_\t_"


def conll_to_csv(fname):
    with codecs.open(fname, 'r', 'utf-8') as f:
        rows, blokk = [], ['"' for _ in range(10)]
        blokk = list(map(lambda x, y: x + y, blokk, ROOT_LINE.split("\t")))
        for line in f:
            if line[0] == '#':
                continue

            if not line.rstrip():
                blokk = list(map(lambda x: x + '"', blokk))
                blokk = ",".join(blokk)
                rows.append(blokk)
                blokk = ['"' for _ in range(10)]
                blokk = list(map(lambda x, y: x + y, blokk, ROOT_LINE.split("\t")))
                continue

            cols = line.rstrip("\n").split("\t")
            blokk = list(map(lambda x, y: x + ',' + y, blokk, cols))

    return "\n".join(rows)


def get_iterators():
    a = conll_to_csv('data/sv-ud-train.conllu')
    if not os.path.exists(".tmp"):
        os.makedirs(".tmp")

    with open(os.path.join(".tmp", "train.csv"), "w") as f:
        f.write(a)

    tokeniser = lambda x: x.split(',')
    tokint = lambda x: list(map(lambda y: int(y), x.split(',')))
    ID = data.Field(tokenize=tokeniser, batch_first=True)
    FORM = data.Field(tokenize=tokeniser, batch_first=True, include_lengths=True)
    LEMMA = data.Field(tokenize=tokeniser, batch_first=True)
    UPOS = data.Field(tokenize=tokeniser, batch_first=True)
    XPOS = data.Field(tokenize=tokeniser, batch_first=True)
    FEATS = data.Field(tokenize=tokeniser, batch_first=True)
    HEAD = data.Field(tokenize=tokeniser, batch_first=True)
    DEPREL = data.Field(tokenize=tokeniser, batch_first=True)
    DEPS = data.Field(tokenize=tokeniser, batch_first=True)
    MISC = data.Field(tokenize=tokeniser, batch_first=True)
    fields = [ID, FORM, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, MISC]


    train = data.TabularDataset(path=".tmp/train.csv", format="csv", fields=[('id', ID), ('form', FORM),
                                                                             ('lemma', LEMMA), ('upos', UPOS),
                                                                             ('xpos', XPOS), ('feats', FEATS),
                                                                             ('head', HEAD), ('deprel', DEPREL),
                                                                             ('deps', DEPS), ('misc', MISC)])

    [i.build_vocab(train) for i in fields]
    (train_iter,) = data.Iterator.splits((train,), batch_sizes=(32,), device=-1)
    for i, batch in enumerate(train_iter):
        (forms, sizes), tags, heads, deprels = batch.form, batch.upos, batch.head, batch.deprel
        mask = torch.zeros(sizes.size()[0], max(sizes)).type(torch.LongTensor)
        for n, i in enumerate(sizes):
            mask[n, 0:i] = 1


