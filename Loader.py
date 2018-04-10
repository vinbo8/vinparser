import os
import codecs
import numpy as np
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
            cols = [i.replace('"', '<qt>').replace(',', '<cm>') for i in line.rstrip("\n").split("\t")]
            if '.' in cols[0]: continue
            #cols = [i.replace(',', '<cm>') for i in line.rstrip("\n").split("\t")]
            blokk = list(map(lambda x, y: x + ',' + y, blokk, cols))

    return "\n".join(rows)


def get_iterators(args, batch_size):
    device = -(not args.cuda)

    if not os.path.exists(".tmp"):
        os.makedirs(".tmp")

    train_csv = conll_to_csv(args.train)
    dev_csv = conll_to_csv(args.dev)
    test_csv = conll_to_csv(args.test)

    for file, text in zip(["train", "dev", "test"], [train_csv, dev_csv, test_csv]):
        with open(os.path.join(".tmp", file + ".csv"), "w") as f:
            f.write(text)

    def dep_to_int(tensor, vocab, _):
        fn = np.vectorize(lambda x: int(vocab.itos[x]))
        return fn(tensor)

    tokeniser = lambda x: x.split(',')
    ID = data.Field(tokenize=tokeniser, batch_first=True)
    FORM = data.Field(tokenize=tokeniser, batch_first=True, include_lengths=True)
    LEMMA = data.Field(tokenize=tokeniser, batch_first=True)
    UPOS = data.Field(tokenize=tokeniser, batch_first=True)
    XPOS = data.Field(tokenize=tokeniser, batch_first=True)
    FEATS = data.Field(tokenize=tokeniser, batch_first=True)
    HEAD = data.Field(tokenize=tokeniser, batch_first=True, pad_token='-1', unk_token='-1', postprocessing=lambda x, y, z: dep_to_int(x, y, z))
    DEPREL = data.Field(tokenize=tokeniser, batch_first=True)
    DEPS = data.Field(tokenize=tokeniser, batch_first=True)
    MISC = data.Field(tokenize=tokeniser, batch_first=True)

    train, dev, test = data.TabularDataset.splits(path=".tmp", train='train.csv', validation='dev.csv', test='test.csv',
                                                  format="csv", fields=[('id', ID), ('form', FORM),
                                                                        ('lemma', LEMMA), ('upos', UPOS),
                                                                        ('xpos', XPOS), ('feats', FEATS),
                                                                        ('head', HEAD), ('deprel', DEPREL),
                                                                        ('deps', DEPS), ('misc', MISC)])

    fields = [ID, FORM, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, MISC]
    [i.build_vocab(train) for i in fields]

    (train_iter, dev_iter, test_iter) = data.Iterator.splits((train, dev, test), batch_sizes=(batch_size, batch_size, batch_size), device=device,
                                                             sort_key=lambda x: len(x.form), sort_within_batch=True,
                                                             repeat=False)
    sizes = {'vocab': len(FORM.vocab), 'postags': len(UPOS.vocab), 'deprels': len(DEPREL.vocab)}

    return (train_iter, dev_iter, test_iter), sizes
