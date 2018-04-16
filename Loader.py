import os
import codecs
import numpy as np
from torchtext import data, datasets, vocab


ROOT_LINE = "0\t__ROOT\t_\t_\t__ROOT\t_\t_\t0\t__ROOT\t_\t_"


def conll_to_csv(fname):
    with codecs.open(fname, 'r', 'utf-8') as f:
        rows, blokk = [], ['"' for _ in range(11)]
        blokk = list(map(lambda x, y: x + y, blokk, ROOT_LINE.split("\t")))
        for line in f:
            if line[0] == '#':
                continue

            if not line.rstrip():
                blokk = list(map(lambda x: x + '"', blokk))
                blokk = ",".join(blokk)
                rows.append(blokk)
                blokk = ['"' for _ in range(11)]
                blokk = list(map(lambda x, y: x + y, blokk, ROOT_LINE.split("\t")))
                continue

            cols = [i.replace('"', '<qt>').replace(',', '<cm>') for i in line.rstrip("\n").split("\t")]
            if '.' in cols[0]: continue
            cols = cols[:2] + [cols[1]] + cols[2:]
            blokk = list(map(lambda x, y: x + ',' + y, blokk, cols))

    return "\n".join(rows)


# loads only the top of each train/dev/test stack
def get_iterators(args, batch_size):
    device = -(not args.cuda)

    if not os.path.exists(".tmp"):
        os.makedirs(".tmp")

    train_csv = conll_to_csv(args.train[0])
    dev_csv = conll_to_csv(args.dev[0])
    test_csv = conll_to_csv(args.test[0])

    for file, text in zip(["train", "dev", "test"], [train_csv, dev_csv, test_csv]):
        with open(os.path.join(".tmp", file + ".csv"), "w") as f:
            f.write(text)

    def dep_to_int(tensor, field_vocab, _):
        fn = np.vectorize(lambda x: int(field_vocab.itos[x]))
        return fn(tensor)

    tokeniser = lambda x: x.split(',')
    ID = data.Field(tokenize=tokeniser, batch_first=True)
    FORM = data.Field(tokenize=tokeniser, batch_first=True, include_lengths=True)
    CHAR = data.Field(tokenize=list, batch_first=True, init_token='<w>')
    NEST = data.NestedField(CHAR, tokenize=tokeniser, include_lengths=True)
    LEMMA = data.Field(tokenize=tokeniser, batch_first=True)
    UPOS = data.Field(tokenize=tokeniser, batch_first=True)
    XPOS = data.Field(tokenize=tokeniser, batch_first=True)
    FEATS = data.Field(tokenize=tokeniser, batch_first=True)
    HEAD = data.Field(tokenize=tokeniser, batch_first=True, pad_token='-1', unk_token='-1',
                      postprocessing=lambda x, y, z: dep_to_int(x, y, z))
    DEPREL = data.Field(tokenize=tokeniser, batch_first=True)
    DEPS = data.Field(tokenize=tokeniser, batch_first=True)
    MISC = data.Field(tokenize=tokeniser, batch_first=True)

    train, dev, test = data.TabularDataset.splits(path=".tmp", train='train.csv', validation='dev.csv', test='test.csv',
                                                  format="csv", fields=[('id', ID), ('form', FORM), ('char', NEST),
                                                                        ('lemma', LEMMA), ('upos', UPOS),
                                                                        ('xpos', XPOS), ('feats', FEATS),
                                                                        ('head', HEAD), ('deprel', DEPREL),
                                                                        ('deps', DEPS), ('misc', MISC)])

    fields = [ID, FORM, NEST, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, MISC]
    for i in fields:
        if i == FORM and args.embed is not None:
            vecs = vocab.Vectors(name=args.embed)
            i.build_vocab(train, vectors=vecs)
        else:
            i.build_vocab(train)

    (train_iter, dev_iter, test_iter) = data.Iterator.splits((train, dev, test),
                                                             batch_sizes=(batch_size, batch_size, batch_size),
                                                             device=device,
                                                             sort_key=lambda x: len(x.form), sort_within_batch=True,
                                                             repeat=False)
    sizes = {'vocab': len(FORM.vocab), 'postags': len(UPOS.vocab), 'deprels': len(DEPREL.vocab),
             'langs': len(MISC.vocab), 'chars': len(NEST.vocab)}

    return (train_iter, dev_iter, test_iter), sizes


def get_iterators_cl(args, batch_size):

    assert len(args.train) == len(args.dev) == len(args.test), \
        "Train/Dev/Test must be provided for all languages."

    NUMLANGS = len(args.train)

    if args.embed is not None:
        assert len(args.embed) == NUMLANGS, \
            "Embeds must be provided for all languages."

    out = []
    for i in range(NUMLANGS):
        loaders, sizes, vocab = get_iterators((args.train[i], args.dev[i], args.test[i]), args.embed[i], batch_size, args.cuda)
        loader_dict = {"train": loaders[0],
                       "dev": loaders[1],
                       "test": loaders[2],
                       "sizes": sizes,
                       "vocab": vocab
                       }
        out.append(loader_dict)

    return out
