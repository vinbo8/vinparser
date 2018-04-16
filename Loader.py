import sys
import os
import codecs
import numpy as np
from torchtext import data, datasets, vocab
import csv 
csv.field_size_limit(sys.maxsize)

ROOT_LINE = "0\t__ROOT\t_\t_\t__ROOT\t_\t_\t0\t__ROOT\t_\t_\t_"


def conll_to_csv(fname):
    with codecs.open(fname, 'r', 'utf-8') as f:
        rows, blokk = [], ['"' for _ in range(12)]
        blokk = list(map(lambda x, y: x + y, blokk, ROOT_LINE.split("\t")))
        for line in f:
            if line[0] == '#':
                continue

            if not line.rstrip():
                blokk = list(map(lambda x: x + '"', blokk))
                blokk = ",".join(blokk)
                rows.append(blokk)
                blokk = ['"' for _ in range(12)]
                blokk = list(map(lambda x, y: x + y, blokk, ROOT_LINE.split("\t")))
                continue

            cols = [i.replace('"', '<qt>').replace(',', '<cm>') for i in line.rstrip("\n").split("\t")]
            cols = cols[:2] + [cols[1]] + cols[2:]
            if '.' in cols[0]: continue
            blokk = list(map(lambda x, y: x + ',' + y, blokk, cols))

    return "\n".join(rows)

def get_iterators(args, batch_size):
    device = -(not args.cuda)
    embeds = args.embed

    if not os.path.exists(".tmp"):
        os.makedirs(".tmp")

    if type(args.train) is list: 
        train_csv = conll_to_csv(args.train[0])
        dev_csv = conll_to_csv(args.dev[0])
        test_csv = conll_to_csv(args.test[0])
    else:
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
    CHAR = data.Field(tokenize=list, batch_first=True, init_token='<w>')
    NEST = data.NestedField(CHAR, tokenize=tokeniser, include_lengths=True)
    LEMMA = data.Field(tokenize=tokeniser, batch_first=True)
    UPOS = data.Field(tokenize=tokeniser, batch_first=True)
    XPOS = data.Field(tokenize=tokeniser, batch_first=True)
    FEATS = data.Field(tokenize=tokeniser, batch_first=True)
    HEAD = data.Field(tokenize=tokeniser, batch_first=True, pad_token='-1', unk_token='-1', postprocessing=lambda x, y, z: dep_to_int(x, y, z))
    DEPREL = data.Field(tokenize=tokeniser, batch_first=True)
    DEPS = data.Field(tokenize=tokeniser, batch_first=True)
    MISC = data.Field(tokenize=tokeniser, batch_first=True)
    SEM  = data.Field(tokenize=tokeniser, batch_first=True)

    train, dev, test = data.TabularDataset.splits(path=".tmp", train='train.csv', validation='dev.csv', test='test.csv',
                                                  format="csv", fields=[('id', ID), ('form', FORM),('char', NEST),
                                                                        ('lemma', LEMMA), ('upos', UPOS),
                                                                        ('xpos', XPOS), ('feats', FEATS),
                                                                        ('head', HEAD), ('deprel', DEPREL),
                                                                        ('deps', DEPS), ('misc', MISC), ('sem', SEM)])

    fields = [ID, FORM, NEST, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, MISC, SEM]
    for i in fields:
        if i == FORM and embeds is not '':
            vecs = vocab.Vectors(name=embeds)
            i.build_vocab(train, vectors=vecs)
        else:
            i.build_vocab(train)

    (train_iter, dev_iter, test_iter) = data.Iterator.splits((train, dev, test), batch_sizes=(batch_size, batch_size, batch_size), device=device,
                                                             sort_key=lambda x: len(x.form), sort_within_batch=True,
                                                             repeat=False)
    sizes = {'vocab': len(FORM.vocab), 'postags': len(UPOS.vocab), 'deprels': len(DEPREL.vocab), 'semtags' : len(SEM.vocab), 'chars': len(CHAR.vocab)}

    loader_dict = {"train": train_iter,
                       "dev": dev_iter,
                       "test":  test_iter,
                       "sizes": sizes,
                       "vocab": FORM.vocab
                       }
    print(sizes)
    return  loader_dict

ROOT_LINE_2 = "\t_\t_"

def two_to_csv(fname):
    with codecs.open(fname, 'r', 'utf-8') as f:
        rows, blokk = [], ['"' for _ in range(2)]
        blokk = list(map(lambda x, y: x + y, blokk, ROOT_LINE_2.split("\t")))
        for line in f:
            #if not line.rstrip():
             #   continue

            if not line.rstrip():
                blokk = list(map(lambda x: x + '"', blokk))
                blokk = ",".join(blokk)
                rows.append(blokk)
                blokk = ['"' for _ in range(11)]
                blokk = list(map(lambda x, y: x + y, blokk, ROOT_LINE_2.split("\t")))
                continue
            cols = [i.replace('"', '<qt>').replace(',', '<cm>') for i in line.rstrip("\n").split("\t")]
            #cols = [i for i in line.rstrip("\n").split("\t")]
            if '.' in cols[0]: continue
            #cols = [i.replace(',', '<cm>') for i in line.rstrip("\n").split("\t")]
            blokk = list(map(lambda x, y: x + ',' + y, blokk, cols))
    return "\n".join(rows)


def two_col_data(args, batch_size):
    device = -(not args.cuda)
    tokeniser = lambda x: x.split(',')
    embeds = args.embed

    if type(args.train) is list:
        train_csv = two_to_csv(args.train[1])
        dev_csv = two_to_csv(args.dev[1])
        test_csv = two_to_csv(args.test[1])
    else:
        train_csv = two_to_csv(args.train)
        dev_csv = two_to_csv(args.dev)
        test_csv = two_to_csv(args.test)

    for file, text in zip(["train_aux", "dev_aux", "test_aux"], [train_csv, dev_csv, test_csv]):
        with open(os.path.join(".tmp", file + ".csv"), "w") as f:
            f.write(text)

    FORM = data.Field(tokenize = tokeniser, batch_first=True, include_lengths=True)
    CHAR = data.Field(tokenize=list, batch_first=True, init_token='<w>')
    NEST = data.NestedField(CHAR, tokenize=tokeniser, include_lengths=True)
    SEM = data.Field(tokenize = tokeniser, batch_first=True)

    train, dev, test = data.TabularDataset.splits(path = '.tmp', train = 'train_aux.csv', validation = 'dev_aux.csv', test = 'test_aux.csv',
                                                      format = 'csv', fields=[('form', FORM), ('sem', SEM), ('char', NEST)])
    fields = [FORM, SEM, NEST]
    for i in fields:
        if i == FORM and embeds is not '':
            vecs = vocab.Vectors(name=embeds)
            i.build_vocab(train, vectors=vecs)
        else:
            i.build_vocab(train)

    (train_iter, dev_iter, test_iter) = data.Iterator.splits((train, dev, test), batch_sizes=(batch_size, batch_size, batch_size), device=device,
                                                             sort_key=lambda x: len(x.form), sort_within_batch=True,
                                                             repeat=False)

    sizes = {'vocab': len(FORM.vocab), 'semtags' : len(SEM.vocab), 'chars': len(CHAR.vocab)}
    loader_dict = {"train": train_iter,
                       "dev": dev_iter,
                       "test": test_iter,
                       "sizes": sizes,
                       "vocab": FORM.vocab
                       }
    return  loader_dict

