import sys
import os
import codecs
import numpy as np
from torchtext import data, datasets, vocab
import csv
import re
from torchtext import datasets

csv.field_size_limit(sys.maxsize)


ROOT_LINE = "0\t__ROOT\t_\t__ROOT\t_\t_\t0\t__ROOT\t_\t_"


def conll_to_csv(args, fname, columns=10):
    col_range = range(columns)

    with codecs.open(fname, 'r', 'utf-8') as f:
        rows, blokk = [], ['"' for _ in col_range]
        # blokk = list(map(lambda x, y: x + y, blokk, ROOT_LINE.split("\t")))
        for line in f:
            if line[0] == '#':
                continue
            if not line.rstrip():
                # [:-1] gets rid of the trailing comma
                blokk = list(map(lambda x: x[:-1] + '"', blokk))
                blokk = ",".join(blokk)
                rows.append(blokk)
                blokk = ['"' for _ in col_range]
                # blokk = list(map(lambda x, y: x + y, blokk, ROOT_LINE.split("\t")))
                continue

            cols = [i.replace('"', '<qt>').replace(',', '<cm>').replace(' ', '<ws>') for i in line.rstrip("\n").split("\t")]
            if '.' in cols[0]: continue
            if args.use_chars:
                cols = cols[:2] + [cols[1]] + cols[2:]
            blokk = list(map(lambda x, y: x + y + ",", blokk, cols))

    return "\n".join(rows)


def dep_to_int(tensor, vocab, _):
    fn = np.vectorize(lambda x: int(vocab.itos[x]))
    return fn(tensor)

'''
1. declare all fields you could every possibly use - this includes CHAR and SEM
2. append/insert them into field_tuples, based on command-line args to select a mode
3. the size of field_tuples = number of columns in the conllu file; pass it to conll_to_csv
4. add the vocab to the vocab dict at the end if you are using them 
'''


def get_iterators(args, batch_size):
    assert len(args.train) == len(args.dev) == len(args.test), "Inconsistent number of treebanks"
    iterators = []

    device = -(not args.use_cuda)
    tokeniser = lambda x: x.split(',')

    for n, (train_eg, dev_eg, test_eg) in enumerate(zip(args.train, args.dev, args.test)):
        ID = data.Field(tokenize=tokeniser, batch_first=True, init_token='0')
        FORM = data.Field(tokenize=tokeniser, batch_first=True, include_lengths=True, init_token='<root>')
        CHAR = data.Field(tokenize=list, batch_first=True, init_token='<w>')
        NEST = data.NestedField(CHAR, tokenize=tokeniser, include_lengths=True, init_token='_')
        LEMMA = data.Field(tokenize=tokeniser, batch_first=True, init_token='<root>')
        UPOS = data.Field(tokenize=tokeniser, batch_first=True, init_token='_')
        XPOS = data.Field(tokenize=tokeniser, batch_first=True, init_token='_')
        FEATS = data.Field(tokenize=tokeniser, batch_first=True, init_token='_')
        HEAD = data.Field(tokenize=tokeniser, batch_first=True, pad_token='-1', init_token='0',
                          unk_token='-1', postprocessing=lambda x, y, z: dep_to_int(x, y, z))
        DEPREL = data.Field(tokenize=tokeniser, batch_first=True, init_token='<root>')
        DEPS = data.Field(tokenize=tokeniser, batch_first=True, init_token='_')
        MISC = data.Field(tokenize=tokeniser, batch_first=True, init_token='_')
        SEM = data.Field(tokenize=tokeniser, batch_first=True, init_token='_')

        # bare conllu
        field_tuples = [('id', ID), ('form', FORM), ('lemma', LEMMA), ('upos', UPOS), ('xpos', XPOS),
                        ('feats', FEATS), ('head', HEAD), ('deprel', DEPREL), ('deps', DEPS), ('misc', MISC)]

        if args.use_chars:
            field_tuples.insert(2, ('char', NEST))

        if args.semtag:
            field_tuples.append(('sem', SEM))

        if not os.path.exists(".tmp"):
            os.makedirs(".tmp")

        # ===== breaking here
        train_csv = conll_to_csv(args, train_eg, len(field_tuples))
        dev_csv = conll_to_csv(args, dev_eg, len(field_tuples))
        test_csv = conll_to_csv(args, test_eg, len(field_tuples))

        for file, text in zip(["train", "dev", "test"], [train_csv, dev_csv, test_csv]):
            with open(os.path.join(".tmp", "{}_{}.csv".format(file, n)), "w") as f:
                f.write(text)

        train, dev, test = data.TabularDataset.splits(path=".tmp", train='train_%s.csv' % n,
                                                      validation='dev_%s.csv' % n, test='test_%s.csv' % n,
                                                      format="csv", fields=field_tuples)

        field_names = [i[1] for i in field_tuples]
        for field in field_names:
            if field == FORM and args.embed:
                vecs = vocab.Vectors(name=args.embed[0])
                field.build_vocab(train, vectors=vecs)
            else:
                field.build_vocab(train)

        current_iterator = data.Iterator.splits((train, dev, test), batch_sizes=(batch_size, batch_size, batch_size),
                                                sort_key=lambda x: len(x.form), sort_within_batch=True,
                                                device=device, repeat=False)

        sizes = {'vocab': len(FORM.vocab), 'postags': len(UPOS.vocab), 'deprels': len(DEPREL.vocab), 'langs': len(MISC.vocab)}

        if args.use_chars:
            sizes['chars'] = len(CHAR.vocab)

        if args.semtag:
            sizes['semtags'] = len(SEM.vocab)

        iterators.append((current_iterator, sizes, FORM.vocab))

    return iterators


def get_treebank_and_txt(args, batch_size):
    device = -(not args.use_cuda)

    tokeniser = lambda x: x.split(',')
    txt_tok = lambda x: x.split(" ")

    ID = data.Field(tokenize=tokeniser, batch_first=True, init_token='0')
    FORM = data.Field(tokenize=tokeniser, batch_first=True, include_lengths=True, init_token='<root>')
    CHAR = data.Field(tokenize=list, batch_first=True, init_token='<w>')
    NEST = data.NestedField(CHAR, tokenize=tokeniser, include_lengths=True, init_token='_')
    LEMMA = data.Field(tokenize=tokeniser, batch_first=True, init_token='<root>')
    UPOS = data.Field(tokenize=tokeniser, batch_first=True, init_token='_')
    XPOS = data.Field(tokenize=tokeniser, batch_first=True, init_token='_')
    FEATS = data.Field(tokenize=tokeniser, batch_first=True, init_token='_')
    HEAD = data.Field(tokenize=tokeniser, batch_first=True, pad_token='-1', init_token='0',
                      unk_token='-1', postprocessing=lambda x, y, z: dep_to_int(x, y, z))
    DEPREL = data.Field(tokenize=tokeniser, batch_first=True, init_token='<root>')
    DEPS = data.Field(tokenize=tokeniser, batch_first=True, init_token='_')
    MISC = data.Field(tokenize=tokeniser, batch_first=True, init_token='_')
    SEM = data.Field(tokenize=tokeniser, batch_first=True, init_token='_')

    LM_FORM = data.Field(tokenize=txt_tok, batch_first=True, include_lengths=True)
    # bare conllu
    field_tuples = [('id', ID), ('form', FORM), ('lemma', LEMMA), ('upos', UPOS), ('xpos', XPOS),
                    ('feats', FEATS), ('head', HEAD), ('deprel', DEPREL), ('deps', DEPS), ('misc', MISC)]

    if args.use_chars:
        field_tuples.insert(2, ('char', NEST))

    if args.semtag:
        field_tuples.append(('sem', SEM))

    if not os.path.exists(".tmp"):
        os.makedirs(".tmp")

    # ===== breaking here
    train_csv = conll_to_csv(args, args.train[0], len(field_tuples))
    dev_csv = conll_to_csv(args, args.dev[0], len(field_tuples))
    test_csv = conll_to_csv(args, args.test[0], len(field_tuples))

    for file, text in zip(["train", "dev", "test"], [train_csv, dev_csv, test_csv]):
        with open(os.path.join(".tmp", "{}_{}.csv".format(file, 0)), "w") as f:
            f.write(text)

    train, dev, test = data.TabularDataset.splits(path=".tmp", train='train_%s.csv' % 0,
                                                  validation='dev_%s.csv' % 0, test='test_%s.csv' % 0,
                                                  format="csv", fields=field_tuples)

    lm_corpus = data.TabularDataset(path=args.lm[0], format="tsv", fields=[('form', FORM)])

    field_names = [i[1] for i in field_tuples]
    for field in field_names:
        if field == FORM and args.embed:
            field.build_vocab(train)
            field.build_vocab(lm_corpus)
        else:
            field.build_vocab(train)

    treebank_iterator = data.Iterator.splits((train, dev, test), batch_sizes=(batch_size, batch_size, batch_size),
                                            sort_key=lambda x: len(x.form), sort_within_batch=True,
                                            device=device, repeat=False)

    lm_iterator = data.Iterator(lm_corpus, batch_size=batch_size, sort_key=lambda x: len(x.form),
                                sort_within_batch=True, repeat=False, device=device)

    sizes = {'vocab': len(FORM.vocab), 'postags': len(UPOS.vocab), 'deprels': len(DEPREL.vocab)}

    if args.use_chars:
        sizes['chars'] = len(CHAR.vocab)

    if args.semtag:
        sizes['semtags'] = len(SEM.vocab)

    return treebank_iterator, lm_iterator, sizes, FORM.vocab



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
    # [i.build_vocab(train) for i in fields]

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

    sizes = {'vocab': len(FORM.vocab), 'postags': len(UPOS.vocab), 'deprels': len(DEPREL.vocab), 'chars': len(CHAR.vocab)}

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


def load_pos():
    WORD = data.Field(init_token="<bos>", eos_token="<eos>", batch_first=True, include_lengths=True)
    LANG = data.Field(init_token="<bos>", eos_token="<eos>", batch_first=True)
    TAG = data.Field(init_token="<bos>", eos_token="<eos>", batch_first=True)

    train, dev, test = datasets.UDPOS.splits(fields=(('word', WORD), ('lang', LANG), ('tag', TAG)),
                                             path="./data/codeswitch/en-hi", train="langid_train.txt",
                                             validation="langid_dev.txt", test="langid_test.txt")

    WORD.build_vocab(train)
    LANG.build_vocab(train)
    TAG.build_vocab(train)

    vocab = {'word': len(WORD.vocab), 'lang': len(LANG.vocab), 'tag': len(TAG.vocab)}
    current_iterator = data.Iterator.splits((train, dev, test), batch_sizes=(10, 1, 1), sort_within_batch=True, repeat=False)
    return current_iterator, vocab
