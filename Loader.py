import sys
import os
import codecs
import numpy as np
from torchtext import data, datasets, vocab
import csv
import re
from torchtext import datasets
import time

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

            cols = [i.replace('"', '<qt>').replace(',', '<cm>') for i in line.rstrip("\n").split("\t")]
            if '.' in cols[0] or '-' in cols[0]: continue
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


# MASSIVE TODO: write iterator getter for single treebank files for running loaded models
def get_iterators(args, batch_size):
    device = -(not args.use_cuda)
    tokeniser = lambda x: x.split(',')

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
    train_csv = conll_to_csv(args, args.train, len(field_tuples))
    dev_csv = conll_to_csv(args, args.dev, len(field_tuples))
    test_csv = conll_to_csv(args, args.test, len(field_tuples))

    seconds_since_epoch = time.mktime(time.localtime())
    for file, text in zip(["train", "dev", "test"], [train_csv, dev_csv, test_csv]):
        with open(os.path.join(".tmp", "{}_{}.csv".format(file, seconds_since_epoch)), "w") as f:
            f.write(text)

    train, dev, test = data.TabularDataset.splits(path=".tmp", train='train_{}.csv'.format(seconds_since_epoch),
                                                    validation='dev_{}.csv'.format(seconds_since_epoch), 
                                                    test='test_{}.csv'.format(seconds_since_epoch),
                                                    format="csv", fields=field_tuples)

    field_names = [i[1] for i in field_tuples]
    for field in field_names:
        if field == FORM and args.embed:
            vecs = vocab.Vectors(name=args.embed)
            field.build_vocab(train, vectors=vecs)
        else:
            field.build_vocab(train)

    train_iterator = data.Iterator(train, batch_size=batch_size, sort_key=lambda x: len(x.form), train=True,
                                    sort_within_batch=True, device=device, repeat=False)

    dev_iterator = data.Iterator(dev, batch_size=1, train=False, sort_within_batch=True, sort_key=lambda x: len(x.form),
                                    sort=False, device=device, repeat=False)

    test_iterator = data.Iterator(test, batch_size=1, train=False, sort_within_batch=True, sort_key=lambda x: len(x.form),
                                    sort=False, device=device, repeat=False)

    current_iterator = [train_iterator, dev_iterator, test_iterator]

    vocabs = {'forms': FORM.vocab, 'postags': UPOS.vocab, 'deprels': DEPREL.vocab, 
             'feats': FEATS.vocab, 'misc': MISC.vocab}
    sizes = {k: len(v) for (k, v) in vocabs.items()}

    if args.use_chars:
        vocabs['chars'] = CHAR.vocab
        sizes['chars'] = len(CHAR.vocab)

    return (current_iterator, sizes, vocabs)

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
