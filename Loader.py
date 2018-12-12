import sys
import random
random.seed(1337)
import os
import codecs
import numpy as np
from torchtext import data, datasets, vocab
import csv
import re
from torchtext import datasets, data, vocab
import time

random.seed(1337)
np.random.seed(1337)
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
            blokk = list(map(lambda x, y: x + y + ",", blokk, cols))

    return "\n".join(rows)


def dep_to_int(tensor, vocab):
    fn = np.vectorize(lambda x: int(vocab.itos[x]))
    return fn(tensor)

'''
1. declare all fields you could every possibly use - this includes CHAR and SEM
2. append/insert them into field_tuples, based on command-line args to select a mode
3. the size of field_tuples = number of columns in the conllu file; pass it to conll_to_csv
4. add the vocab to the vocab dict at the end if you are using them 
'''

def get_iterators(args, src_file, train_fields=None):
    tokeniser = lambda x: x.split(',')

    ID = data.Field(tokenize=tokeniser, batch_first=True, init_token='0')
    FORM = data.Field(tokenize=tokeniser, batch_first=True, include_lengths=True, init_token='<root>')
    LEMMA = data.Field(tokenize=tokeniser, batch_first=True, init_token='<root>')
    UPOS = data.Field(tokenize=tokeniser, batch_first=True, init_token='_')
    XPOS = data.Field(tokenize=tokeniser, batch_first=True, init_token='_')
    FEATS = data.Field(tokenize=tokeniser, batch_first=True, init_token='_')
    HEAD = data.Field(tokenize=tokeniser, batch_first=True, pad_token='-1', init_token='0',
                        unk_token='-1', postprocessing=lambda x, y: dep_to_int(x, y))
    DEPREL = data.Field(tokenize=tokeniser, batch_first=True, init_token='<root>')
    DEPS = data.Field(tokenize=tokeniser, batch_first=True, init_token='_')
    MISC = data.Field(tokenize=tokeniser, batch_first=True, init_token='_')
    SEM = data.Field(tokenize=tokeniser, batch_first=True, init_token='_')

    # bare conllu
    field_tuples = [('id', ID), ('form', FORM), ('lemma', LEMMA), ('upos', UPOS), ('xpos', XPOS),
                    ('feats', FEATS), ('head', HEAD), ('deprel', DEPREL), ('deps', DEPS), ('misc', MISC)]

    if train_fields:
        field_tuples = train_fields

    if not os.path.exists(".tmp"):
        os.makedirs(".tmp")

    # ===== breaking here
    src_csv = conll_to_csv(args, src_file, len(field_tuples))

    seconds_since_epoch = time.mktime(time.localtime())
    with open(os.path.join(".tmp", "src_{}.csv".format(seconds_since_epoch)), "w") as f:
        f.write(src_csv)

    src_dataset = data.TabularDataset(path=".tmp/src_{}.csv".format(seconds_since_epoch), format="csv", fields=field_tuples)

    if not train_fields:
        field_names = [i[1] for i in field_tuples]
        for field in field_names:
            if field == FORM and args.src_embed_file:
                vecs = vocab.Vectors(name=args.src_embed_file)
                field.build_vocab(src_dataset, vectors=vecs)
                print("* using source language embeddings")
            else:
                field.build_vocab(src_dataset)

    out_iterator = data.Iterator(src_dataset, batch_size=args.batch_size, sort_key=lambda x: len(x.form), train=not train_fields,
                                 sort_within_batch=True, device=args.device, repeat=False)

    vocabs = {'forms': FORM.vocab, 'postags': UPOS.vocab, 'deprels': DEPREL.vocab, 
              'feats': FEATS.vocab, 'misc': MISC.vocab} if not train_fields else None

    return out_iterator, field_tuples, vocabs


def get_mt(args, vocab_from_dep):
    src_lang = data.Field(lower=True, batch_first=True, include_lengths=True)
    trg_lang = data.Field(lower=True, batch_first=True, include_lengths=True)

    mt_data = datasets.TranslationDataset(os.path.join(args.mt, ''),
                                          ('en.txt', '%s.txt' % args.lang),
                                          (src_lang, trg_lang))

    src_lang.vocab = vocab_from_dep
    
    if args.trg_embed_file:
        vecs = vocab.Vectors(name=args.trg_embed_file)
        trg_lang.build_vocab(mt_data, vectors=vecs)
        print("* using target language embeddings")
    else:
        trg_lang.build_vocab(mt_data)

    mt_iterator = data.Iterator(mt_data, batch_size=args.batch_size, train=True,
                                repeat=False, shuffle=True, device=args.device)

    trg_field = trg_lang
    vocabs = {'src': src_lang.vocab, 'trg': trg_lang.vocab}
    
    return mt_iterator, trg_field, vocabs
