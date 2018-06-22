import sys
import os
import codecs
import numpy as np
from torchtext import data, datasets, vocab
import csv
csv.field_size_limit(sys.maxsize)


ROOT_LINE = "0\t__ROOT\t_\t__ROOT\t_\t_\t0\t__ROOT\t_\t_"


def conll_to_seg_csv(args, fname, columns=10, train=True):
    col_range = range(columns)
    rows = []
    with codecs.open(fname, 'r', 'utf-8') as f:
        for n, line in enumerate(f):
            if line[0] == '#':
                continue
            elif not line.rstrip():
                rows[-1][1] = 1
            else:
                cols = [i.replace('"', '<qt>').replace(',', '<cm>') for i in line.rstrip("\n").split("\t")]
                if '.' in cols[0] or '-' in cols[0]: continue
                rows.append([cols[1], 0])
                # rows.append('"{}",{}'.format(cols[1], 0))
    rows = ['"{}","{}"'.format(word, num) for word, num in rows]
    return "\n".join(rows)


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

# just load raw text, write tokenised conllu for next loader
def seg_iterators(args, batch_size):
    batch = []
    device = -(not args.use_cuda)

    WORD = data.Field(batch_first=True, init_token='<w>')
    SWITCH = data.Field(batch_first=True, init_token='0')

    field_tuples = [('word', WORD), ('switch', SWITCH)]
    
    train_csv = conll_to_seg_csv(args, args.train, len(field_tuples))
    dev_csv = conll_to_seg_csv(args, args.dev, len(field_tuples))
    test_csv = conll_to_seg_csv(args, args.test, len(field_tuples))

    for file, text in zip(["train", "dev", "test"], [train_csv, dev_csv, test_csv]):
        with open(os.path.join(".tmp", "{}.csv".format(file)), "w") as f:
            f.write(text)

    train, dev, test = data.TabularDataset.splits(path=".tmp", train='train.csv',
                                                    validation='dev.csv', test='test.csv',
                                                    format="csv", fields=field_tuples)

    field_names = [i[1] for i in field_tuples]
    for field in field_names:
        field.build_vocab(train)

    train_iterator = data.Iterator(train, batch_size=batch_size, train=True,
                                    sort_within_batch=False, device=device, repeat=False)

    dev_iterator = data.Iterator(dev, batch_size=1, train=False, sort_within_batch=True, sort_key=lambda x: len(x.form),
                                    sort=False, device=device, repeat=False)

    test_iterator = data.Iterator(test, batch_size=1, train=False, sort_within_batch=True, sort_key=lambda x: len(x.form),
                                    sort=False, device=device, repeat=False)

    current_iterator = [train_iterator, dev_iterator, test_iterator]

    sizes = {'vocab': len(WORD.vocab), 'states': len(SWITCH.vocab)}

    # if args.use_chars:
    #     sizes['chars'] = len(CHAR.vocab)

    return (current_iterator, sizes, [WORD.vocab, SWITCH.vocab])

'''
1. declare all fields you could every possibly use - this includes CHAR and SEM
2. append/insert them into field_tuples, based on command-line args to select a mode
3. the size of field_tuples = number of columns in the conllu file; pass it to conll_to_csv
4. add the vocab to the vocab dict at the end if you are using them 
'''
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

    for file, text in zip(["train", "dev", "test"], [train_csv, dev_csv, test_csv]):
        with open(os.path.join(".tmp", "{}.csv".format(file)), "w") as f:
            f.write(text)

    train, dev, test = data.TabularDataset.splits(path=".tmp", train='train.csv',
                                                    validation='dev.csv', test='test.csv',
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

    # current_iterator = data.Iterator.splits((train, dev, test), batch_sizes=(batch_size, 1, 1),
    #                                         sort_key=lambda x: len(x.form), sort_within_batch=True,
    #                                         device=device, repeat=False)

    current_iterator = [train_iterator, dev_iterator, test_iterator]

    sizes = {'vocab': len(FORM.vocab), 'postags': len(UPOS.vocab), 'deprels': len(DEPREL.vocab), 'feats': len(FEATS.vocab)}

    if args.use_chars:
        sizes['chars'] = len(CHAR.vocab)

    if args.semtag:
        sizes['semtags'] = len(SEM.vocab)

    return (current_iterator, sizes, [FORM.vocab, DEPREL.vocab, UPOS.vocab, FEATS.vocab])


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
