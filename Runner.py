import sys
import torch
import argparse
import configparser
import Loader
from torch import nn
from types import SimpleNamespace
from sacred import Experiment
from sacred.observers import TelegramObserver
from Runnables import Parser, MTMapper

ex = Experiment('vinparser')

TAG_PARAMS, PARSE_PARAMS = {}, {}

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--name', action='store')
arg_parser.add_argument('--device', action='store', default=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
arg_parser.add_argument('--src', action='store')
arg_parser.add_argument('--src_dev', action='store')
arg_parser.add_argument('--mt', action='store')
arg_parser.add_argument('--trg_eval', action='store')
arg_parser.add_argument('--lang', action='store', default='de')
arg_parser.add_argument('--src_embed_file', action='store')
arg_parser.add_argument('--trg_embed_file', action='store')

params = arg_parser.add_argument_group('params')
params.add_argument('--batch_size', action='store', type=int, default=40)
params.add_argument('--embed_dim', action='store', type=int, default=300)
params.add_argument('--lstm_dim', action='store', type=int, default=500)
params.add_argument('--lstm_layers', action='store', type=int, default=1)
params.add_argument('--mlp_arc_dim', action='store', type=int, default=400)
params.add_argument('--mlp_label_dim', action='store', type=int, default=200)
params.add_argument('--dropout', action='store', type=float, default=0.33)
params.add_argument('--epochs', action='store', type=int, default=10)
params.add_argument('--lr', action='store', type=float, default=1e-3)

arg_parser.add_argument('--save_src', action='store')
arg_parser.add_argument('--load_src', action='store')
arg_parser.add_argument('--save_trg', action='store')
arg_parser.add_argument('--load_trg', action='store')

arg_parser.add_argument('--print_every', action='store', type=int, default=100)

ex.add_config({'args': vars(arg_parser.parse_args())})

if torch.cuda.is_available():
    ex.observers.append(TelegramObserver.from_config('/home/ravishankar/personal_work_troja/vinparser/telegram.json'))


@ex.main
def main(_run, args):
    args = SimpleNamespace(**args)

    train_iterator, fields, vocabs = Loader.get_iterators(args, args.src)
    if args.src_dev:
        dev_iterator, _, _ = Loader.get_iterators(args, args.src_dev, train_fields=fields)

    src_parser = Parser(args, vocabs).to(args.device)

    # load/train parser on source lang
    if args.load_src:
        print("loading source language parser..")
        with open(args.load_src, "rb") as f:
            src_parser.load_state_dict(torch.load(f, map_location=lambda storage, loc: storage))

    else:
        print("training source language parser..")
        for epoch in range(args.epochs):
            src_parser.train_(epoch, train_iterator)
            if args.src_dev:
                src_parser.evaluate_(dev_iterator)

        if args.save_src:
            print("saving source language parser..")
            with open(args.save_src, "wb") as f:
                torch.save(src_parser.state_dict(), f)

    # load MT data
    mt_iterator, trg_field, vocabs = Loader.get_mt(args, src_parser.vocabs['forms'])
    mapper = MTMapper(args, vocabs, src_parser)

    if args.load_trg:
        print("loading mapped target parser..")
        with open(args.load_trg, "rb") as f:
            src_parser.load_state_dict(torch.load(f, map_location=lambda storage, loc: storage))

    else:
        print("training mapped target parser..")
        for epoch in range(args.epochs):
            mapper.train_(epoch, mt_iterator)

        if args.save_trg:
            print("saving mapped target parser..")
            with open(args.save_trg, "wb") as f:
                torch.save(mapper.state_dict(), f)

    print("evaluating mapped parser..")
    fields = [(i, j) if i != 'form' else ('form', trg_field) for (i, j) in fields]
    eval_iterator, fields, vocabs = Loader.get_iterators(args, args.trg_eval, train_fields=fields)

    trg_parser = mapper.trg_parser
    trg_parser.evaluate_(eval_iterator)


ex.run()
