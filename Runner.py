import sys
import torch
import argparse
import configparser
import Loader
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
arg_parser.add_argument('--trg', action='store')
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

ex.add_config({'args': vars(arg_parser.parse_args())})

@ex.main
def main(_run, args):
    args = SimpleNamespace(**args)

    train_iterator, fields, vocabs = Loader.get_iterators(args, args.src)
    if args.src_dev:
        dev_iterator, _, _ = Loader.get_iterators(args, args.src_dev, train_fields=fields)

    runnable = Parser(args, vocabs).to(args.device)

    # load/train parser on source lang
    if args.load_src:
        print("Loading")
        with open(args.load, "rb") as f:
            runnable.load_state_dict(torch.load(f, map_location=lambda storage, loc: storage))

    else:
        print("Training")
        for epoch in range(args.epochs):
            runnable.train_(epoch, train_iterator)
            if dev_iterator:
                runnable.evaluate_(dev_iterator)

    if args.save_src:
        print("Saving..")
        with open(args.save, "wb") as f:
            torch.save(runnable, f)

    # load MT data
    mt_iterator, vocabs = Loader.get_mt(args, runnable.vocabs['forms'])
    mapper = MTMapper(args, vocabs, runnable)

    print("Training mapper")
    for epoch in range(args.epochs):
        mapper.train_(epoch, mt_iterator)

    print("evaluating mapped parser..")


ex.run()
