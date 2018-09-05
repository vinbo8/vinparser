import sys
import torch
import argparse
import configparser
import Loader
from types import SimpleNamespace
from sacred import Experiment
from sacred.observers import TelegramObserver
from Runnables import Parser, LangID, LangSwitch

ex = Experiment('vinparser')

ex.observers.append(TelegramObserver.from_config('./telegram.json'))

TAG_PARAMS, PARSE_PARAMS = {}, {}

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--debug_new', action='store_true')
arg_parser.add_argument('--langid', action='store_true')
arg_parser.add_argument('--use_misc', action='store_true')
arg_parser.add_argument('--fix_embeds', action='store_true')
arg_parser.add_argument('--parse', action='store_true')
arg_parser.add_argument('--config', default='./config.ini')
arg_parser.add_argument('--save', action='store')
arg_parser.add_argument('--load', action='store')
arg_parser.add_argument('--train', action='store')
arg_parser.add_argument('--dev', action='store')
arg_parser.add_argument('--test', action='store')
arg_parser.add_argument('--outfile', action='store')
arg_parser.add_argument('--embed', action='store')
arg_parser.add_argument('--use_chars', action='store_true')
arg_parser.add_argument('--use_cuda', action='store_true')
arg_parser.add_argument('--elmo', action='store_true')
# aux tasks
arg_parser.add_argument('--semtag', action='store_true')
arg_parser.add_argument('--lm', action='store_true')
ex.add_config({'args': vars(arg_parser.parse_args())})

@ex.main
def main(_run, args):
    args = SimpleNamespace(**args)
    config = configparser.ConfigParser()
    config.read(args.config)

    # parser
    PARSE_BATCH_SIZE = int(config['parser']['BATCH_SIZE'])
    PARSE_EMBED_DIM = int(config['parser']['EMBED_DIM'])
    PARSE_LSTM_DIM = int(config['parser']['LSTM_DIM'])
    PARSE_LSTM_LAYERS = int(config['parser']['LSTM_LAYERS'])
    PARSE_REDUCE_DIM_ARC = int(config['parser']['REDUCE_DIM_ARC'])
    PARSE_REDUCE_DIM_LABEL = int(config['parser']['REDUCE_DIM_LABEL'])
    PARSE_LEARNING_RATE = float(config['parser']['LEARNING_RATE'])

    # tagger
    TAG_BATCH_SIZE = int(config['tagger']['BATCH_SIZE'])
    TAG_EMBED_DIM = int(config['tagger']['EMBED_DIM'])
    TAG_LSTM_DIM = int(config['tagger']['LSTM_DIM'])
    TAG_LSTM_LAYERS = int(config['tagger']['LSTM_LAYERS'])
    TAG_LEARNING_RATE = float(config['tagger']['LEARNING_RATE'])
    TAG_MLP_DIM = int(config['tagger']['MLP_DIM'])

    PARSE_EPOCHS = int(config['parser']['EPOCHS'])
    TAG_EPOCHS = int(config['tagger']['EPOCHS'])

    # set seeds
    torch.manual_seed(1337)
    if args.use_cuda:
        torch.backends.cudnn.enabled = False
        torch.cuda.manual_seed(1337)
        torch.cuda.manual_seed_all(1337)

    # ========================
    # Normal stuff
    # ========================
    (train_loader, dev_loader, test_loader), sizes, vocab = Loader.get_iterators(args, PARSE_BATCH_SIZE)

    # ============
    # Wall of code
    # ============
    if args.parse:
        runnable = Parser(args, sizes, vocab, embed_dim=PARSE_EMBED_DIM, lstm_dim=PARSE_LSTM_DIM, lstm_layers=PARSE_LSTM_LAYERS,
                            reduce_dim_arc=PARSE_REDUCE_DIM_ARC, reduce_dim_label=PARSE_REDUCE_DIM_LABEL, learning_rate=PARSE_LEARNING_RATE)

    elif args.debug_new:
        runnable = LangSwitch(args, sizes, vocab)

    # cuda
    if args.use_cuda:
        runnable.cuda()

    # training
    if args.load:
        print("Loading")
        with open(args.load, "rb") as f:
            runnable.load_state_dict(torch.load(f, map_location=lambda storage, loc: storage))

    else:
        print("Training")
        for epoch in range(PARSE_EPOCHS):
            runnable.train_(epoch, train_loader)
            runnable.evaluate_(dev_loader)

    # test
    print("Eval")
    runnable.evaluate_(test_loader, print_conll=True)

    if args.save:
        print("Saving..")
        with open(args.save, "wb") as f:
            torch.save(runnable, f)

ex.run()
