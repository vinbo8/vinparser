import argparse
import configparser
import Loader
from Runnables import Tagger, Parser


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--tag', action='store_true')
    arg_parser.add_argument('--parse', action='store_true')
    arg_parser.add_argument('--config', default='./config.ini')
    arg_parser.add_argument('--train', action='store', nargs='*')
    arg_parser.add_argument('--dev', action='store', nargs='*')
    arg_parser.add_argument('--test', action='store', nargs='*')
    arg_parser.add_argument('--embed', action='store', nargs='*')
    arg_parser.add_argument('--use_chars', action='store_true')
    arg_parser.add_argument('--use_cuda', action='store_true')
    # aux tasks
    arg_parser.add_argument('--semtag', action='store_true')
    args = arg_parser.parse_args()

    # sanity check
    # later, allow both tag and parse to do something like tag-first-parser
    assert args.tag + args.parse == 1

    config = configparser.ConfigParser()
    config.read(args.config)

    BATCH_SIZE = int(config['parser']['BATCH_SIZE'])
    EMBED_DIM = int(config['parser']['EMBED_DIM'])
    LSTM_DIM = int(config['parser']['LSTM_DIM'])
    LSTM_LAYERS = int(config['parser']['LSTM_LAYERS'])
    REDUCE_DIM_ARC = int(config['parser']['REDUCE_DIM_ARC'])
    REDUCE_DIM_LABEL = int(config['parser']['REDUCE_DIM_LABEL'])
    LEARNING_RATE = float(config['parser']['LEARNING_RATE'])
    MLP_DIM = int(config['tagger']['MLP_DIM'])
    EPOCHS = int(config['parser']['EPOCHS'])

    # args
    (train_loader, dev_loader, test_loader), sizes = Loader.get_iterators(args, BATCH_SIZE)
    if args.parse:
        runnable = Parser(sizes, args, embeddings=None, embed_dim=EMBED_DIM, lstm_dim=LSTM_DIM, lstm_layers=LSTM_LAYERS,
                          reduce_dim_arc=REDUCE_DIM_ARC, reduce_dim_label=REDUCE_DIM_LABEL, learning_rate=LEARNING_RATE)
    elif args.tag:
        runnable = Tagger(sizes, args, embeddings=None, embed_dim=EMBED_DIM, lstm_dim=LSTM_DIM, lstm_layers=LSTM_LAYERS,
                          mlp_dim=MLP_DIM, learning_rate=LEARNING_RATE)

    if args.use_cuda:
        runnable.cuda()

    # training
    print("Training")
    for epoch in range(EPOCHS):
        runnable.train_(epoch, train_loader)
        runnable.evaluate_(dev_loader)

    # test
    print("Eval")
    runnable.evaluate_(test_loader)
