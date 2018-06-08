import sys
import torch
import argparse
import configparser
import Loader
from Runnables import Tagger, Parser, CSParser, LangID, CLTagger, Analyser


if __name__ == '__main__':
    TAG_PARAMS, PARSE_PARAMS = {}, {}

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--langid', action='store_true')
    arg_parser.add_argument('--tag', action='store_true')
    arg_parser.add_argument('--parse', action='store_true')
    arg_parser.add_argument('--morph', action='store_true')
    arg_parser.add_argument('--config', default='./config.ini')
    arg_parser.add_argument('--save', action='store')
    arg_parser.add_argument('--load', action='store')
    arg_parser.add_argument('--train', action='store')
    arg_parser.add_argument('--dev', action='store')
    arg_parser.add_argument('--test', action='store')
    arg_parser.add_argument('--embed', action='store')
    arg_parser.add_argument('--use_chars', action='store_true')
    arg_parser.add_argument('--use_cuda', action='store_true')
    # aux tasks
    arg_parser.add_argument('--semtag', action='store_true')
    arg_parser.add_argument('--lm', action='store_true')
    args = arg_parser.parse_args()

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

    # ==========================
    # Actual loading begins here
    # Start with args
    # ==========================
    if args.langid:
        score = 0
        (train_loader, dev_loader, test_loader), sizes = Loader.load_pos()
        runnable = LangID(args, sizes)

        print("Training..")
        for epoch in range(1):
            runnable.train_(epoch, train_loader)

        runnable.evaluate_(test_loader)


    # ========================
    # Run language model thing
    # ========================
    elif args.lm:
        (train_loader, dev_loader, test_loader), lm_loader, sizes, vocab = Loader.get_treebank_and_txt(args, PARSE_BATCH_SIZE)

        runnable = CSParser(sizes, args, vocab, embeddings=vocab, embed_dim=PARSE_EMBED_DIM, lstm_dim=PARSE_LSTM_DIM, lstm_layers=PARSE_LSTM_LAYERS,
                            reduce_dim_arc=PARSE_REDUCE_DIM_ARC, reduce_dim_label=PARSE_REDUCE_DIM_LABEL, learning_rate=PARSE_LEARNING_RATE)

        if args.use_cuda:
            runnable.cuda()

        print("Training")
        for epoch in range(PARSE_EPOCHS):
            runnable.train_(epoch, lm_loader, task_type="aux")
            # runnable.train_(epoch, train_loader)
            runnable.evaluate_(dev_loader)

    # ========================
    # Normal stuff
    # ========================
    else:
        (train_loader, dev_loader, test_loader), sizes, vocab = Loader.get_iterators(args, PARSE_BATCH_SIZE)

        # ============
        # Wall of code
        # ============
        if args.parse and args.tag:
            runnable = Tagger(sizes, args, vocab, chain=True, embeddings=None, embed_dim=TAG_EMBED_DIM, lstm_dim=TAG_LSTM_DIM, lstm_layers=TAG_LSTM_LAYERS,
                              mlp_dim=TAG_MLP_DIM, learning_rate=TAG_LEARNING_RATE)

            if args.use_cuda: runnable.cuda()

            print("Training tagger")
            for epoch in range(TAG_EPOCHS):
                runnable.train_(epoch, train_loader)
                runnable.evaluate_(dev_loader)

            # test
            print("Evaluating tagger")
            tag_tensors = runnable.evaluate_(test_loader, print_conll=True)
            test_loader.data().upos = (i for i in tag_tensors)

            runnable = Parser(args, sizes, vocab, embed_dim=PARSE_EMBED_DIM, lstm_dim=PARSE_LSTM_DIM, lstm_layers=PARSE_LSTM_LAYERS,
                              reduce_dim_arc=PARSE_REDUCE_DIM_ARC, reduce_dim_label=PARSE_REDUCE_DIM_LABEL, learning_rate=PARSE_LEARNING_RATE)
            
            if args.use_cuda: runnable.cuda()

            print("Training parser")
            for epoch in range(PARSE_EPOCHS):
                runnable.train_(epoch, train_loader)
                runnable.evaluate_(dev_loader)

            # test
            print("Evaluating parser")
            runnable.evaluate_(test_loader, print_conll=True)
            sys.exit() 

        elif args.parse:
            runnable = Parser(args, sizes, vocab, embed_dim=PARSE_EMBED_DIM, lstm_dim=PARSE_LSTM_DIM, lstm_layers=PARSE_LSTM_LAYERS,
                              reduce_dim_arc=PARSE_REDUCE_DIM_ARC, reduce_dim_label=PARSE_REDUCE_DIM_LABEL, learning_rate=PARSE_LEARNING_RATE)
        elif args.tag:
            runnable = Tagger(sizes, args, vocab, embeddings=None, embed_dim=TAG_EMBED_DIM, lstm_dim=TAG_LSTM_DIM, lstm_layers=TAG_LSTM_LAYERS,
                              mlp_dim=TAG_MLP_DIM, learning_rate=TAG_LEARNING_RATE)

        elif args.morph:
            runnable = Analyser(sizes, args, vocab)

        if args.use_cuda:
            runnable.cuda()

        # training
        if args.load:
            print("Loading")
            with open(args.load, "rb") as f:
                runnable.load_state_dict(torch.load(f, map_location=lambda storage, loc: storage))

        else:
            print("Training")
            for epoch in range(TAG_EPOCHS):
                runnable.train_(epoch, train_loader)
                runnable.evaluate_(dev_loader)

        # test
        print("Eval")
        runnable.evaluate_(test_loader, print_conll=True)
