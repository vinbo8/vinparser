import argparse
import configparser
import Loader
from Runnables import Tagger, Parser, CSParser, CLTagger, LangID


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--langid', action='store_true')
    arg_parser.add_argument('--tag', action='store_true')
    arg_parser.add_argument('--parse', action='store_true')
    arg_parser.add_argument('--config', default='./config.ini')
    arg_parser.add_argument('--train', action='store', nargs='*')
    arg_parser.add_argument('--dev', action='store', nargs='*')
    arg_parser.add_argument('--test', action='store', nargs='*')
    arg_parser.add_argument('--embed', action='store', nargs='*')
    arg_parser.add_argument('--lm', action='store', nargs='*')
    arg_parser.add_argument('--use_chars', action='store_true')
    arg_parser.add_argument('--use_cuda', action='store_true')
    # aux tasks
    arg_parser.add_argument('--semtag', action='store_true')
    arg_parser.add_argument('--code_switch', action='store_true')
    arg_parser.add_argument('--cl_tagger', action='store_true')
    arg_parser.add_argument('--random_bs', action='store', nargs='*')
    args = arg_parser.parse_args()

    # sanity checks
    # later, allow both tag and parse to do something like tag-first-parser
    assert args.tag + args.parse + args.langid == 1
    assert args.semtag + args.cl_tagger <= 1

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

    # =============================
    # Ignore these functions
    # Seriously, don't look at them
    # =============================
    def run_cl_tagger(args, iterators):
        assert len(iterators) == 2, "r u ok because this not how aux tasks work fam"

        main, aux = iterators[0], iterators[1]
        (train_loader_main, dev_loader_main, test_loader_main), sizes_main, vocab_main = main
        (train_loader_aux, dev_loader_aux, test_loader_aux), sizes_aux, vocab_aux = aux

        runnable = CLTagger(args, sizes_main, sizes_aux, vocab_main, vocab_aux)

        for epoch in range(EPOCHS):
            runnable.train_(epoch, train_loader_main, type_task="main")
            runnable.train_(epoch, train_loader_aux, type_task="aux")


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


    elif args.cl_tagger:
        iterators = Loader.get_iterators(args, BATCH_SIZE)
        run_cl_tagger(args, iterators)

    # ========================
    # Run language model thing
    # ========================
    elif args.lm:
        (train_loader, dev_loader, test_loader), lm_loader, sizes, vocab = Loader.get_treebank_and_txt(args, BATCH_SIZE)

        runnable = CSParser(sizes, args, embeddings=None, embed_dim=EMBED_DIM, lstm_dim=LSTM_DIM, lstm_layers=LSTM_LAYERS,
                            reduce_dim_arc=REDUCE_DIM_ARC, reduce_dim_label=REDUCE_DIM_LABEL, learning_rate=LEARNING_RATE)

        if args.use_cuda:
            runnable.cuda()

        print("Training")
        for epoch in range(EPOCHS):
            runnable.train_(epoch, lm_loader, task_type="aux")
            # runnable.train_(epoch, train_loader)
            runnable.evaluate_(dev_loader)

    # ========================
    # Normal stuff
    # ========================
    else:
        (train_loader, dev_loader, test_loader), sizes, vocab = Loader.get_iterators(args, BATCH_SIZE)[0]
        if args.parse:
            if args.code_switch:
                embeddings = vocab.vectors if args.embed else None
                runnable = CSParser(sizes, args, embeddings=embeddings, embed_dim=EMBED_DIM, lstm_dim=LSTM_DIM, lstm_layers=LSTM_LAYERS,
                                  reduce_dim_arc=REDUCE_DIM_ARC, reduce_dim_label=REDUCE_DIM_LABEL, learning_rate=LEARNING_RATE)
            else:
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
