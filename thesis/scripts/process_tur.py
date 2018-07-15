import sys

id = 1
for line in sys.stdin:
    if not line.rstrip():
        id = 1
        sys.stdout.write(line)

    else:
        _, word, lang, postag = line.rstrip("\n").split("\t")
        sys.stdout.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(id, word, word, postag, "_", "_", "_", "_", "_", lang))
        id += 1
