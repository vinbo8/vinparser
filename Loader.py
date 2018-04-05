import codecs
from torchtext import data, datasets


ROOT_LINE = "0\t__ROOT\t_\t__ROOT\t_\t_\t0\t__ROOT\t_\t_"


def conll_to_csv(fname):
    with codecs.open(fname, 'r', 'utf-8') as f:
        rows, blokk = [], ['"' for _ in range(10)]
        blokk = list(map(lambda x, y: x + y, blokk, ROOT_LINE.split("\t")))
        for line in f:
            if line[0] == '#':
                continue

            if not line.rstrip():
                blokk = list(map(lambda x: x + '"', blokk))
                blokk = ",".join(blokk)
                rows.append(blokk)
                blokk = ['"' for _ in range(10)]
                blokk = list(map(lambda x, y: x + y, blokk, ROOT_LINE.split("\t")))
                continue

            cols = line.rstrip("\n").split("\t")
            blokk = list(map(lambda x, y: x + ',' + y, blokk, cols))

    return "\n".join(rows)


if __name__ == '__main__':
    a = conll_to_csv('data/sv-ud-train.conllu')
