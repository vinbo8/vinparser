import sys
import torch
import numpy as np
import torch.nn.functional as F


class ConllLine:
    def __init__(self, line):
        self.line = line.rstrip("\n")
        line = self.line.split("\t")
        self.id, self.form, self.lemma, self.upos, self.xpos = line[0:5]
        self.feats, self.head, self.deprel, self.deps, self.misc = line[5:10]
        self.id, self.head = int(self.id), int(self.head)

    def __repr__(self):
        return self.line


class ConllBlock(list):
    def __init__(self):
        super().__init__()

    def append(self, p_object):
        if isinstance(p_object, ConllLine):
            super().append(p_object)
        else:
            raise TypeError("Elements must be ConllLine instances")

    def forms(self, separator=" "):
        return separator.join([line.form for line in self])

    def upos(self, separator=" "):
        return separator.join([line.upos for line in self])

    def rels(self):
        return [(line.head, line.id) for line in self]


class ConllParser(list):
    def __init__(self, buffer, orig=None, seed=42):
        super().__init__()
        if not orig:
            self.vocab, self.postags = [], []
        self.longest_sent = 0

        block = ConllBlock()
        for line in buffer:
            if line[0] == '#':
                continue

            if not line.strip():
                if len(block) > self.longest_sent:
                    self.longest_sent = len(block)
                self.append(block)
                block = ConllBlock()
                continue

            line = ConllLine(line)
            block.append(line)
            if not orig:
                self.vocab.append(line.form)
                self.postags.append(line.upos)

        if orig:
            self.vocab, self.postags = orig.vocab, orig.postags
            self.word_to_idx, self.pos_to_idx = orig.word_to_idx, orig.pos_to_idx

        else:
            self.vocab, self.postags = set(self.vocab), set(self.postags) #
            self.vocab_size = len(self.vocab) + 3   # PAD, ROOT, UNK
            self.pos_size = len(self.postags) + 2   # PAD, ROOT
            self.word_to_idx = {word: i + 2 for i, word in enumerate(self.vocab)}
            self.word_to_idx['ROOT'] = 1
            self.word_to_idx['PAD'] = 0
            self.word_to_idx['UNK'] = len(self.word_to_idx)

            self.pos_to_idx = {pos: i + 2 for i, pos in enumerate(self.postags)}
            self.pos_to_idx['ROOT'] = 1
            self.pos_to_idx['PAD'] = 0

        # weird
        self.longest_sent += 1

    def get_id(self, word):
        try:
            return self.word_to_idx[word]
        except KeyError:
            return self.word_to_idx['UNK']

    def get_pos_id(self, tag):
        return self.pos_to_idx[tag]

    def get_tensors(self):
        sents = [[self.get_id('ROOT')] + [self.get_id(word) for word in block.forms().split()] for block in self]
        tags = [[self.get_pos_id('ROOT')] + [self.get_pos_id(tag) for tag in block.upos().split()] for block in self]
        rels = [block.rels() for block in self]

        # sents, rels, tags = [list(i) for i in zip(*sorted(zip(sents, rels, tags), key=lambda x: len(x[1])))]

        # pad sents
        sents = torch.stack([F.pad(torch.LongTensor(sent), (0, self.longest_sent - len(sent))).data for sent in sents])
        rels = torch.stack([F.pad(torch.LongTensor(rel), (0, 0, 0, self.longest_sent - len(rel)), value=-1).data for rel in rels])
        tags = torch.stack([F.pad(torch.LongTensor(tag), (0, self.longest_sent - len(tag))).data for tag in tags])
        return sents, rels, tags

    def render(self):
        for block in self:
            for line in block:
                sys.stdout.write(line.line + "\n")
            sys.stdout.write("\n")


