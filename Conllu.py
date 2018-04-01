import sys
import torch
import numpy as np
import torch.nn.functional as F

ROOT_LINE = "0\t__ROOT\t_\t__ROOT\t_\t_\t0\t__ROOT\t_\t_\n"


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
    def __init__(self, pad_with_root=False):
        super().__init__()
        if pad_with_root:
            line = ConllLine(ROOT_LINE)
            super().append(line)

    def append(self, p_object):
        if isinstance(p_object, ConllLine):
            super().append(p_object)
        else:
            raise TypeError("Elements must be ConllLine instances")

    def forms(self, separator="---"):
        return separator.join([line.form for line in self])

    def upos(self, separator="---"):
        return separator.join([line.upos for line in self])

    def heads(self):
        return [(line.head, line.id) for line in self]

    def deprels(self, separator="---"):
        return separator.join([line.deprel for line in self])


class ConllParser(list):
    def __init__(self, buffer, orig=None, seed=42):
        super().__init__()
        if not orig:
            # self.vocab, self.postags, self.deprels = [], [], []
            self.sizes = {'vocab': 0, 'postags': 0, 'deprels': 0}
            self.sets = {'vocab': [], 'postags': [], 'deprels': []}
            self.maps = {'vocab': {}, 'postags': {}, 'deprels': {}}
        self.longest_sent = 0

        block = ConllBlock(pad_with_root=True)
        for line in buffer:
            if line[0] == '#':
                continue

            if not line.strip():
                if len(block) > self.longest_sent:
                    self.longest_sent = len(block)
                self.append(block)
                block = ConllBlock(pad_with_root=True)
                continue

            line = ConllLine(line)
            block.append(line)
            if not orig:
                self.sets['vocab'].append(line.form)
                self.sets['postags'].append(line.upos)
                self.sets['deprels'].append(line.deprel)

        if orig:
            self.sets = orig.sets
            self.maps = orig.maps

        else:
            self.sets = {k: set(self.sets[k]) for k in self.sets}
            self.sizes = {k: len(self.sets[k]) + 3 for k in self.sets}   # PAD, ROOT, UNK
            self.maps = {k: {word: i + 3 for i, word in enumerate(self.sets[k])} for k in self.sets}
            for key in self.maps.keys():
                self.maps[key]['PAD'] = 0
                self.maps[key]['__ROOT'] = 1
                self.maps[key]['UNK'] = 2

        # weird
        self.longest_sent += 1

    def get_id(self, word):
        try:
            return self.maps['vocab'][word]
        except KeyError:
            return self.maps['vocab']['UNK']

    def get_pos_id(self, tag):
        return self.maps['postags'][tag]

    def get_deprel_id(self, deprel):
        try:
            return self.maps['deprels'][deprel]
        except KeyError:
            return self.maps['deprels']['UNK']

    def get_tensors(self):
        words, forms, postags, deprels, heads = [], [], [], [], []
        # iterate thru blocks
        # returns file list of block lists
        for block in self:
            words.append([word for word in block.forms().split('---')])
            forms.append([self.get_id(word) for word in block.forms().split('---')])
            postags.append([self.get_pos_id(tag) for tag in block.upos().split('---')])
            deprels.append([self.get_deprel_id(deprel) for deprel in block.deprels().split('---')])
            heads.append(block.heads())

        package = zip(words, forms, postags, deprels, heads)
        words, forms, postags, deprels, heads = [], [], [], [], []
        for w, f, p, d, h in package:
            diff = self.longest_sent - len(f)
            for i in range(diff):
                w.append('PAD')
            words.append(w)
            forms.append(F.pad(torch.LongTensor(f), (0, diff)).data)
            postags.append(F.pad(torch.LongTensor(p), (0, diff)).data)
            deprels.append(F.pad(torch.LongTensor(d), (0, diff)).data)
            heads.append(F.pad(torch.LongTensor(h), (0, 0, 0, diff)).data)
            assert forms[-1].size()[0] == postags[-1].size()[0] == deprels[-1].size()[0] == heads[-1].size()[0]

        forms, postags, deprels, heads = map(torch.stack, [forms, postags, deprels, heads])

        return words, forms, postags, deprels, heads

    def render(self):
        for block in self:
            for line in block:
                sys.stdout.write(line.line + "\n")
            sys.stdout.write("\n")


