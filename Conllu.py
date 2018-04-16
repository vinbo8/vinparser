import sys
import torch
import numpy as np
from collections import Counter
import torch.nn.functional as F

ROOT_LINE = "0\t__ROOT\t_\t__ROOT\t_\t_\t0\t__ROOT\t_\t_\n"


class ConllLine:
    def __init__(self, line):
        self.line = line.rstrip("\n")
        line = self.line.split("\t")
        print(line)
        self.id, self.form, self.lemma, self.upos, self.xpos = line[0:5]
        self.feats, self.head, self.deprel, self.deps, self.misc, self.sem = line[5:11]
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

    def forms(self, separator="-_-"):
        return separator.join([line.form for line in self])

    def upos(self, separator="-_-"):
        return separator.join([line.upos for line in self])

    def sem(self, separator="-_-"):
        return separator.join([line.sem for line in self])

    def heads(self):
        return [(line.head, line.id) for line in self]

    def deprels(self, separator="-_-"):
        return separator.join([line.deprel for line in self])


class ConllParser(list):
    def __init__(self, buffer, orig=None, seed=42):
        super().__init__()
        if not orig:
            # self.vocab, self.postags, self.deprels = [], [], []
            self.sizes = {'vocab': 0, 'postags': 0, 'deprels': 0, 'chars': 0, 'semtags': 0}
            self.sets = {'vocab': [], 'postags': [], 'deprels': [], 'chars': [], 'semtags': []}
            self.maps = {'vocab': {}, 'postags': {}, 'deprels': {}, 'chars': {}, 'semtags': {}}
        self.longest_sent = 0
        self.longest_word = 0

        block = ConllBlock(pad_with_root=True)
        for line in buffer:
            if line[0] == '#':
                continue

            if not line.strip():
                if len(block) > self.longest_sent:
                    self.longest_sent = len(block)
                for word in block.forms().split('-_-'):
                    if len(word) > self.longest_word:
                        self.longest_word = len(word)
                self.append(block)
                block = ConllBlock(pad_with_root=True)
                continue

            line = ConllLine(line)
            block.append(line)
            if not orig:
                self.sets['vocab'].append(line.form)
                self.sets['postags'].append(line.upos)
                self.sets['deprels'].append(line.deprel)
                self.sets['chars'].extend(list(line.form))
                self.sets['semtags'].append(line.sem)
        if orig:
            self.singleton_words = orig.singleton_words
            self.sets = orig.sets
            self.maps = orig.maps

        else:
            self.singleton_words = set(k for k, v in Counter(self.sets['vocab']).items() if v == 1)
            self.sets = {k: set(self.sets[k]) for k in self.sets}
            self.sizes = {k: len(self.sets[k]) + 3 for k in self.sets}   # PAD, ROOT, UNK
            self.maps = {k: {word: i + 3 for i, word in enumerate(self.sets[k])} for k in self.sets}
            for key in self.maps.keys():
                self.maps[key]['__PAD'] = 0
                self.maps[key]['__ROOT'] = 1
                self.maps[key]['__UNK'] = 2

        # weird
        self.longest_sent += 1

    def get_form_id(self, word):
        try:
            if word in self.singleton_words:
                raise KeyError
            return self.maps['vocab'][word]
        except KeyError:
            return self.maps['vocab']['__UNK']

    def get_char_id(self, char):
        try:
            return self.maps['chars'][char]
        except KeyError:
            return self.maps['chars']['__UNK']

    def get_pos_id(self, tag):
        return self.maps['postags'][tag]

    def get_sem_id(self, tag):
        return self.maps['semtags'][tag]

    def get_deprel_id(self, deprel):
        try:
            return self.maps['deprels'][deprel]
        except KeyError:
            return self.maps['deprels']['__UNK']

    def get_tensors(self):
        words, forms, chars, postags, deprels, heads, semtags = [], [], [], [], [], [], []
        # iterate thru blocks
        # returns file list of block lists
        for block in self:
            words.append([word for word in block.forms().split('-_-')])
            forms.append([self.get_form_id(word) for word in block.forms().split('-_-')])
            word_tensor = []
            for word in block.forms().split('-_-'):
                word_tensor.append(F.pad(torch.LongTensor([self.get_char_id(char) for char in word]), (0, self.longest_word - len(word))))
            chars.append(torch.stack(word_tensor))
            postags.append([self.get_pos_id(tag) for tag in block.upos().split('-_-')])
            semtags.append([self.get_sem_id(tag) for tag in block.sem().split('-_-')])
            deprels.append([self.get_deprel_id(deprel) for deprel in block.deprels().split('-_-')])
            heads.append(block.heads())

        package = zip(words, forms, chars, postags, deprels, heads, semtags)
        words, forms, chars, postags, deprels, heads , semtags = [], [], [], [], [], [], []
        for w, f, c, p, d, h in package:
            diff = self.longest_sent - len(f)
            for i in range(diff):
                w.append('__PAD')
            words.append(w)
            forms.append(F.pad(torch.LongTensor(f), (0, diff)).data)
            chars.append(F.pad(c, (0, 0, 0, diff)).data)
            postags.append(F.pad(torch.LongTensor(p), (0, diff)).data)
            semtags.append(F.pad(torch.LongTensor(p), (0, diff)).data)
            deprels.append(F.pad(torch.LongTensor(d), (0, diff)).data)
            heads.append(F.pad(torch.LongTensor(h), (0, 0, 0, diff)).data)
            assert forms[-1].size()[0] == postags[-1].size()[0] == deprels[-1].size()[0] == heads[-1].size()[0] == semtags[-1].size()[0]

        forms, chars, postags, deprels, heads, semtags = map(torch.stack, [forms, chars, postags, deprels, heads, semtags])

        return words, forms, chars, postags, deprels, heads, semtags

    def render(self):
        for block in self:
            for line in block:
                sys.stdout.write(line.line + "\n")
            sys.stdout.write("\n")


