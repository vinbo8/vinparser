import sys

class ConllLine:
    def __init__(self, line):
        self.line = line.rstrip("\n")
        line = self.line.split("\t")
        self.id, self.form, self.lemma, self.upos, self.xpos = line[0:5]
        self.feats, self.head, self.deprel, self.deps, self.misc = line[5:10]

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

    def raw(self, separator=" "):
        return separator.join([line.form for line in self])


class ConllParser(list):
    def __init__(self):
        super().__init__()

    def build(self, buffer):
        block = ConllBlock()
        for line in buffer:
            # skip comments for now
            if line[0] == '#':
                continue

            if not line.strip():
                self.append(block)
                block = ConllBlock()
                continue

            block.append(ConllLine(line))

    def render(self):
        for block in self:
            for line in block:
                sys.stdout.write(line.line + "\n")
            sys.stdout.write("\n")


