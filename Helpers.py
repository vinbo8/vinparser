import os
import torch
from conllu import ConllParser

DEBUG_SIZE = 1000


def build_data(fname, batch_size, train_conll=None):
    # build data
    with open(os.path.join('data', fname), 'r') as f:
        conll = ConllParser(f) if not train_conll else ConllParser(f, train_conll)

    # sentences
    print("Preparing %s.." % fname)
    forms, rels, tags = conll.get_tensors()
    assert forms.shape == torch.Size([len(conll), conll.longest_sent])
    assert tags.shape == torch.Size([len(conll), conll.longest_sent])

    # labels
    labels = -torch.ones(forms.shape[0], conll.longest_sent, 1)
    for batch_no, _ in enumerate(rels):
        for rel in rels[batch_no]:
            # if rel[1] == 0:
            #   con
            labels[batch_no, rel[1]] = rel[0]

    labels = torch.squeeze(labels.type(torch.LongTensor))
    assert labels.shape == torch.Size([len(conll), conll.longest_sent])

    # sizes
    sizes_int = torch.zeros(len(conll)).view(-1, 1).type(torch.LongTensor)
    sizes = torch.zeros(len(conll), conll.longest_sent)
    for n, form in enumerate(forms):
        sizes_int[n] = form[form != 0].shape[0]

    for n, size in enumerate(sizes_int):
        sizes[n, 1:size[0]] = 1

    assert sizes.shape == torch.Size([len(conll), conll.longest_sent])

    # build loader & model
    data = list(zip(forms, tags, labels, sizes))[:DEBUG_SIZE]
    loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True)

    return conll, loader