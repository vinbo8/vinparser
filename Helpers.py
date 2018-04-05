import os
import torch
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable
from Conllu import ConllParser

DEBUG_SIZE = -1


def build_data(fname, batch_size, train_conll=None):
    # build data
    with open(fname, 'r') as f:
        conll = ConllParser(f) if not train_conll else ConllParser(f, train_conll)

    # sentences
    print("Preparing %s.." % fname)
    # rels turns into heads later
    words, forms, chars, tags, deprels, rels = conll.get_tensors()
    assert forms.shape == torch.Size([len(conll), conll.longest_sent])
    assert tags.shape == torch.Size([len(conll), conll.longest_sent])
    assert deprels.shape == torch.Size([len(conll), conll.longest_sent])

    # heads
    heads = -torch.ones(forms.shape[0], conll.longest_sent)
    heads.scatter_(1, rels[:, :, 1], rels[:, :, 0].type(torch.FloatTensor))
    heads[:, 0] = 0
    heads = heads.type(torch.LongTensor)

    assert heads.shape == torch.Size([len(conll), conll.longest_sent])

    # sizes
    sizes_int = torch.zeros(len(conll)).view(-1, 1).type(torch.LongTensor)
    sizes = torch.zeros(len(conll), conll.longest_sent)
    for n, form in enumerate(forms):
        sizes_int[n] = form[form != 0].shape[0]

    for n, size in enumerate(sizes_int):
        sizes[n, 1:size[0]] = 1

    assert sizes.shape == torch.Size([len(conll), conll.longest_sent])

    # build loader & model
    data = list(zip(forms, tags, chars, heads, deprels, sizes))[:DEBUG_SIZE]
    loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True)

    return conll, loader


def process_batch(batch, cuda=False):
    forms, tags, chars, heads, deprels, sizes = [torch.stack(list(i)) for i in zip(*sorted(zip(*batch),
                                                                            key=lambda x: x[5].nonzero().size(0),
                                                                            reverse=True))]
    trunc = max([i.nonzero().size(0) + 1 for i in sizes])
    x_forms = Variable(forms[:, :trunc])
    x_tags = Variable(tags[:, :trunc])
    x_chars = Variable(chars[:, :trunc])
    mask = Variable(sizes[:, :trunc])
    pack = Variable(torch.LongTensor([i.nonzero().size(0) + 1 for i in sizes]))
    y_heads = Variable(heads[:, :trunc], requires_grad=False)
    y_deprels = Variable(deprels[:, :trunc], requires_grad=False)

    output = [x_forms, x_tags, x_chars, mask, pack, y_heads, y_deprels]
    if cuda:
        return [i.cuda() for i in output]
    return output


def extract_best_label_logits(pred_arcs, label_logits, lengths):
    pred_arcs = pred_arcs.data
    size = label_logits.size()
    output_logits = Variable(torch.zeros(size[0], size[1], size[3]))

    if label_logits.is_cuda:
        output_logits = output_logits.cuda()

    for batch_index, (_logits, _arcs, _length) in enumerate(zip(label_logits, pred_arcs, lengths)):
        for i in range(int(_length)):
            output_logits[batch_index] = _logits[_arcs[i]]
    return output_logits


def build_character_dict(vocab):
    charset = []
    longest_word_len = -1
    for sentence in vocab:
        for word in sentence:
            if word.startswith('__'):
                continue
            word = list(word)
            charset.extend(word)
            if len(word) > longest_word_len:
                longest_word_len = len(word)

    charset = set(charset)
    char_dict = {char: i + 1 for i, char in enumerate(charset)}

    out = []
    for sentence in vocab:
        sent_tensor = []
        for word in sentence:
            if word == '__ROOT':
                word_tensor = F.pad(Variable(torch.LongTensor([1])), (0, longest_word_len - 1))
            elif word == '__PAD':
                word_tensor = Variable(torch.zeros(longest_word_len).type(torch.LongTensor))
            else:
                try:
                    word_tensor = F.pad(Variable(torch.LongTensor([safe_char_lookup(char_dict, char) for char in word])), (0, longest_word_len - len(word)))
                except AssertionError:
                    temp = torch.LongTensor([safe_char_lookup(char_dict, char) for char in word])
                    pass
            sent_tensor.append(word_tensor)
        out.append(torch.stack(sent_tensor))
    out = torch.stack(out)

    return char_dict


def safe_char_lookup(char_dict, char):
    try:
        return char_dict[char]
    except KeyError:
        return 2
