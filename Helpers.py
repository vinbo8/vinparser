import os
import torch
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable
from Conllu import ConllParser
import sys

DEBUG_SIZE = -1


def write_tags_to_conllu(fname, tags, write_at):
    with open(fname, "r") as f:
        current_sent = 0
        for line in f:
            if line[0] == '#':
                if current_sent == write_at:
                    sys.stdout.write(line)
                continue
            
            if not line.rstrip():
                if current_sent == write_at:
                    sys.stdout.write(line)
                    break
                current_sent += 1
                continue

            if current_sent == write_at:
                cols = line.split("\t")
                id = cols[0]
                if '_' in id or '.' in id:
                    sys.stdout.write(line)
                    continue

                cols[3] = str(tags[int(id)])
                sys.stdout.write("\t".join(cols))


def write_to_conllu(fname, out_dict, deprels, write_at):
    with open(fname, "r") as f:
        current_sent = 0
        for line in f:
            if line[0] == '#':
                if current_sent == write_at:
                    sys.stdout.write(line)
                continue

            if not line.rstrip():
                if current_sent == write_at:
                    sys.stdout.write(line)
                    break
                current_sent += 1
                continue

            if current_sent == write_at:
                cols = line.split("\t")
                id = cols[0]
                # print line and skip
                if "-" in id or "." in id:
                    sys.stdout.write(line)
                    continue
                # ===

                cols[6] = str(out_dict[int(id)])
                cols[7] = str(deprels[int(id)])
                sys.stdout.write("\t".join(cols))


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

def spawn_bucket_vocab(loader, train=True):
        itos = []
        for sentence in loader.dataset.feats:
            for word in sentence:
                if word == '_':
                    continue
                else:
                    feats = word.split("|")
                    for feat in feats:
                        key = feat.split("=")[0]
                        if key not in itos:
                            itos.append(key)

        itos.append('<unk>') 
        stoi = {i: n for (n, i) in enumerate(itos)}
        return (itos, stoi)

def extract_batch_bucket_dict(batch, morph_vocab):
    batch_morph = batch.feats
    new_batch_tensor = []
    for sent_no, sentence in enumerate(batch_morph):
        sentence_tensor = []
        for word_no, word in enumerate(sentence):
            word = morph_vocab.itos[word.data[0]]
            if word in ['_', '<unk>', '<pad>']:
                sentence_tensor.append({})
            else:
                pairs = {k: v for (k, v) in [j.split("=") for j in word.split("|")]}
                sentence_tensor.append(pairs)

        new_batch_tensor.append(sentence_tensor)
    return new_batch_tensor


def extract_batch_bucket_class(batch, morph_vocab, bucket_itos, bucket_stoi):
    batch_morph = batch.feats
    new_batch_tensor = []
    # get vectors
    for sent_no, sentence in enumerate(batch_morph):
        sentence_tensor = [] 
        for word_no, word in enumerate(sentence):
            word = morph_vocab.itos[word.data[0]]
            if word == '_':
                sentence_tensor.append(bucket_stoi['_'])

            elif word == '<pad>':
                sentence_tensor.append(bucket_stoi['<pad>'])

            else:
                feats = "&".join([j.split("=")[0] for j in word.split("|")])
                try:
                    sentence_tensor.append(bucket_stoi[feats])
                    # check whether this is necessary - maybe just don't bother with unknown features in test
                    # seeing as you can't really predict a value for an unknown key anyway
                except KeyError:
                    sentence_tensor.append(bucket_stoi['<unk>'])

        new_batch_tensor.append(Variable(torch.LongTensor(sentence_tensor)))

    return torch.stack(new_batch_tensor)

def extract_batch_bucket_vector(batch, morph_vocab, bucket_itos, bucket_stoi):
    default_feat_vector = torch.LongTensor([False for i in bucket_itos])
    batch_morph = batch.feats
    new_batch_tensor = []
    # get vectors
    for sent_no, sentence in enumerate(batch_morph):
        sentence_tensor = [] 
        for word_no, word in enumerate(sentence):
            word = morph_vocab.itos[word.data[0]]
            if word == '_':
                sentence_tensor.append(Variable(default_feat_vector.clone()))

            elif word == '<pad>':
                current_feat_vector = default_feat_vector.clone()
                current_feat_vector[bucket_stoi['<pad>']] = True
                sentence_tensor.append(Variable(current_feat_vector))

            else:
                current_feat_vector = default_feat_vector.clone()
                feats = word.split("|")
                for feat in feats:
                    key = feat.split("=")[0]
                    try:
                        current_feat_vector[bucket_stoi[key]] = True
                    # check whether this is necessary - maybe just don't bother with unknown features in test
                    # seeing as you can't really predict a value for an unknown key anyway
                    except KeyError:
                        current_feat_vector[bucket_stoi['<unk>']] = True

                sentence_tensor.append(Variable(current_feat_vector))
        new_batch_tensor.append(torch.stack(sentence_tensor))

    return torch.stack(new_batch_tensor)
