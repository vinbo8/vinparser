import copy
from functools import reduce
import json
import os
import re
import tarfile
import tempfile
import numpy as np
import codecs
from collections import defaultdict
np.random.seed(1337)  # for reproducibility

import keras
import keras.backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Input, Dense, Dropout, AveragePooling1D, LSTM, Conv1D, MaxPooling1D, Flatten, Embedding, GRU, TimeDistributed, Reshape,  Bidirectional, Concatenate, Multiply, Subtract, Lambda
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.layers.wrappers import Bidirectional
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.regularizers import l2
from keras.utils import np_utils
from keras.initializers import TruncatedNormal, glorot_uniform, glorot_normal
from shared import WeightedCombinationLayer, WeightedCombination, WeightedCombinationLayerUnb


def c2i(sents, max_seq, MAX_CHAR):
    c2i = defaultdict(lambda: len(c2i))
    for i, sent in enumerate(sents):
        sent = sent.split(" ")
        sent = sent[:max_seq]
        for j, word in enumerate(sent):
            if len(word) > MAX_CHAR - 2:
                quarter = int(MAX_CHAR/4) - 1
                word = word[:quarter] + word[-quarter:]
            for k, char in enumerate(word):
                c2i[char]
    return c2i

def proc_chars(sents, max_seq, MAX_CHAR, c2i):
    #c2i = defaultdict(lambda: len(c2i))
    #max_word = get_max_word_len(sents)

    print(sents[0])
    chars = np.zeros((len(sents), max_seq, MAX_CHAR))
    print(chars.shape)

    for i, sent in enumerate(sents):
       # print(len(sent))
        sent = sent.split(" ")
        print(len(sent))
        sent = sent[:max_seq] 
        for j, word in enumerate(sent):
            #print(word)
            if len(word) > MAX_CHAR - 2:
                quarter = int(MAX_CHAR/4) - 1
                word = word[:quarter] + word[-quarter:]
            for k, char in enumerate(word):
                #print(k)
                chars[i][j][k] = c2i[char]
    return chars, MAX_CHAR

def read_embeds(efile, word_index):
    embedding_index = {}
    f = open(efile, "r")
    for line in f:
        values = line.split(' ')
        embedding_dim = len(values)-1
        word = values[0].strip()
        coefs = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = coefs
    f.close()

    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix, embedding_dim

#get tag2id 
def t2i(filenames):
    tag_to_id = defaultdict(lambda: len(tag_to_id))
    for file in filenames:
        file = codecs.open(file)
        for line in file.readlines():
            if line.strip():
                separated = line.split()
                tag_to_id[separated[1].strip()]
    print("tag_to_id): ", tag_to_id)
    return tag_to_id

def w2i(filenames):
    word_to_id = defaultdict(lambda: len(word_to_id))
    for file in filenames:
        file = codecs.open(file)
        for line in file.readlines():
            if line.strip():
                separated = line.split()
                word_to_id[separated[0].strip()]
    return word_to_id

def get_tag_full_data(filename, MAX_LEN, tag_to_id, word_to_id):
  file = codecs.open(filename)
  sent_X, sent_Y, curr_words, curr_tags, curr_text, text_X = [],[],[],[],[],[]

  for line in file.readlines():
      if line.strip():
          separated = line.split()
          curr_words.append(word_to_id[separated[0].strip()])
          curr_tags.append(tag_to_id[separated[1].strip()])
          curr_text.append(separated[0].strip())
      else:
          sent_Y.append(curr_tags)
          sent_X.append(curr_words)
          text_X.append(' '.join(curr_text))
          curr_words, curr_tags, curr_text = [], [], []

  sent_X = pad_sequences(sent_X,  MAX_LEN)
  sent_Y = pad_sequences(sent_Y,  MAX_LEN)

  print("len(tag_to_id): ", len(tag_to_id))
  print("len(word_to_id): ", len(word_to_id))

  Y_tag = np_utils.to_categorical(sent_Y, len(tag_to_id))
  print("Y_TAG: ", Y_tag.shape)
  return   text_X, sent_X, Y_tag

def make_model(word_index, max_seq, embeds, char_index, MAX_CHAR, embed_dim, ACTIVATION, L2, DP, SENT_REP_SIZE, NO_TAGS_ud):

    embedding_layer = Embedding(len(word_index) + 1,
                            embed_dim,
                            weights = [embeds],
                            input_length=max_seq,
                            trainable = TRAIN_EMBED)

    embedding_layer_rand = Embedding(len(word_index) + 1,
                            int(embed_dim / 2),
                            input_length=max_seq,
                            trainable = TRAIN_EMBED)

    char_embedding_layer = TimeDistributed(Embedding(len(char_index) + 1,
                                                         output_dim = 32,
                                                         input_length = MAX_CHAR))

    input = Input(shape=(max_seq,), dtype='int32')
    char_input = Input(shape=(max_seq, MAX_CHAR,), dtype='int32')

    embedded_sequences =  embedding_layer(input)
    embedded_sequences_rand = embedding_layer_rand(input)   
    #char
    embedded_chars = char_embedding_layer(char_input)
    embedded_chars = TimeDistributed(Flatten())(embedded_chars)
    embedded_chars = Bidirectional(LSTM(25, activation="relu", return_sequences = True))(embedded_chars)

    embedded_sequences = keras.layers.concatenate([embedded_sequences, embedded_sequences_rand, embedded_chars], axis = -1) 

    #layer 1
    tag_f = LSTM(SENT_REP_SIZE, activation="relu", dropout=DP, recurrent_dropout=DP,
                              return_sequences = True)
    tag_b = LSTM(SENT_REP_SIZE, activation="relu", dropout=DP, recurrent_dropout=DP,
                              return_sequences = True, go_backwards = True)

    seqs_f_ud = tag_f(embedded_sequences)
    seqs_b_ud = tag_b(embedded_sequences)
   
    #layer 2 
  #  tag_f_2 = LSTM(SENT_REP_SIZE, activation="relu", dropout=DP, recurrent_dropout=DP,
    #                          return_sequences = True)
   # tag_b_2 = LSTM(SENT_REP_SIZE, activation="relu", dropout=DP, recurrent_dropout=DP,
     #                         return_sequences = True, go_backwards = True)

    #seqs_f_ud_2 = tag_f_2(seqs_f_ud)
    #seqs_b_ud_2 = tag_b_2(seqs_b_ud)

    seqs_ud_2 = keras.layers.concatenate([seqs_f_ud, seqs_b_ud], axis = -1)

    #seqs_ud_2 = Dropout(DP)(seqs_ud_2)
    
    tags_ud = TimeDistributed(Dense(NO_TAGS_ud, activation='softmax'))(seqs_ud_2)

    return  input,  char_input, tags_ud

MAX_LEN = 42
MAX_CHAR = 32
#get tag2id sem
#all files for joint word index
allfnames = ['/data/s3094723/thesis/pos/CCGbank1.2/ccgbank.train.simple', '/data/s3094723/thesis/pos/CCGbank1.2/ccgbank.dev.simple'
, '/data/s3094723/thesis/pos/CCGbank1.2/ccgbank.test.simple']
word_to_id = w2i(allfnames)

#ud data
fnames_ud = ['/data/s3094723/thesis/pos/CCGbank1.2/ccgbank.train.simple', '/data/s3094723/thesis/pos/CCGbank1.2/ccgbank.dev.simple'
, '/data/s3094723/thesis/pos/CCGbank1.2/ccgbank.test.simple']
tag_to_id_ud = t2i(fnames_ud)

train_X_sent_ud, train_X_ud, train_Y_ud = get_tag_full_data('/data/s3094723/thesis/pos/CCGbank1.2/ccgbank.train.simple', MAX_LEN, tag_to_id_ud, word_to_id)
dev_X_sent_ud, dev_X_ud, dev_Y_ud = get_tag_full_data('/data/s3094723/thesis/pos/CCGbank1.2/ccgbank.dev.simple', MAX_LEN, tag_to_id_ud, word_to_id)
test_X_sent_ud, test_X_ud, test_Y_ud = get_tag_full_data('/data/s3094723/thesis/pos/CCGbank1.2/ccgbank.test.simple', MAX_LEN, tag_to_id_ud, word_to_id)

tokenizer = Tokenizer(lower=False, filters = '')
tokenizer.fit_on_texts(train_X_sent_ud +  dev_X_sent_ud + test_X_sent_ud)

#set variables
NO_TAGS_ud = len(tag_to_id_ud)

print('NO_TAGS_ud: ', NO_TAGS_ud)

VOCAB_LEN = len(word_to_id) + 1
TRAIN_EMBED = True
BATCH_SIZE = 128
PATIENCE = 4 
MAX_EPOCHS_TAG = 20
DP = 0.3
OPTIMIZER = keras.optimizers.Adam(lr =  0.0001)
ACTIVATION = 'relu'
SENT_REP_SIZE = 400
EFILE = "/data/s3094723/embeddings/en/glove.6B.100d.txt"
L2 = 4e-6

word_index = tokenizer.word_index
print(len(word_index))


#char data
char_to_id = c2i(train_X_sent_ud + dev_X_sent_ud + test_X_sent_ud, MAX_LEN, MAX_CHAR)

print(char_to_id)
#ud
sequences_train_X_ud_sent = tokenizer.texts_to_sequences(train_X_sent_ud)
max_seq_train_X_ud_sent = len(max(sequences_train_X_ud_sent, key=len))
chars_train_X_ud_sent,  max_word_train_X_ud_sent = proc_chars(train_X_sent_ud, 
                                                              MAX_LEN, MAX_CHAR, char_to_id)

sequences_dev_X_ud_sent = tokenizer.texts_to_sequences(dev_X_sent_ud)
max_seq_dev_X_ud_sent = len(max(sequences_dev_X_ud_sent, key=len))
chars_dev_X_ud_sent,  max_word_dev_X_ud_sent = proc_chars(dev_X_sent_ud,
                                                          MAX_LEN, MAX_CHAR, char_to_id)

sequences_test_X_ud_sent = tokenizer.texts_to_sequences(test_X_sent_ud)
max_seq_test_X_ud_sent = len(max(sequences_test_X_ud_sent, key=len))
chars_test_X_ud_sent,  max_word_test_X_ud_sent = proc_chars(test_X_sent_ud, 
                                                            MAX_LEN, MAX_CHAR, char_to_id)

print('Build model...')
print('Vocab size =', VOCAB_LEN)

#load embeddings
embedding_matrix, embedding_dim = read_embeds(EFILE, word_index)
print('Total number of null word embeddings:')
print(np.sum(np.sum(embedding_matrix, axis=1) == 0))
print(embedding_matrix.shape)

#format data
#char 
X_train_chars = chars_train_X_ud_sent
X_dev_chars = chars_dev_X_ud_sent 
X_test_chars = chars_test_X_ud_sent
#word
#X
X_train = train_X_ud
X_dev = dev_X_ud 
X_test = test_X_ud

#Y
Y_train_ud = train_Y_ud
Y_dev_ud = dev_Y_ud
Y_test_ud = test_Y_ud


#order: word_index, max_seq, embeds, char_index, MAX_CHAR, embed_dim, ACTIVATION, L2, DP, SENT_REP_SIZE, NO_TAGS_ud
#MODEL 
input, char_input, output_ud = make_model(word_index, MAX_LEN, embedding_matrix, char_to_id, MAX_CHAR, embedding_dim, 
                                           ACTIVATION, L2, DP, SENT_REP_SIZE, NO_TAGS_ud)

tagdmodel = Model(inputs = [input, char_input], outputs = output_ud)

tagdmodel.compile(optimizer=OPTIMIZER, loss='categorical_crossentropy', metrics=['accuracy'])
tagdmodel.summary()
tagdmodel.fit([X_train, X_train_chars], Y_train_ud,
         batch_size=BATCH_SIZE, epochs = MAX_EPOCHS_TAG,
         validation_data=([X_dev, X_dev_chars], Y_dev_ud))

acc  = tagdmodel.evaluate([X_test, X_test_chars], Y_test_ud, batch_size=BATCH_SIZE)

print(acc)



