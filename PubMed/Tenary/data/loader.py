"""
Data loader for Nary json files.
"""

import json
import random
import torch
import numpy as np

from utils import constant


def read_file(filename, vocab, opt, is_train):
    print(filename)
    with open(filename) as infile:
        data = json.load(infile)

    label2id = constant.LABEL_TO_ID
    processed = []
    counter = 0
    for d_no, d in enumerate(data):
        tokens = list(d['token'])
        
        if is_train:
           if len(tokens) > 400:
              counter += 1
              continue

        if opt['lower']:
            tokens = [t.lower() for t in tokens]

        if "next" in d['stanford_deprel']:
            cross = True
        else:
            cross = False

        real_tokens = tokens
        tokens = map_to_ids(tokens, vocab.word2id)
        pos = map_to_ids(d['stanford_pos'], constant.POS_TO_ID)
        deprel = map_to_ids(d['stanford_deprel'], constant.DEPREL_TO_ID)
        head = [int(x) for x in d['stanford_head']]
        zero_counter = 0
        for x in head:
            if x == 0:
                zero_counter += 1
        if zero_counter != 1:
            print(d_no)
            print(real_tokens)
            print(head)
            exit()
        # assert any([x == 0 for x in head])
        l = len(tokens)

        relation = label2id[d['relation']]

        first_positions = get_positions(d['first_start'], d['first_end'], l)
        second_positions = get_positions(d['second_start'], d['second_end'], l)

        if 'third_start' in d:
            third_positions = get_positions(d['third_start'], d['third_end'], l)
        else:
            print(tokens)
            third_positions = list()

        processed += [(tokens, pos, deprel, head, first_positions, second_positions, third_positions, cross, relation)]
    
    print(str(counter) + " instances are removed")
    indices = list(range(len(processed)))
    random.shuffle(indices)
    processed = [processed[i] for i in indices]
    
    return processed


class DataLoader(object):
    """
    Load data from json files, preprocess and prepare batches.
    """
    def __init__(self, data, batch_size, opt, evaluation=False):
        self.batch_size = batch_size
        self.opt = opt
        self.eval = evaluation
        self.label2id = constant.LABEL_TO_ID
        self.id2label = dict([(v, k) for k, v in self.label2id.items()])

        self.labels = [self.id2label[d[-1]] for d in data]
        self.num_examples = len(data)

        # chunk into batches
        data = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
        self.data = data
        print("{} batches created".format(len(data)))

    def gold(self):
        """ Return gold labels as a list. """
        return self.labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        """ Get a batch with index. """
        if not isinstance(key, int):
            raise TypeError
        if key < 0 or key >= len(self.data):
            raise IndexError
        batch = self.data[key]
        batch_size = len(batch)
        batch = list(zip(*batch))

        # for nary dataset
        assert len(batch) == 9

        # sort all fields by lens for easy RNN operations
        lens = [len(x) for x in batch[0]]
        batch, orig_idx = sort_all(batch, lens)

        # word dropout
        if not self.eval:
            words = [word_dropout(sent, self.opt['word_dropout']) for sent in batch[0]]
        else:
            words = batch[0]

        # convert to tensors
        words = get_long_tensor(words, batch_size)
        masks = torch.eq(words, 0)
        pos = get_long_tensor(batch[1], batch_size)
        deprel = get_long_tensor(batch[2], batch_size)
        head = get_long_tensor(batch[3], batch_size)
        first_positions = get_long_tensor(batch[4], batch_size)
        second_positions = get_long_tensor(batch[5], batch_size)
        third_positions = get_long_tensor(batch[6], batch_size)
        cross = batch[7]
        rels = torch.LongTensor(batch[8])

        return (words, masks, pos, deprel, head, first_positions, second_positions, third_positions, cross, rels, orig_idx)


def map_to_ids(tokens, vocab):
    ids = [vocab[t] if t in vocab else constant.UNK_ID for t in tokens]
    return ids


def get_positions(start_idx, end_idx, length):
    """ Get subj/obj position sequence. """
    return list(range(-start_idx, 0)) + [0]*(end_idx - start_idx + 1) + \
            list(range(1, length-end_idx))


def sort_all(batch, lens):
    """ Sort all fields by descending order of lens, and return the original indices. """
    unsorted_all = [lens] + [range(len(lens))] + list(batch)
    sorted_all = [list(t) for t in zip(*sorted(zip(*unsorted_all), reverse=True))]
    return sorted_all[2:], sorted_all[1]


def word_dropout(tokens, dropout):
    """ Randomly dropout tokens (IDs) and replace them with <UNK> tokens. """
    return [constant.UNK_ID if x != constant.UNK_ID and np.random.random() < dropout \
            else x for x in tokens]


def get_long_tensor(tokens_list, batch_size):
    """ Convert list of list of tokens to a padded LongTensor. """
    token_len = max(len(x) for x in tokens_list)
    tokens = torch.LongTensor(batch_size, token_len).fill_(constant.PAD_ID)
    for i, s in enumerate(tokens_list):
        tokens[i, :len(s)] = torch.LongTensor(s)
    return tokens
