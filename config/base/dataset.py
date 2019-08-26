# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from common import config
import math
import numpy as np
import torch


class Vocab(object):

    def __init__(self, fpath):
        """
        fpath is the path of vocabulary file
        In vocabulary file, each word occupies a single line
        """
        self.itos = open(fpath, 'r').read().split('\n')
        if len(self.itos[0].split(' ')) == 2:
            self.itos = [tok.split(' ')[0] for tok in self.itos]
        self.itos = config.SPECIAL_TOKENS + self.itos
        self.stoi = {}
        for i, w in enumerate(self.itos):
            self.stoi[w] = i


class Text(object):

    def __init__(self, vocab, bos, eos):
        self.vocab = vocab
        self.bos = bos
        self.eos = eos


class Batch(object):

    def __init__(self, src, tgt):
        self.src = src
        self.tgt = tgt

class Dataset(object):

    def __init__(self, sent, vocab):
        # sent: List of strings, each string is a tokenized sentence
        # vocab: A Vocab instance
        self.bos_index = vocab.stoi[config.BOS]
        self.eos_index = vocab.stoi[config.EOS]
        self.pad_index = vocab.stoi[config.PAD]
        self.vocab = vocab
        self.batch_size = config.BATCH_SIZE
        self.tokens_per_batch = config.tokens_per_batch
        self.max_batch_size = config.max_batch_size

        self.sent = sent

        # Turn self.sent into intergers
        sent_ = []
        for s in self.sent:
            sent_.append([vocab.stoi[tok] if tok in vocab.stoi else vocab.stoi[config.UNK] for tok in s.split()])
        self.sent = sent_
        self.lengths = np.array([len(s) for s in self.sent])


    def __len__(self):
        """
        Number of sentences in the dataset.
        """
        return len(self.sent)

    
    def batch_sentences(self, sentences, bos, eos):
        """
        Take as input a list of n sentences (List of intergers) and return
        a tensor of size (n, slen) where slen is the length of the longest
        sentence, and a vector lengths containing the length of each sentence.
        """
        batch_size = len(sentences)
        slen = max([len(s) for s in sentences])
        if bos:
            slen += 1
        if eos:
            slen += 1
        
        def pad_sent(s, max_len, bos, eos):
            ret = s
            if bos:
                ret = [self.bos_index] + ret
            if eos:
                ret = ret + [self.eos_index]
            ret = ret + [self.pad_index for _ in range(max_len - len(ret))]
            return ret

        sent_tensor = [pad_sent(s, slen, bos, eos) for s in sentences]
        sent_tensor = torch.from_numpy(np.array(sent_tensor)).long()

        return sent_tensor


    def get_batches_iterator(self, batches, bos, eos):
        """
        Return a sentences iterator, given the associated sentence batches.
        """

        for sentence_ids in batches:
            if 0 < self.max_batch_size < len(sentence_ids):
                np.random.shuffle(sentence_ids)
                sentence_ids = sentence_ids[:self.max_batch_size]
            sent = [self.sent[idx] for idx in sentence_ids]
            sent = self.batch_sentences(sent, bos, eos)
            yield sent


    def get_iterator(self, shuffle, group_by_size=False, bos=False, eos=False):
        """
        Return a sentences iterator.
        """
        assert type(shuffle) is bool and type(group_by_size) is bool
        assert group_by_size is False or shuffle is True


        # select sentences to iterate over
        if shuffle:
            indices = np.random.permutation(len(self.sent))
        else:
            indices = np.arange(len(sent))

        # group sentences by lengths
        if group_by_size:
            indices = indices[np.argsort(self.lengths[indices], kind='mergesort')]

        # create batches - either have a fixed number of sentences, or a similar number of tokens
        if self.tokens_per_batch == -1:
            batches = np.array_split(indices, math.ceil(len(indices) * 1. / self.batch_size))
        else:
            batch_ids = np.cumsum(self.lengths[indices]) // self.tokens_per_batch
            _, bounds = np.unique(batch_ids, return_index=True)
            batches = [indices[bounds[i]:bounds[i + 1]] for i in range(len(bounds) - 1)]
            if bounds[-1] < len(indices):
                batches.append(indices[bounds[-1]:])

        # optionally shuffle batches
        if shuffle:
            np.random.shuffle(batches)

        # sanity checks
        assert self.lengths[indices].sum() == sum([self.lengths[x].sum() for x in batches])

        # return the iterator
        return self.get_batches_iterator(batches, bos, eos)


class ParallelDataset(Dataset):

    def __init__(self, sent1, sent2, vocab1, vocab2):

        for sp_tok in config.SPECIAL_TOKENS:
            assert vocab1.stoi[sp_tok] == vocab2.stoi[sp_tok]

        self.bos_index = vocab1.stoi[config.BOS]
        self.eos_index = vocab1.stoi[config.EOS]
        self.pad_index = vocab1.stoi[config.PAD]

        self.batch_size = config.BATCH_SIZE
        self.tokens_per_batch = config.tokens_per_batch
        self.max_batch_size = config.max_batch_size

        self.sent1 = sent1
        self.sent2 = sent2

        self.vocab1 = vocab1
        self.vocab2 = vocab2
        
        sent_ = []
        for s in self.sent1:
            sent_.append([vocab1.stoi[tok] if tok in vocab1.stoi else vocab1.stoi[config.UNK] for tok in s.split()])
        self.sent1 = sent_
        
        sent_ = []
        for s in self.sent2:
            sent_.append([vocab2.stoi[tok] if tok in vocab2.stoi else vocab2.stoi[config.UNK] for tok in s.split()])
        self.sent2 = sent_

        assert len(self.sent1) == len(self.sent2)

        self.lengths1 = np.array([len(s) for s in self.sent1])
        self.lengths2 = np.array([len(s) for s in self.sent2])
        

    def __len__(self):
        """
        Number of sentences in the dataset.
        """
        assert len(self.sent1) == len(self.sent2)
        return len(self.sent1)

    def get_batches_iterator(self, batches):
        """
        Return a sentences iterator, given the associated sentence batches.
        """

        for sentence_ids in batches:
            if 0 < self.max_batch_size < len(sentence_ids):
                np.random.shuffle(sentence_ids)
                sentence_ids = sentence_ids[:self.max_batch_size]
            sent1 = self.batch_sentences([self.sent1[idx] for idx in sentence_ids], bos=False, eos=False)
            sent2 = self.batch_sentences([self.sent2[idx] for idx in sentence_ids], bos=True, eos=True)
            yield Batch(sent1, sent2)

    def get_iterator(self, shuffle, group_by_size=False):
        """
        Return a sentences iterator.
        """
        assert type(shuffle) is bool and type(group_by_size) is bool
        assert group_by_size is False or shuffle is True

        # sentence lengths
        #lengths = self.lengths1 + self.lengths2 + 2
        lengths = self.lengths1

        # select sentences to iterate over
        if shuffle:
            indices = np.random.permutation(len(self.sent1))
        else:
            indices = np.arange(len(sent1))

        # group sentences by lengths
        if group_by_size:
            indices = indices[np.argsort(lengths[indices], kind='mergesort')]

        # create batches - either have a fixed number of sentences, or a similar number of tokens
        if self.tokens_per_batch == -1:
            batches = np.array_split(indices, math.ceil(len(indices) * 1. / self.batch_size))
        else:
            batch_ids = np.cumsum(lengths[indices]) // self.tokens_per_batch
            _, bounds = np.unique(batch_ids, return_index=True)
            batches = [indices[bounds[i]:bounds[i + 1]] for i in range(len(bounds) - 1)]
            if bounds[-1] < len(indices):
                batches.append(indices[bounds[-1]:])

        # optionally shuffle batches
        if shuffle:
            np.random.shuffle(batches)

        # return the iterator
        return self.get_batches_iterator(batches)


def get():
    SRC = Text(Vocab(config.SRC_VOCAB_PATH), False, False)
    TGT = Text(Vocab(config.TGT_VOCAB_PATH), True, True)
    
    def read_sents(fpath):
        return open(fpath, 'r').read().split('\n')

    train_iter = ParallelDataset(
            read_sents(config.SRC_RAW_TRAIN_PATH),
            read_sents(config.TGT_RAW_TRAIN_PATH),
            SRC.vocab,
            TGT.vocab
            )

    valid_iter = ParallelDataset(
            read_sents(config.SRC_RAW_VALID_PATH),
            read_sents(config.TGT_RAW_VALID_PATH),
            SRC.vocab,
            TGT.vocab
            )
    valid_iter.tokens_per_batch = 2000

    return train_iter, valid_iter, SRC, TGT


if __name__ == "__main__":
    train_iter, valid_iter, SRC_TEXT, TGT_TEXT = get()

    from utils import get_batch
    def test_iterator(to_test):
        def get_sents(tokens, vocab):
            tokens = tokens.cpu().numpy().tolist()
            sents = []
            for s in tokens:
                sents.append(" ".join(vocab.itos[x] for x in s))
            return sents
        for i_batch, raw_batch in enumerate(iter(to_test)):
            batch = get_batch(raw_batch.src, raw_batch.tgt, SRC_TEXT.vocab, TGT_TEXT.vocab)

            src_sents = get_sents(batch["src"], SRC_TEXT.vocab)
            tgt_sents = get_sents(batch["target"], TGT_TEXT.vocab)
            tgt_in_sents = get_sents(batch["tgt"], TGT_TEXT.vocab)
            assert len(src_sents) == len(tgt_sents)
            assert len(tgt_sents) == len(tgt_in_sents)
            for idx in range(len(src_sents)):
                print(src_sents[idx].replace("<pad>", ""))
                print(tgt_in_sents[idx].replace("<pad>", ""))
                print(tgt_sents[idx].replace("<pad>", ""))
                print("\n")

            print("Batch size:{}\n".format(batch["src"].size()))

            if i_batch >= 0:
                break
    test_iterator(train_iter.get_iterator(True, True))
    #test_iterator(valid_iter.get_iterator(True, True))
