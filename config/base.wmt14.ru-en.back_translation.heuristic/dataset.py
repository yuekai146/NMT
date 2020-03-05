# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from common import config
import argparse
import dill
import math
import numpy as np
import os
import pickle
import torch


class Vocab(object):

    def __init__(self, fpath):
        """
        fpath is the path of vocabulary file
        In vocabulary file, each word occupies a single line
        """
        self.itos = open(fpath, 'r').read().split('\n')[:-1]
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

    
    def batch_sentences(self, sentences, bos, eos, indices=None):
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
        
        if indices is None:
            return sent_tensor
        else:
            return sent_tensor, indices


    def get_batches_iterator(self, batches, bos, eos, include_indices=False):
        """
        Return a sentences iterator, given the associated sentence batches.
        """

        for sentence_ids in batches:
            if 0 < self.max_batch_size < len(sentence_ids):
                np.random.shuffle(sentence_ids)
                sentence_ids = sentence_ids[:self.max_batch_size]
            sent = [self.sent[idx] for idx in sentence_ids]
            if include_indices:
                sent = self.batch_sentences(sent, bos, eos, sentence_ids)
            else:
                sent = self.batch_sentences(sent, bos, eos)
            yield sent


    def get_iterator(self, shuffle, group_by_size=False, bos=False, eos=False, include_indices=False):
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
        return self.get_batches_iterator(batches, bos, eos, include_indices)


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


    def get_batches_iterator(self, batches, include_indices=False):
        """
        Return a sentences iterator, given the associated sentence batches.
        """

        for sentence_ids in batches:
            if 0 < self.max_batch_size < len(sentence_ids):
                np.random.shuffle(sentence_ids)
                sentence_ids = sentence_ids[:self.max_batch_size]
            if include_indices:
                sent1 = self.batch_sentences([self.sent1[idx] for idx in sentence_ids], bos=False, eos=False)
                sent2 = self.batch_sentences([self.sent2[idx] for idx in sentence_ids], bos=True, eos=True)
                yield Batch(sent1, sent2), sentence_ids
            else:
                sent1 = self.batch_sentences([self.sent1[idx] for idx in sentence_ids], bos=False, eos=False)
                sent2 = self.batch_sentences([self.sent2[idx] for idx in sentence_ids], bos=True, eos=True)
                yield Batch(sent1, sent2)


    def get_batch_ids(self, shuffle, group_by_size=False, num_subsets=None):
        assert type(shuffle) is bool and type(group_by_size) is bool
        assert group_by_size is False or shuffle is True
        assert num_subsets is None or isinstance(num_subsets, int)

        if num_subsets is not None:
            assert num_subsets > 1

        # sentence lengths
        #lengths = self.lengths1 + self.lengths2 + 2
        lengths = self.lengths1 + self.lengths2

        # select sentences to iterate over
        if shuffle:
            indices = np.random.permutation(len(self.sent1))
        else:
            indices = np.arange(len(self.sent1))

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

        if num_subsets is not None:
            subset_batch_num = len(batches) // num_subsets
            subset_batches = [batches[i*subset_batch_num:(i+1)*subset_batch_num] for i in range(num_subsets)]
            batches = subset_batches

        return batches


    def get_iterator(self, shuffle, group_by_size=False, include_indices=False):
        """
        Return a sentences iterator.
        """
        batches = self.get_batch_ids(shuffle=shuffle, group_by_size=group_by_size)

        # return the iterator
        return self.get_batches_iterator(batches, include_indices=include_indices)


def get(params=None):
    if params is not None:
        SRC_VOCAB_PATH = config.SRC_VOCAB_PATH if params.SRC_VOCAB_PATH is None else params.SRC_VOCAB_PATH
        TGT_VOCAB_PATH = config.TGT_VOCAB_PATH if params.TGT_VOCAB_PATH is None else params.TGT_VOCAB_PATH
        SRC_RAW_TRAIN_PATH = config.SRC_RAW_TRAIN_PATH if params.SRC_RAW_TRAIN_PATH is None else params.SRC_RAW_TRAIN_PATH
        TGT_RAW_TRAIN_PATH = config.TGT_RAW_TRAIN_PATH if params.TGT_RAW_TRAIN_PATH is None else params.TGT_RAW_TRAIN_PATH
        SRC_RAW_VALID_PATH = config.SRC_RAW_VALID_PATH if params.SRC_RAW_VALID_PATH is None else params.SRC_RAW_VALID_PATH
        TGT_RAW_VALID_PATH = config.TGT_RAW_VALID_PATH if params.TGT_RAW_VALID_PATH is None else params.TGT_RAW_VALID_PATH

    SRC = Text(Vocab(SRC_VOCAB_PATH), False, False)
    TGT = Text(Vocab(TGT_VOCAB_PATH), True, True)
    
    def read_sents(fpath):
        return open(fpath, 'r').read().split('\n')[:-1]

    src_sents = read_sents(SRC_RAW_TRAIN_PATH)
    tgt_sents = read_sents(TGT_RAW_TRAIN_PATH)
    print("Parallel sentences already read!")
    assert len(src_sents) == len(tgt_sents)

    train_iter = ParallelDataset(
            src_sents,
            tgt_sents,
            SRC.vocab,
            TGT.vocab
            )
    
    valid_iter = ParallelDataset(
            read_sents(SRC_RAW_VALID_PATH),
            read_sents(TGT_RAW_VALID_PATH),
            SRC.vocab,
            TGT.vocab
            )
    valid_iter.tokens_per_batch = 2000
    #valid_iter.batch_size = 32

    return train_iter, valid_iter, SRC, TGT


def load():
        # Load train dataset
        f = open(os.path.join(config.train_iter_dump_path), 'rb')
        train_iter = dill.load(f)
        f.close()
        
        # Load valid dataset
        f = open(os.path.join(config.valid_iter_dump_path), 'rb')
        valid_iter = dill.load(f)
        f.close()
        
        # Dump src vocab
        f = open(os.path.join(config.src_vocab_dump_path), 'rb')
        SRC_TEXT = dill.load(f)
        f.close()
        
        # Dump tgt vocab
        f = open(os.path.join(config.tgt_vocab_dump_path), 'rb')
        TGT_TEXT = dill.load(f)
        f.close()

        return train_iter, valid_iter, SRC_TEXT, TGT_TEXT


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--store', action="store_true", help="Store preprocessed dataset and vocab")
    parser.add_argument('--test', action='store_true', help="Test dataset code")
    parser.add_argument('--SRC_RAW_TRAIN_PATH', default=None, type=str, help="Path to store train source text")
    parser.add_argument('--TGT_RAW_TRAIN_PATH', default=None, type=str, help="Path to store train target text")
    parser.add_argument('--SRC_RAW_VALID_PATH', default=None, type=str, help="Path to store validation source text")
    parser.add_argument('--TGT_RAW_VALID_PATH', default=None, type=str, help="Path to store validation target text")
    parser.add_argument('--SRC_VOCAB_PATH', default=None, type=str, help="Path to store source vocab")
    parser.add_argument('--TGT_VOCAB_PATH', default=None, type=str, help="Path to store target vocab")
    parser.add_argument('--data_bin', default=None, type=str, help="Path to store binarized data")
    args = parser.parse_args()

    train_iter, valid_iter, SRC_TEXT, TGT_TEXT = get(args)
    
    if args.test:
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

    if args.store: 
        if args.data_bin is None:
            args.data_bin = config.data_bin
        if os.path.exists(args.data_bin) == False:
            os.makedirs(args.data_bin)
        
        if args.data_bin != config.data_bin:
            train_iter_dump_path = args.data_bin + 'train_iter'
            valid_iter_dump_path = args.data_bin + 'valid_iter'
            src_vocab_dump_path = args.data_bin + 'SRC'
            tgt_vocab_dump_path = args.data_bin + 'TGT'
        else:
            train_iter_dump_path = config.data_bin + 'train_iter'
            valid_iter_dump_path = config.data_bin + 'valid_iter'
            src_vocab_dump_path = config.data_bin + 'SRC'
            tgt_vocab_dump_path = config.data_bin + 'TGT'

        # Dump train dataset
        f = open(os.path.join(train_iter_dump_path), 'wb')
        pickle.dump(train_iter, f)
        f.close()
        
        # Dump valid dataset
        f = open(os.path.join(valid_iter_dump_path), 'wb')
        pickle.dump(valid_iter, f)
        f.close()
        
        # Dump src vocab
        f = open(os.path.join(src_vocab_dump_path), 'wb')
        pickle.dump(SRC_TEXT, f)
        f.close()
        
        # Dump tgt vocab
        f = open(os.path.join(tgt_vocab_dump_path), 'wb')
        pickle.dump(TGT_TEXT, f)
        f.close()
