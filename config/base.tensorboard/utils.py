from common import config
from torch.autograd import Variable
import numpy as np
import torch

import logging
import os
import time
from datetime import timedelta


class LogFormatter():

    def __init__(self):
        self.start_time = time.time()

    def format(self, record):
        elapsed_seconds = round(record.created - self.start_time)

        prefix = "%s - %s - %s" % (
            record.levelname,
            time.strftime('%x %X'),
            timedelta(seconds=elapsed_seconds)
        )
        message = record.getMessage()
        message = message.replace('\n', '\n' + ' ' * (len(prefix) + 3))
        return "%s - %s" % (prefix, message) if message else ''


def create_logger(filepath, rank):
    """
    Create a logger.
    Use a different log file for each process.
    """
    # create log formatter
    log_formatter = LogFormatter()

    # create file handler and set level to debug
    if filepath is not None:
        if rank > 0:
            filepath = '%s-%i' % (filepath, rank)
        file_handler = logging.FileHandler(filepath, "a")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(log_formatter)

    # create console handler and set level to info
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_formatter)

    # create logger and set level to debug
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    if filepath is not None:
        logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # reset logger elapsed time
    def reset_time():
        log_formatter.start_time = time.time()
    logger.reset_time = reset_time

    return logger


def subsequent_mask(size):
    # Create a mask for decoder self attention
    # Lower triangular matrix with all non zero elements equals to 1
    # The diagonal are all ones
    attn_shape = (1, size, size)
    mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(mask) == 0


def to_cuda(batch):
    for k in batch.keys():
        try:
            batch[k] = batch[k].cuda()
        except AttributeError:
            continue
    return batch


def get_batch(src, tgt, src_vocab, tgt_vocab):
    # src: torch.LongTensor, (bsz, src_len)
    # tgt: torch.LongTensor, (bsz, tgt_len)
    # Note that target_mask is for caculating loss
    # tgt_mask is network input.
    src_mask = (src != src_vocab.stoi[config.PAD]).unsqueeze(-2)
    bsz = tgt.size(0)
    target = tgt[:, 1:]
    non_eos_mask = (tgt != tgt_vocab.stoi[config.EOS])
    tgt_in = torch.masked_select(tgt, non_eos_mask).view(bsz, -1)

    # Create target mask
    tgt_mask = ( tgt_in != tgt_vocab.stoi[config.PAD] ).unsqueeze(-2)
    tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt_in.size(-1)).type_as(tgt_mask.data)
            )
    target_mask = ( target != tgt_vocab.stoi[config.PAD] )
    n_tokens = torch.sum(target_mask).item()

    batch = {"src":src, "tgt":tgt_in,
            "src_mask":src_mask, "tgt_mask":tgt_mask,
            "target":target, "target_mask":target_mask,
            "n_tokens":n_tokens} 
    if config.use_cuda:
        batch = to_cuda(batch)

    return batch
