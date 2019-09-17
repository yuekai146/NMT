from common import config
from torch.autograd import Variable
import numpy as np
import torch

import logging
import os
import time
from datetime import timedelta

import collections
import re


def average_checkpoints(inputs):
    """Loads checkpoints from inputs and returns a model with averaged weights.

    Args:
      inputs: An iterable of string paths of checkpoints to load from.

    Returns:
      A dict of string keys mapping to various values. The 'model' key
      from the returned dict should correspond to an OrderedDict mapping
      string parameter names to torch Tensors.
    """
    params_dict = collections.OrderedDict()
    params_keys = None
    new_state = None
    num_models = len(inputs)

    for f in inputs:
        state = torch.load(
            f,
            map_location=(
                lambda s, _: torch.serialization.default_restore_location(s, 'cpu')
            ),
        )
        # Copies over the settings from the first checkpoint
        if new_state is None:
            new_state = state

        model_params = state['net']

        model_params_keys = list(model_params.keys())
        if params_keys is None:
            params_keys = model_params_keys
        elif params_keys != model_params_keys:
            raise KeyError(
                'For checkpoint {}, expected list of params: {}, '
                'but found: {}'.format(f, params_keys, model_params_keys)
            )

        for k in params_keys:
            p = model_params[k]
            if isinstance(p, torch.HalfTensor):
                p = p.float()
            if k not in params_dict:
                params_dict[k] = p.clone()
                # NOTE: clone() is needed in case of p is a shared parameter
            else:
                params_dict[k] += p

    averaged_params = collections.OrderedDict()
    for k, v in params_dict.items():
        averaged_params[k] = v
        averaged_params[k].div_(num_models)
    new_state['model'] = averaged_params
    return new_state


def last_n_checkpoints(path, n, upper_bound=None):
    pt_regexp = re.compile(r'checkpoint_(\d+)\.pth')
    files = os.listdir(path)

    entries = []
    for f in files:
        m = pt_regexp.fullmatch(f)
        if m is not None:
            sort_key = int(m.group(1))
            if upper_bound is None or sort_key <= upper_bound:
                entries.append((sort_key, m.group(0)))
    if len(entries) < n:
        raise Exception('Found {} checkpoint files but need at least {}', len(entries), n)
    return [os.path.join(path, x[1]) for x in sorted(entries, reverse=True)[:n]]


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


def remove_bpe(s):
    return (s + ' ').replace("@@ ", "").rstrip()


def remove_special_tok(s):
    s = s.replace("<s>", "")
    s = s.replace("</s>", "")
    s = s.replace("<pad>", "")
    s = s.replace("<unk>", "")
    return s
