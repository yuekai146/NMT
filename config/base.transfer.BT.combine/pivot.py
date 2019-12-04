import argparse
import time
import torch
import torch.nn.functional as F
import argparse
from torch.autograd import Variable
import os
import re

import model
from common import config
from utils import remove_bpe, remove_special_tok, subsequent_mask
from dataset import ParallelDataset


def make_model(args):
    """
    Reload a checkpoint if we find one.
    """
    assert os.path.isfile(args.encoder_model)
    assert os.path.isfile(args.decoder_model)
    ckpt_enc = torch.load(args.encoder_model, map_location='cpu')
    ckpt_dec = torch.load(args.decoder_model, map_location='cpu')

    # reload model parameters
    s_dict = {}
    for k in ckpt_enc["net"]:
        assert k in ckpt_dec["net"]
        new_k = k[7:]

        if new_k.split('.')[0] in ['encoder', 'src_emb']:
            s_dict[k] = ckpt_enc["net"][k]
        elif new_k.split('.')[0] in ['decoder', 'tgt_emb', 'generator']:
            s_dict[k] = ckpt_dec["net"][k]

        new_ckpt = {}
        new_ckpt["net"] = s_dict
        new_ckpt["src_vocab"] = ckpt_enc["src_vocab"]
        new_ckpt["tgt_vocab"] = ckpt_dec["tgt_vocab"]

    return new_ckpt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-em', '--encoder_model', type=str,
            help="Path to load encoder parameters"
            )
    parser.add_argument('-dm', '--decoder_model', type=str,
            help="Path to load decoder parameters"
            )
    parser.add_argument('--store', type=str,
            help="Where to store new model"
            )
    args = parser.parse_args()

    net = make_model(args)
    torch.save(net, args.store)


if __name__ == "__main__":
    main()
