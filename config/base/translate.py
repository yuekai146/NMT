import argparse
import time
import torch
import torch.nn.functional as F
import argparse
from torch.autograd import Variable
import os
import re

import model
from beam import beam_search
from common import config
from utils import remove_bpe, remove_special_tok


def load_model(checkpoint_path, net):
    """
    Reload a checkpoint if we find one.
    """
    assert os.path.isfile(checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location='cpu')

    # reload model parameters
    s_dict = {}
    for k in ckpt["net"]:
        new_k = k[7:]
        s_dict[new_k] = ckpt["net"][k]

    net.load_state_dict(s_dict)

    src_vocab = ckpt["src_vocab"]
    tgt_vocab = ckpt["tgt_vocab"]
    return net, src_vocab, tgt_vocab


def multiple_replace(dict, text):
  # Create a regular expression  from the dictionary keys
  regex = re.compile("(%s)" % "|".join(map(re.escape, dict.keys())))

  # For each match, look-up corresponding value in dictionary
  return regex.sub(lambda mo: dict[mo.string[mo.start():mo.end()]], text) 


def translate_sentence(sentence, net, args, src_vocab, tgt_vocab):
    
    net.eval()
    indexed = []
    for tok in sentence:
        if tok not in ["<pad>", "<s>", "</s>"]:
            try:
                tok_i = src_vocab.stoi[tok]
            except KeyError:
                tok_i = src_vocab.stoi["<unk>"]
            indexed.append(tok_i)
    sentence = Variable(torch.LongTensor([indexed]))
    if args.use_cuda:
        sentence = sentence.cuda()
    
    sentence = beam_search(sentence, net, src_vocab, tgt_vocab, args)

    #return  multiple_replace({' ?' : '?',' !':'!',' .':'.','\' ':'\'',' ,':','}, sentence)
    return sentence


def translate(args, net, src_vocab, tgt_vocab):
    "done"
    sentences = [l.split() for l in args.text]
    translated = []

    for i_s, sentence in enumerate(sentences):
        s_trans = translate_sentence(sentence, net, args, src_vocab, tgt_vocab)
        s_trans = remove_special_tok(remove_bpe(s_trans))
        translated.append(s_trans)
        print(translated[-1])

    return translated


def main():
    "done"
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-ckpt', required=True)
    parser.add_argument('-k', type=int, default=5)
    parser.add_argument('-max_len', type=int, default=250)
    parser.add_argument('-max_ratio', type=int, default=1.5)
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-text', type=str, required=True)
    parser.add_argument('-lp', '--length_penalty', type=float, default=0.7)
    
    args = parser.parse_args()
    args.use_cuda = ( args.no_cuda == False) and torch.cuda.is_available()

    assert args.k > 0
    assert args.max_len > 10

    net, _ = model.get()
    net, src_vocab, tgt_vocab = load_model(args.ckpt, net)

    if args.use_cuda:
        net = net.cuda()
    
    fpath = args.text
    try:
        args.text = open(fpath, encoding='utf-8').read().split('\n')[:-1]
    except:
        print("error opening or reading text file")
    
    translate(args, net, src_vocab, tgt_vocab)

if __name__ == '__main__':
    main()

