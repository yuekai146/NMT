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
from utils import remove_bpe, remove_special_tok, subsequent_mask
from dataset import Dataset


def greedy(args, net, src, src_mask, src_vocab, tgt_vocab):
    # src: torch.LongTensor (bsz, slen)
    # src_mask: torch.ByteTensor (bsz, 1, slen)
    net.eval()
    max_len = args.max_len
    with torch.no_grad():
        bsz = src.size(0)
        enc_out = net.encode(src=src, src_mask=src_mask)
        generated = src.new(bsz, max_len)
        generated.fill_(tgt_vocab.stoi[config.PAD])
        generated[:, 0].fill_(tgt_vocab.stoi[config.BOS])
        generated = generated.long()
        
        cur_len = 1
        gen_len = src.new_ones(bsz).long()
        unfinished_sents = src.new_ones(bsz).long()

        cache = {'cur_len':cur_len - 1}

        while cur_len < max_len:
            x = generated[:, cur_len - 1].unsqueeze(-1)
            tgt_mask = ( generated[:, :cur_len] != tgt_vocab.stoi[config.PAD] ).unsqueeze(-2)
            tgt_mask = tgt_mask & Variable(
                    subsequent_mask(cur_len).type_as(tgt_mask.data)
                    )

            logit = net.decode(
                    enc_out, src_mask, x,
                    tgt_mask[:, cur_len-1, :].unsqueeze(-2), cache
                    )
            scores = net.generator(logit).exp().squeeze()
            
            next_words = torch.topk(scores, 1)[1].squeeze()

            assert next_words.size()  == (bsz,)
            generated[:, cur_len] = next_words * unfinished_sents + tgt_vocab.stoi[config.PAD] * (1 - unfinished_sents)
            gen_len.add_(unfinished_sents)
            unfinished_sents.mul_(next_words.ne(tgt_vocab.stoi[config.EOS]).long())
            cur_len = cur_len + 1
            cache['cur_len'] = cur_len - 1

            if unfinished_sents.max() == 0:
                break

        if cur_len == max_len:
            generated[:, -1].masked_fill_(unfinished_sents.bool(), tgt_vocab.stoi[config.EOS])

        return generated[:cur_len], gen_len


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

    return sentence


def gen_batch2str(src, generated, gen_len, src_vocab, tgt_vocab):
    generated = generated.cpu().numpy().tolist()
    gen_len = gen_len.cpu().numpy().tolist()
    src = src.cpu().numpy().tolist()
    translated = []
    for i, l in enumerate(generated):
        l = l[:gen_len[i]]
        sys_sent = " ".join([tgt_vocab.itos[tok] for tok in l])
        src_sent = " ".join([src_vocab.itos[tok] for tok in src[i]])
        sys_sent = remove_special_tok(remove_bpe(sys_sent))
        src_sent = remove_special_tok(remove_bpe(src_sent))
        translated.append("S: " + src_sent)
        translated.append("H: " + sys_sent)
    return translated


def translate(args, net, src_vocab, tgt_vocab):
    "done"
    sentences = [l.split() for l in args.text]
    translated = []

    if args.greedy:
        src_dataset = Dataset(args.text, src_vocab)
        if args.batch_size is not None:
            src_dataset.BATCH_SIZE = args.batch_size
        if args.max_batch_size is not None:
            src_dataset.max_batch_size = args.max_batch_size
        if args.tokens_per_batch is not None:
            src_dataset.tokens_per_batch = args.tokens_per_batch

        src_dataiter = iter(src_dataset.get_iterator(True, True))
        for src in src_dataiter:
            src_mask = (src != src_vocab.stoi[config.PAD]).unsqueeze(-2)
            if args.use_cuda:
                src, src_mask = src.cuda(), src_mask.cuda()
            generated, gen_len = greedy(args, net, src, src_mask, src_vocab, tgt_vocab)
            translated.extend(gen_batch2str(src, generated, gen_len, src_vocab, tgt_vocab))
            for res_sent in translated:
                print(res_sent)
    else:
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
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--max_batch_size', type=int, default=None)
    parser.add_argument('--tokens_per_batch', type=int, default=None)
    parser.add_argument('--greedy', action='store_true')

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

