from utils import remove_bpe, remove_special_tok
import argparse
import time
import torch
import torch.nn.functional as F
import argparse
from torch.autograd import Variable
import numpy as np
import os
import re

import model
from common import config
from utils import remove_bpe, remove_special_tok, subsequent_mask
from dataset import ParallelDataset


def greedy(args, net, src, src_mask, src_vocab, tgt_vocab):
    # src: torch.LongTensor (bsz, slen)
    # src_mask: torch.ByteTensor (bsz, 1, slen)
    slen = src.size(1)
    max_len = min(args.max_len, int(slen * args.gen_a + args.gen_b))
    with torch.no_grad():
        net.eval()
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
            
            next_words = torch.topk(scores, 1)[1].view(bsz)

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

        return generated[:, :cur_len], gen_len


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


def gen_batch2str(src, tgt, generated, gen_len, src_vocab, tgt_vocab, indices=None, active_out=None):
    generated = generated.cpu().numpy().tolist()
    gen_len = gen_len.cpu().numpy().tolist()
    src = src.cpu().numpy().tolist()
    tgt = tgt.cpu().numpy().tolist()
    translated = []
    for idx, l in enumerate(generated):
        l = l[:gen_len[idx]]
        sys_sent = " ".join([tgt_vocab.itos[tok] for tok in l])
        src_sent = " ".join([src_vocab.itos[tok] for tok in src[idx]])
        ref_sent = " ".join([tgt_vocab.itos[tok] for tok in tgt[idx]])
        sys_sent = remove_special_tok(sys_sent)
        src_sent = remove_special_tok(src_sent)
        ref_sent = remove_special_tok(ref_sent)
        translated.append("S: " + src_sent)
        translated.append("H: " + sys_sent)
        translated.append("T: " + ref_sent)
        if indices is not None and active_out is not None:
            translated.append("V: " + str(active_out[indices[idx]][3]))
            translated.append(active_out[indices[idx]][4])
    return translated


def translate(args, net, src_vocab, tgt_vocab, active_out=None):
    "done"
    sentences = [l.split() for l in args.text]
    translated = []

    infer_dataset = ParallelDataset(args.text, args.ref_text, src_vocab, tgt_vocab)
    if args.batch_size is not None:
        infer_dataset.BATCH_SIZE = args.batch_size
    if args.max_batch_size is not None:
        infer_dataset.max_batch_size = args.max_batch_size
    if args.tokens_per_batch is not None:
        infer_dataset.tokens_per_batch = args.tokens_per_batch

    infer_dataiter = iter(infer_dataset.get_iterator(
        shuffle=True, group_by_size=True, include_indices=True)
        )

    for (raw_batch, indices) in infer_dataiter:
        src_mask = (raw_batch.src != src_vocab.stoi[config.PAD]).unsqueeze(-2)
        if args.use_cuda:
            src, src_mask = raw_batch.src.cuda(), src_mask.cuda()
        else:
            src = raw_batch.src
        generated, gen_len = greedy(args, net, src, src_mask, src_vocab, tgt_vocab)
        new_translations = gen_batch2str(src, raw_batch.tgt, generated, gen_len, src_vocab, tgt_vocab, indices, active_out)
        translated.extend(new_translations)

    return translated


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help='two modes, get or translate')
    
    parser_get = subparsers.add_parser(
            'get', help='Get texts that needs to be labeled or translated'
            )
    parser_get.add_argument('-AO', '--active_out', type=str, default=None,
            help="Output file generated by active.py score mode"
            )
    parser_get.add_argument('-tb', '--tok_budget', type=int,
            help="Token budget"
            )
    parser_get.add_argument('-bttb', '--back_translation_tok_budget', type=int,
            help="Back translation token budget"
            )
    parser_get.add_argument('--sort', action="store_true",
            help="Whether to sort active out by value"
            )
    parser_get.add_argument('-o', '--output', type=str,
            help="Output filepath"
            )
    parser_get.add_argument('-on', '--output_num', type=int, default=1,
            help="Output filepath"
            )

    parser_trans = subparsers.add_parser(
            'translate', help='Translate sentences'
            )
    parser_trans.add_argument('-i', '--input', type=str,
            help='Input file'
            )
    parser_trans.add_argument('-o', '--output', type=str,
            help="Output file"
            )
    parser_trans.add_argument('--ckpt', required=True)
    parser_trans.add_argument('--max_len', type=int, default=250)
    parser_trans.add_argument('--gen_a', type=float, default=1.3)
    parser_trans.add_argument('--gen_b', type=int, default=5)
    parser_trans.add_argument('--no_cuda', action='store_true')
    parser_trans.add_argument('--batch_size', type=int, default=None)
    parser_trans.add_argument('--max_batch_size', type=int, default=None)
    parser_trans.add_argument('--tokens_per_batch', type=int, default=None)


    args = parser.parse_args()
    args.mode = "get" if hasattr(args, 'active_out') else "translate"
    if args.mode == 'translate':
        args.use_cuda = ( args.no_cuda == False ) and torch.cuda.is_available()

    if args.mode == "get":
        f = open(args.active_out, 'r')
        lines = f.read().split('\n')[:-1]
        f.close()

        assert len(lines) % 4 == 0
        active_out = [(lines[idx], lines[idx+1], float(lines[idx+2].split(' ')[-1]), lines[idx+3]) for idx in range(0, len(lines), 4)]
        if args.sort:
            active_out = sorted(active_out, key=lambda item:item[2])
        
        indices = np.arange(len(active_out))
        lengths = np.array([len(remove_special_tok(remove_bpe(item[0][len("S: "):])).split(' ')) for item in active_out])
        include_oracle = np.cumsum(lengths) <= args.tok_budget
        include_pseudo = np.cumsum(lengths) <= ( args.tok_budget + args.back_translation_tok_budget )
        include_pseudo = np.logical_xor(include_pseudo, include_oracle)
        include_pseudo = indices[include_pseudo]
        include_oracle = indices[include_oracle]
        others = [idx for idx in indices if (idx not in include_pseudo) and (idx not in include_oracle)]
        
        # Output oracle and others
        output_oracle = args.output + '_oracle'
        f = open(output_oracle, 'w')
        out = []
        for idx in include_oracle:
            item = []
            item.append(active_out[idx][0])
            item.append('H: ' + active_out[idx][1][len('T: '):])
            item.append('T: ' + active_out[idx][1][len('T: '):])
            item.append('V: ' + str(active_out[idx][2]))
            item.append(active_out[idx][3])
            out.extend(item)

        f.write('\n'.join(out) + '\n')
        f.close()
        
        output_others = args.output + '_others'
        f = open(output_others, 'w')
        out = []
        for idx in others:
            item = []
            item.append(active_out[idx][0])
            item.append('H: ' + active_out[idx][1][len('T: '):])
            item.append('T: ' + active_out[idx][1][len('T: '):])
            item.append('V: ' + str(active_out[idx][2]))
            item.append(active_out[idx][3])
            out.extend(item)

        f.write('\n'.join(out) + '\n')
        f.close()

        # Output pseudo
        if args.output_num > 1:
            n_lines = len(include_pseudo) // args.output_num + 1
            for n in range(args.output_num):
                output_pseudo = args.output + '_pseudo_' + str(n)
                f = open(output_pseudo, 'w')
                out = []

                for idx in include_pseudo[n*n_lines:(n+1)*n_lines]:
                    item = []
                    item.append(active_out[idx][0])
                    item.append('H: ' + active_out[idx][1][len('T: '):])
                    item.append('T: ' + active_out[idx][1][len('T: '):])
                    item.append('V: ' + str(active_out[idx][2]))
                    item.append(active_out[idx][3])
                    out.extend(item)

                f.write('\n'.join(out) + '\n')
                f.close()
        else:
            assert args.output_num == 1
            output_pseudo = args.output + '_pseudo'
            f = open(output_pseudo, 'w')
            out = []

            for idx in include_pseudo:
                item = []
                item.append(active_out[idx][0])
                item.append('H: ' + active_out[idx][1][len('T: '):])
                item.append('T: ' + active_out[idx][1][len('T: '):])
                item.append('V: ' + str(active_out[idx][2]))
                item.append(active_out[idx][3])
                out.extend(item)

            f.write('\n'.join(out) + '\n')
            f.close()
    elif args.mode == 'translate':

        assert args.max_len > 10

        net, _ = model.get()
        net, src_vocab, tgt_vocab = load_model(args.ckpt, net)

        if args.use_cuda:
            net = net.cuda()
        
        fpath = args.input
        try:
            lines = open(fpath, 'r').read().split('\n')[:-1]
            active_out = [(lines[idx], lines[idx+1], lines[idx+2], float(lines[idx+3].split(' ')[-1]), lines[idx+4]) for idx in range(0, len(lines), 5)]
            args.text = [a[0][len('S: '):].strip() for a in active_out]
            args.ref_text = [a[2][len('T: '):].strip() for a in active_out]
        except:
            print("error opening or reading text file")
        
        out = translate(args, net, src_vocab, tgt_vocab, active_out)

        f = open(args.output, 'w')
        f.write('\n'.join(out) + '\n')
        f.close()


if __name__ == "__main__":
    main()
