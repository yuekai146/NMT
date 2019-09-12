import torch
import torch.nn.functional as F
import math

from common import config
from utils import subsequent_mask


def init_vars(src, net, src_vocab, tgt_vocab, args):
    # src: torch.LongTensor, (1, src_len)
    assert src.dim() == 2
    src_len = src.size(-1)
    max_len = int(min(args.max_len, args.max_ratio * src_len))
    
    init_tok = tgt_vocab.stoi[config.BOS]
    src_mask = (src != src_vocab.stoi[config.PAD]).unsqueeze(-2)
    e_output = net.encode(src, src_mask)
    
    outputs = torch.LongTensor([[init_tok]])
    if args.use_cuda:
        outputs = outputs.cuda()
    
    tgt_mask = subsequent_mask(1)
    if args.use_cuda:
        tgt_mask = tgt_mask.cuda()
    
    out = net.generator.proj( net.decode(e_output, src_mask, outputs, tgt_mask) )
    out = F.softmax(out, dim=-1)
    
    probs, ix = out[:, -1].data.topk(args.k)
    log_scores = torch.Tensor([math.log(prob) for prob in probs.data[0]]).unsqueeze(0)
    
    outputs = torch.zeros(args.k, max_len).long()
    if args.use_cuda:
        outputs = outputs.cuda()
    outputs[:, 0] = init_tok
    outputs[:, 1] = ix[0]
    
    e_outputs = torch.zeros(args.k, e_output.size(-2), e_output.size(-1))
    if args.use_cuda:
        e_outputs = e_outputs.cuda()
    e_outputs[:, :] = e_output[0]
    
    return outputs, e_outputs, log_scores


def k_best_outputs(outputs, out, log_scores, i, k):
    # outputs: torch.LongTensor (k, i), previously decoded sentences
    # out: torch.FloatTensor (k, i+1, n_vocab), softmax prob
    # log_scores: torch.FloatTensor (1, k), sentence score per beam
    # i, k: int
    
    probs, ix = out[:, -1].data.topk(k)
    log_probs = torch.Tensor([math.log(p) for p in probs.data.view(-1)]).view(k, -1) + log_scores.transpose(0, 1)
    k_probs, k_ix = log_probs.view(-1).topk(k)
    
    row = k_ix // k
    col = k_ix % k

    outputs[:, :i] = outputs[row, :i]
    outputs[:, i] = ix[row, col]

    log_scores = k_probs.unsqueeze(0)
    
    return outputs, log_scores


def beam_search(src, net, src_vocab, tgt_vocab, args):    

    """
    This implementation of beam search is problematic.
    log_scores should not include scores of special tokens like <pad>, <s> etc.
    """
    outputs, e_outputs, log_scores = init_vars(src, net, src_vocab, tgt_vocab, args)
    eos_tok = tgt_vocab.stoi[config.EOS]
    src_mask = (src != src_vocab.stoi[config.PAD]).unsqueeze(-2)
    ind = None

    src_len = src.size(-1)
    max_len = int(min(args.max_len, args.max_ratio * src_len))
    for i in range(1, max_len):
    
        tgt_mask = subsequent_mask(i)
        if args.use_cuda:
            tgt_mask = tgt_mask.cuda()

        out = net.generator.proj( net.decode(e_outputs, src_mask, outputs[:, :i], tgt_mask) )

        out = F.softmax(out, dim=-1)
    
        outputs, log_scores = k_best_outputs(outputs, out, log_scores, i, args.k)
        
        ones = (outputs == eos_tok).nonzero() # Occurrences of end symbols for all input sentences.
        sentence_lengths = torch.zeros(len(outputs), dtype=torch.long).cuda()
        for vec in ones:
            i = vec[0]
            if sentence_lengths[i] == 0: # First end symbol has not been found yet
                sentence_lengths[i] = vec[1] # Position of first end symbol

        num_finished_sentences = len([s for s in sentence_lengths if s > 0])

        if num_finished_sentences == args.k:
            alpha = args.length_penalty
            div = 1 / ( sentence_lengths.type_as(log_scores) ** alpha )
            _, ind = torch.max(log_scores * div, 1)
            ind = ind.data[0]
            break
    
    if ind is None:
        try:
            length = ( outputs[0] == eos_tok ).nonzero()[0]
            return ' '.join([tgt_vocab.itos[tok] for tok in outputs[0][1:length]])
        except IndexError:
            return ' '.join([tgt_vocab.itos[tok] for tok in outputs[0]])
    else:
        length = ( outputs[ind] == eos_tok ).nonzero()[0]
        return ' '.join([tgt_vocab.itos[tok] for tok in outputs[ind][1:length]])
