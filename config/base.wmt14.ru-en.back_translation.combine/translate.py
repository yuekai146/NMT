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

        return generated[:, :cur_len], gen_len


class BeamHypotheses(object):

    def __init__(self, n_hyp, max_len, length_penalty, early_stopping):
        """
        Initialize n-best list of hypotheses.
        """
        self.max_len = max_len - 1  # ignoring <BOS>
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.n_hyp = n_hyp
        self.hyp = []
        self.worst_score = 1e9

    
    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.hyp)

    
    def add(self, hyp, sum_logprobs):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / len(hyp) ** self.length_penalty
        if len(self) < self.n_hyp or score > self.worst_score:
            self.hyp.append((score, hyp))
            if len(self) > self.n_hyp:
                sorted_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.hyp)])
                del self.hyp[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    
    def is_done(self, best_sum_logprobs):
        """
        If there are enough hypotheses and that none of the hypotheses being generated
        can become better than the worst one in the heap, then we are done with this sentence.
        """
        if len(self) < self.n_hyp:
            return False
        elif self.early_stopping:
            return True
        else:
            return self.worst_score >= best_sum_logprobs / self.max_len ** self.length_penalty


def generate_beam(args, net, src, src_mask, src_vocab, tgt_vocab):
    """
    Decode a sentence given initial start.
    `x`:
        - LongTensor(bs, slen)
            <EOS> W1 W2 W3 <EOS> <PAD>
            <EOS> W1 W2 W3   W4  <EOS>
    `lengths`:
        - LongTensor(bs) [5, 6]
    """

    max_len = args.max_len
    beam_size = args.beam_size
    length_penalty = args.length_penalty
    early_stopping = args.early_stopping

    with torch.no_grad():
        net.eval()
        # batch size / number of words
        n_words = net.generator.n_vocab
        bs = src.size(0)
        
        # calculate encoder output
        src_enc = net.encode(src=src, src_mask=src_mask)
        src_len = src_mask.view(bs, -1).sum(dim=-1).long()
 
        # check inputs
        assert src_enc.size(0) == src_len.size(0)
        assert beam_size >= 1


        # expand to beam size the source latent representations / source lengths
        src_enc = src_enc.unsqueeze(1).expand((bs, beam_size) + src_enc.shape[1:]).contiguous().view((bs * beam_size,) + src_enc.shape[1:])
        src_len = src_len.unsqueeze(1).expand(bs, beam_size).contiguous().view(-1)
        src_mask = src_mask.unsqueeze(1).expand((bs, beam_size) + src_mask.shape[1:]).contiguous().view((bs * beam_size,) + src_mask.shape[1:])

        # generated sentences (batch with beam current hypotheses)
        generated = src_len.new(max_len, bs * beam_size)  # upcoming output
        generated.fill_(tgt_vocab.stoi[config.PAD])                   # fill upcoming ouput with <PAD>
        generated[0].fill_(tgt_vocab.stoi[config.BOS])                # we use <EOS> for <BOS> everywhere

        # generated hypotheses
        generated_hyps = [BeamHypotheses(beam_size, max_len, length_penalty, early_stopping) for _ in range(bs)]

        # scores for each sentence in the beam
        beam_scores = src_enc.new(bs, beam_size).fill_(0)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view(-1)

        # current position
        cur_len = 1

        # cache compute states
        cache = {'cur_len': 0}

        # done sentences
        done = [False for _ in range(bs)]

        while cur_len < max_len:

            # compute word scores
            x = generated[cur_len - 1].unsqueeze(-1)
            tgt_mask = ( generated[:cur_len] != tgt_vocab.stoi[config.PAD] ).transpose(0, 1).unsqueeze(-2)
            tgt_mask = tgt_mask & Variable(
                    subsequent_mask(cur_len).type_as(tgt_mask.data)
                    )

            tensor = net.decode(
                    src_enc, src_mask, x,
                    tgt_mask[:, cur_len-1, :].unsqueeze(-2), cache
                    )
            
            assert tensor.size() == (bs * beam_size, 1, config.d_model)
            tensor = tensor.data.view(bs * beam_size, -1)               # (bs * beam_size, dim)
            scores = net.generator(tensor)  # (bs * beam_size, n_words)
            assert scores.size() == (bs * beam_size, n_words)

            # select next words with scores
            _scores = scores + beam_scores[:, None].expand_as(scores)  # (bs * beam_size, n_words)
            _scores = _scores.view(bs, beam_size * n_words)            # (bs, beam_size * n_words)

            next_scores, next_words = torch.topk(_scores, 2 * beam_size, dim=1, largest=True, sorted=True)
            assert next_scores.size() == next_words.size() == (bs, 2 * beam_size)

            # next batch beam content
            # list of (bs * beam_size) tuple(next hypothesis score, next word, current position in the batch)
            next_batch_beam = []

            # for each sentence
            for sent_id in range(bs):

                # if we are done with this sentence
                done[sent_id] = done[sent_id] or generated_hyps[sent_id].is_done(next_scores[sent_id].max().item())
                if done[sent_id]:
                    next_batch_beam.extend([(0, tgt_vocab.stoi[config.PAD], 0)] * beam_size)  # pad the batch
                    continue

                # next sentence beam content
                next_sent_beam = []

                # next words for this sentence
                for idx, value in zip(next_words[sent_id], next_scores[sent_id]):

                    # get beam and word IDs
                    beam_id = idx // n_words
                    word_id = idx % n_words

                    # end of sentence, or next word
                    if word_id == tgt_vocab.stoi[config.EOS] or cur_len + 1 == max_len:
                        generated_hyps[sent_id].add(generated[:cur_len, sent_id * beam_size + beam_id].clone(), value.item())
                    else:
                        next_sent_beam.append((value, word_id, sent_id * beam_size + beam_id))

                    # the beam for next step is full
                    if len(next_sent_beam) == beam_size:
                        break

                # update next beam content
                assert len(next_sent_beam) == 0 if cur_len + 1 == max_len else beam_size
                if len(next_sent_beam) == 0:
                    next_sent_beam = [(0, tgt_vocab.stoi[config.PAD], 0)] * beam_size  # pad the batch
                next_batch_beam.extend(next_sent_beam)
                assert len(next_batch_beam) == beam_size * (sent_id + 1)

            # sanity check / prepare next batch
            assert len(next_batch_beam) == bs * beam_size
            beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
            beam_words = generated.new([x[1] for x in next_batch_beam])
            beam_idx = src_len.new([x[2] for x in next_batch_beam])

            # re-order batch and internal states
            generated = generated[:, beam_idx]
            generated[cur_len] = beam_words
            for k in cache.keys():
                if k != 'cur_len':
                    cache[k] = (cache[k][0][beam_idx], cache[k][1][beam_idx])

            # update current length
            cur_len = cur_len + 1
            cache['cur_len'] = cur_len - 1

            # stop when we are done with each sentence
            if all(done):
                break

        # select the best hypotheses
        tgt_len = src_len.new(bs)
        best = []

        for i, hypotheses in enumerate(generated_hyps):
            best_hyp = max(hypotheses.hyp, key=lambda x: x[0])[1]
            tgt_len[i] = len(best_hyp) + 1  # +1 for the <EOS> symbol
            best.append(best_hyp)

        # generate target batch
        decoded = src_len.new(tgt_len.max().item(), bs).fill_(tgt_vocab.stoi[config.PAD])
        for i, hypo in enumerate(best):
            decoded[:tgt_len[i] - 1, i] = hypo
            decoded[tgt_len[i] - 1, i] = tgt_vocab.stoi[config.PAD]

        return decoded.transpose(0, 1), tgt_len


def load_model(checkpoint_path):
    """
    Reload a checkpoint if we find one.
    """
    assert os.path.isfile(checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    config.src_n_vocab = ckpt['net']['module.src_emb.0.emb.weight'].size(0)
    config.tgt_n_vocab = ckpt['net']['module.tgt_emb.0.emb.weight'].size(0)
    net, _ = model.get()

    # reload model parameters
    s_dict = {}
    for k in ckpt["net"]:
        new_k = k[7:]
        s_dict[new_k] = ckpt["net"][k]

    net.load_state_dict(s_dict)

    src_vocab = ckpt["src_vocab"]
    tgt_vocab = ckpt["tgt_vocab"]
    return net, src_vocab, tgt_vocab


def gen_batch2str(src, tgt, generated, gen_len, src_vocab, tgt_vocab):
    generated = generated.cpu().numpy().tolist()
    gen_len = gen_len.cpu().numpy().tolist()
    src = src.cpu().numpy().tolist()
    tgt = tgt.cpu().numpy().tolist()
    translated = []
    for i, l in enumerate(generated):
        l = l[:gen_len[i]]
        sys_sent = " ".join([tgt_vocab.itos[tok] for tok in l])
        src_sent = " ".join([src_vocab.itos[tok] for tok in src[i]])
        ref_sent = " ".join([tgt_vocab.itos[tok] for tok in tgt[i]])
        sys_sent = remove_special_tok(remove_bpe(sys_sent))
        src_sent = remove_special_tok(remove_bpe(src_sent))
        ref_sent = remove_special_tok(remove_bpe(ref_sent))
        translated.append("S: " + src_sent)
        translated.append("T: " + ref_sent)
        translated.append("H: " + sys_sent)
    return translated


def translate(args, net, src_vocab, tgt_vocab):
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

    infer_dataiter = iter(infer_dataset.get_iterator(True, True))

    for raw_batch in infer_dataiter:
        src_mask = (raw_batch.src != src_vocab.stoi[config.PAD]).unsqueeze(-2)
        if args.use_cuda:
            src, src_mask = raw_batch.src.cuda(), src_mask.cuda()
        if args.greedy:
            generated, gen_len = greedy(args, net, src, src_mask, src_vocab, tgt_vocab)
        else:
            generated, gen_len = generate_beam(args, net, src, src_mask, src_vocab, tgt_vocab)
        new_translations = gen_batch2str(src, raw_batch.tgt, generated, gen_len, src_vocab, tgt_vocab)
        for res_sent in new_translations:
            print(res_sent)
        translated.extend(new_translations)

    return translated


def main():
    "done"
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-ckpt', required=True)
    parser.add_argument('-k', '--beam_size', type=int, default=4)
    parser.add_argument('-lp', '--length_penalty', type=float, default=0.7)
    parser.add_argument('--early_stopping', action="store_true")
    parser.add_argument('-max_len', type=int, default=250)
    parser.add_argument('--gen_a', type=float, default=1.3)
    parser.add_argument('--gen_b', type=int, default=5)
    parser.add_argument('-max_ratio', type=int, default=1.5)
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-text', type=str, required=True)
    parser.add_argument('-ref_text', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--max_batch_size', type=int, default=None)
    parser.add_argument('--tokens_per_batch', type=int, default=None)
    parser.add_argument('--greedy', action='store_true')

    args = parser.parse_args()
    args.use_cuda = ( args.no_cuda == False ) and torch.cuda.is_available()

    assert args.beam_size > 0
    assert args.max_len > 10

    net, src_vocab, tgt_vocab = load_model(args.ckpt, net)

    if args.use_cuda:
        net = net.cuda()
    
    fpath = args.text
    try:
        args.text = open(fpath, encoding='utf-8').read().split('\n')[:-1]
    except:
        print("error opening or reading text file")
    
    fpath = args.ref_text
    try:
        args.ref_text = open(fpath, encoding='utf-8').read().split('\n')[:-1]
    except:
        print("error opening or reading text file")
    
    translate(args, net, src_vocab, tgt_vocab)

if __name__ == '__main__':
    main()
