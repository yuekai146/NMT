from common import config
from dataset import Dataset
from translate import load_model
import argparse
import model
import numpy as np
import os
import random
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from utils import subsequent_mask, remove_bpe, remove_special_tok
import math


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


def _get_scores(args, net, active_func, src, src_mask, indices, src_vocab, tgt_vocab):
    net.eval()
    max_len = min(args.max_len, int(args.gen_a * src.size(1) + args.gen_b))
    average_by_length = (active_func != "tte")
    result = []
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
        query_scores = src.new_zeros(bsz).float()

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
            scores = net.generator(logit).exp().view(bsz, -1).data
            
            # Calculate activation function value
            # The smaller query score is, the more uncertain model is about the sentence
            if active_func == "lc":
                q_scores, _ = torch.topk(scores, 1, dim=-1)
                q_scores = -(1.0 - q_scores).squeeze()
            elif active_func == "margin":
                q_scores, _ = torch.topk(scores, 2, dim=-1)
                q_scores = q_scores[:, 0] - q_scores[:, 1]
            elif active_func == "te" or active_func == "tte":
                q_scores = -torch.distributions.categorical.Categorical(probs=scores).entropy()
            q_scores = q_scores.view(bsz)
            assert q_scores.size() == (bsz,), q_scores
            
            query_scores = query_scores + unfinished_sents.float() * q_scores
            
            next_words = torch.topk(scores, 1)[1].squeeze()

            next_words = next_words.view(bsz)
            assert next_words.size()  == (bsz,)
            generated[:, cur_len] = next_words * unfinished_sents + tgt_vocab.stoi[config.PAD] * (1 - unfinished_sents)
            gen_len.add_(unfinished_sents)
            unfinished_sents.mul_(next_words.ne(tgt_vocab.stoi[config.EOS]).long())
            cur_len = cur_len + 1
            cache['cur_len'] = cur_len - 1

            if unfinished_sents.max() == 0:
                break

        '''
        if cur_len == max_len:
            generated[:, -1].masked_fill_(unfinished_sents.bool(), tgt_vocab.stoi[config.EOS])
        
        translated = gen_batch2str(src, generated[:, :cur_len], gen_len, src_vocab, tgt_vocab)
        for new_sent in translated:
            print(new_sent)
        '''

        if average_by_length:
            query_scores = query_scores / gen_len.float()
        query_scores = query_scores.cpu().numpy().tolist()
        indices = indices.tolist()
        assert len(query_scores) == len(indices)
        for q_s, idx in zip(query_scores, indices):
            result.append((q_s, idx))
    return result


def split_batch(src, indices, max_batch_size=800):
    bsz = src.size(0)
    if bsz <= max_batch_size:
        splited = False
        return src, indices, splited
    else:
        src = torch.split(src, max_batch_size)
        indices_chunks = []
        splited = True
        for chunk in src:
            bsz = chunk.size(0)
            indices_chunks.append(indices[:bsz])
            indices = indices[bsz:]
        return src, indices_chunks, splited


def get_scores(args, net, active_func, infer_dataiter, src_vocab, tgt_vocab):
    results = []
    for (src, indices) in infer_dataiter:
        src, indices, splited = split_batch(src, indices)

        if splited:
            assert len(src) == len(indices)
            for src_chunk, indices_chunk in zip(src, indices):
                assert src_chunk.size(0) == len(indices_chunk)
                src_mask = (src_chunk != src_vocab.stoi[config.PAD]).unsqueeze(-2)
                if args.use_cuda:
                    src_chunk, src_mask = src_chunk.cuda(), src_mask.cuda()
                result = _get_scores(args, net, active_func, src_chunk, src_mask, indices_chunk, src_vocab, tgt_vocab)
                results.extend(result)    
        else:
            src_mask = (src != src_vocab.stoi[config.PAD]).unsqueeze(-2)
            if args.use_cuda:
                src, src_mask = src.cuda(), src_mask.cuda()
            result = _get_scores(args, net, active_func, src, src_mask, indices, src_vocab, tgt_vocab)
            results.extend(result)    

    return results


def query_instances(args, unlabeled_dataset, oracle, active_func="random", labeled_dataset=None):
    # lc stands for least confident
    # te stands for token entropy
    # tte stands for total token entropy
    assert active_func in ["random", "longest", "shortest", "lc", "margin", "te", "tte", "dden"]

    # lengths represents number of tokens, so BPE should be removed
    lengths = np.array([len(remove_special_tok(remove_bpe(s)).split()) for s in unlabeled_dataset])
    
    # Preparations before querying instances
    if active_func in ["lc", "margin", "te", "tte"]:
        # Reloading network parameters
        args.use_cuda = ( args.no_cuda == False ) and torch.cuda.is_available()
        net, _ = model.get()

        assert os.path.exists(args.checkpoint)
        net, src_vocab, tgt_vocab = load_model(args.checkpoint, net)

        if args.use_cuda:
            net = net.cuda()
        
        # Initialize inference dataset (Unlabeled dataset)
        infer_dataset = Dataset(unlabeled_dataset, src_vocab)
        if args.batch_size is not None:
            infer_dataset.BATCH_SIZE = args.batch_size
        if args.max_batch_size is not None:
            infer_dataset.max_batch_size = args.max_batch_size
        if args.tokens_per_batch is not None:
            infer_dataset.tokens_per_batch = args.tokens_per_batch

        infer_dataiter = iter(infer_dataset.get_iterator(
            shuffle=True, group_by_size=True, include_indices=True
            ))

    # Start ranking unlabeled dataset
    indices = np.arange(len(unlabeled_dataset))
    if active_func == "random":
        np.random.shuffle(indices)
        for idx in indices:
            print("S:", unlabeled_dataset[idx])
            print("T:", oracle[idx])
            print("V:", str(0.0))
            print("I:", args.input, args.reference, idx)
    elif active_func == "longest":
        indices = indices[np.argsort(-lengths[indices])]
        for idx in indices:
            print("S:", unlabeled_dataset[idx])
            print("T:", oracle[idx])
            print("V:", -lengths[idx])
            print("I:", args.input, args.reference, idx)
    elif active_func == "shortest":
        indices = indices[np.argsort(lengths[indices])]
        for idx in indices:
            print("S:", unlabeled_dataset[idx])
            print("T:", oracle[idx])
            print("V:", lengths[idx])
            print("I:", args.input, args.reference, idx)
    elif active_func in ["lc", "margin", "te", "tte"]:
        result = get_scores(args, net, active_func, infer_dataiter, src_vocab, tgt_vocab)
        result = sorted(result, key=lambda item:item[0])
        indices = [item[1] for item in result]
        indices = np.array(indices).astype('int')

        for idx in range(len(result)):
            print("S:", unlabeled_dataset[result[idx][1]])
            print("T:", oracle[result[idx][1]])
            print("V:", result[idx][0])
            print("I:", args.input, args.reference, result[idx][1])
    elif active_func == "dden":
        punc = [".", ",", "?", "!", "'", "<", ">", ":", ';', "(", ")",  
                "{", "}", "[", "]", "-", "..", "...", "...."]
        lamb1 = 1
        lamb2 = 1
        p_u = {}
        unlabeled_dataset_without_bpe = []
        labeled_dataset_without_bpe = [[], []]
        for s in unlabeled_dataset:
            unlabeled_dataset_without_bpe.append(remove_special_tok(remove_bpe(s)))
        for s in labeled_dataset[0]:
            labeled_dataset_without_bpe[0].append(remove_special_tok(remove_bpe(s)))
        for s in labeled_dataset[1]:
            labeled_dataset_without_bpe[1].append(remove_special_tok(remove_bpe(s)))
        for s in unlabeled_dataset_without_bpe:
            sentence = s.split()
            for token in sentence:
                if token not in punc:
                    if token in p_u.keys():
                        p_u[token] += 1
                    else:
                        p_u[token] = 1
        total_dden = 0
        for token in p_u.keys():
            p_u[token] = math.log(p_u[token] + 1)
            total_dden += p_u[token]
        for token in p_u.keys():
            p_u[token] /= total_dden
        count_l = {}
        for s in labeled_dataset_without_bpe[0]:
            sentence = s.split()
            for token in sentence:
                if token not in punc:
                    if token in count_l.keys():
                        count_l[token] += 1
                    else:
                        count_l[token] = 1
        dden = []
        for s in unlabeled_dataset_without_bpe:
            sentence = s.split()
            len_for_sentence = 0
            sum_for_sentence = 0
            for token in sentence:
                if token not in punc:
                    if token in count_l.keys():
                        sum_for_sentence += 0
                        sum_for_sentence += p_u[token] * math.exp(-lamb1 * count_l[token])
                    else:
                        sum_for_sentence += p_u[token]
                len_for_sentence += 1
            if len_for_sentence != 0:
                sum_for_sentence /= len_for_sentence
            dden.append(sum_for_sentence)
        unlabeled_with_index = []
        for i in range(len(unlabeled_dataset)):
            unlabeled_with_index.append((dden[i], i))
        unlabeled_with_index.sort(key = lambda x:x[0], reverse = True)
        count_batch = {}
        dden_new = []
        for _, i in unlabeled_with_index:
            sentence = unlabeled_dataset_without_bpe[i].split()
            len_for_sentence = 0
            sum_for_sentence = 0
            for token in sentence:
                if token not in punc:
                    p_tmp = p_u[token]
                    if token in count_batch.keys():
                        p_tmp = 0
                        p_tmp *= math.exp(-lamb2 * count_batch[token])
                    if token in count_l.keys():
                        #p_tmp = 0
                        p_tmp *= math.exp(-lamb1 * count_l[token])
                    sum_for_sentence += p_tmp
                len_for_sentence += 1
            for token in sentence:
                if token not in punc:
                    if token in count_batch.keys():
                        count_batch[token] += 1
                    else:
                        count_batch[token] = 1
            if len_for_sentence != 0:
                sum_for_sentence /= len_for_sentence
            dden_new.append((sum_for_sentence, i))
        dden_new.sort(key = lambda x:x[1])
        dden_sort = []
        for dden_num, _ in dden_new:
            dden_sort.append(dden_num)
        ddens = np.array(dden_sort)
        indices = indices[np.argsort(-ddens)]
        for idx in indices:
            print("S:", unlabeled_dataset[idx])
            print("T:", oracle[idx])
            print("V:", -ddens[idx])
            print("I:", args.input, args.reference, idx)

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help='two modes, score or modify')
    
    # Add argument for score mode
    parser_score = subparsers.add_parser(
            'score', help='Get active function scores for each unlabeled sentence'
            )
    parser_score.add_argument(
            "-a", "--active_func", type=str,
            help="Active query function type", required=True
            )
    parser_score.add_argument(
            "-i", "--input", type=str,
            help="where to read unlabeled data"
            )
    parser_score.add_argument(
            "-lb", "--input_labeled", type=str,
            help="where to read labeled data"
            )
    parser_score.add_argument(
            "-ref", "--reference", type=str,
            help="where to read oracle data"
            )
    parser_score.add_argument(
            '-ckpt', '--checkpoint', type=str,
            help="Checkpoint path to reload network parameters"
            )
    parser_score.add_argument(
            '-max_len', type=int, default=250,
            help="Maximum length for generating translations"
            )
    parser_score.add_argument(
            '-no_cuda', action="store_true",
            help="Use cpu to do translation"
            )
    parser_score.add_argument(
            '--batch_size', type=int, default=None,
            help="Batch size for generating translations"
            )
    parser_score.add_argument(
            '--max_batch_size', type=int, default=None,
            help="Maximum batch size if tokens_per_batch is not None"
            )
    parser_score.add_argument(
            '--tokens_per_batch', type=int, default=None,
            help="Maximum number of tokens in a batch when generating translations"
            )
    parser_score.add_argument(
            '-gen_a', type=float, default=1.3,
            help="Maximum length for generating translations is gen_a * l + gen_b"
            )
    parser_score.add_argument(
            '-gen_b', type=float, default=5,
            help="Maximum length for generating translations is gen_a * l + gen_b"
            )
    
    # Add argument for modify mode
    parser_modify = subparsers.add_parser(
            'modify', help='Change labeled, unlabeled oracle dataset after activation function values is calculated'
            )
    parser_modify.add_argument(
            "-U", "--unlabeled_dataset", type=str,
            help="where to read unlabelded dataset", required=True
            )
    parser_modify.add_argument(
            "-L", "--labeled_dataset", type=str,
            help="where to read labeled dataset, split by comma, e.g. l.de,l.en", required=True
            )
    parser_modify.add_argument(
            "--oracle", type=str,
            help="where to read oracle dataset",
            required=True
            )
    parser_modify.add_argument(
            "-tb", "--tok_budget", type=int,
            help="Token budget", required=True
            )
    parser_modify.add_argument(
            "-OU", "--output_unlabeled_dataset", type=str,
            help="path to store new unlabeled dataset", required=True
            )
    parser_modify.add_argument(
            "-OL", "--output_labeled_dataset", type=str,
            help="path to store new labeled dataset", required=True
            )
    parser_modify.add_argument(
            "-OO", "--output_oracle", type=str,
            help="path to oracle", required=True
            )
    parser_modify.add_argument('-AO', '--active_out', type=str,
            help="path to active function output"
            )
    args = parser.parse_args()

    args.mode = "score" if hasattr(args, "active_func") else "modify"
    
    if args.mode == "score":
        f = open(args.input, 'r')
        text = f.read().split('\n')
        if text[-1] == "":
            text = text[:-1]
        f.close()
        
        f = open(args.reference, 'r')
        ref_text = f.read().split('\n')
        if ref_text[-1] == "":
            ref_text = ref_text[:-1]
        f.close()

        if hasattr(args, "input_labeled"):
            lab_text = []
            f = open(args.input_labeled + '.de', 'r')
            labeled_text = f.read().split('\n')
            if labeled_text[-1] == "":
                labeled_text = labeled_text[:-1]
            f.close()
            lab_text.append(labeled_text)

            f = open(args.input_labeled + '.en', 'r')
            labeled_text = f.read().split('\n')
            if labeled_text[-1] == "":
                labeled_text = labeled_text[:-1]
            f.close()
            lab_text.append(labeled_text)

            query_instances(args, text, ref_text, args.active_func, lab_text)
        else:
            query_instances(args, text, ref_text, args.active_func)

    else:
        # Read labeled and unlabeled datasets
        f = open(args.unlabeled_dataset, 'r')
        unlabeled_dataset = f.read().split("\n")[:-1]
        f.close()

        src_labeled_dataset, tgt_labeled_dataset = args.labeled_dataset.split(",")
        labeled_dataset = []
        f = open(src_labeled_dataset, 'r')
        labeled_dataset.append(f.read().split("\n")[:-1])
        f.close()

        f = open(tgt_labeled_dataset, 'r')
        labeled_dataset.append(f.read().split("\n")[:-1])
        f.close()

        # Read oracle
        f = open(args.oracle, "r")
        oracle = f.read().split("\n")[:-1]
        assert len(oracle) == len(unlabeled_dataset)

        # Read active out
        f = open(args.active_out, "r")
        active_out = f.read().split("\n")[:-1]

        # Sort active_out
        assert len(active_out) % 4 == 0
        assert len(active_out) / 4 == len(oracle)
        active_out = [[active_out[i], active_out[i+1], float(active_out[i+2].split(' ')[-1]), active_out[i+3]] for i in range(0, len(active_out), 4)]
        active_out = sorted(active_out, key=lambda item: item[2])

        # Change datasets
        indices = np.arange(len(active_out))
        lengths = np.array([len(remove_special_tok(remove_bpe(item[0][len("S: "):])).split(' ')) for item in active_out])
        include = np.cumsum(lengths) <= args.tok_budget
        not_include = (1 - include).astype('bool')
        include = indices[include]
        not_include = indices[not_include]
        
        for idx in include:
            labeled_dataset[0].append(active_out[idx][0][len("S: "):])
            labeled_dataset[1].append(active_out[idx][1][len("T: "):])
        
        unlabeled_dataset = []
        oracle = []
        for idx in not_include:
            unlabeled_dataset.append(active_out[idx][0][len("S: "):])
            oracle.append(active_out[idx][1][len("T: "):])

        combined = list(zip(unlabeled_dataset, oracle))
        random.shuffle(combined)

        unlabeled_dataset[:], oracle[:] = zip(*combined)
        
        # Store new labeled, unlabeled, oracle dataset
        f = open(args.output_unlabeled_dataset, 'w')
        f.write("\n".join(unlabeled_dataset) + "\n")
        f.close()

        output_src_labeled_dataset, output_tgt_labeled_dataset = args.output_labeled_dataset.split(",")
        f = open(output_src_labeled_dataset, 'w')
        f.write("\n".join(labeled_dataset[0]) + "\n")
        f.close()

        f = open(output_tgt_labeled_dataset, 'w')
        f.write("\n".join(labeled_dataset[1]) + "\n")
        f.close()

        f = open(args.output_oracle, 'w')
        f.write("\n".join(oracle) + "\n")
        f.close()


if __name__ == "__main__":
    main()
