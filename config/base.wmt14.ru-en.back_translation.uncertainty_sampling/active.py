from common import config
from dataset import Dataset
import argparse
import model
import numpy as np
import os
import random
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from utils import subsequent_mask, remove_bpe, remove_special_tok


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


def gen_batch2str(generated, gen_len, tgt_vocab):
    generated = generated.cpu().numpy().tolist()
    gen_len = gen_len.cpu().numpy().tolist()
    translated = []
    for i, l in enumerate(generated):
        l = l[:gen_len[i]]
        sys_sent = " ".join([tgt_vocab.itos[tok] for tok in l])
        sys_sent = remove_special_tok(sys_sent)
        translated.append(sys_sent)
    return translated


def _get_scores(args, net, active_func, src, src_mask, indices, src_vocab, tgt_vocab):
    net.eval()
    max_len = args.max_len
    average_by_length = (active_func != "tte")
    result = []
    with torch.no_grad():
        bsz = src.size(0)
        enc_out = net.encode(src=src, src_mask=src_mask)
        generated = src.new(bsz, max_len)
        generated.fill_(tgt_vocab.stoi[config.PAD])
        generated[:, 0].fill_(tgt_vocab.stoi[config.BOS])
        generated = generated.long()
        
        max_gen_len = src_mask.sum(dim=-1).float() * args.gen_a + args.gen_b
        max_gen_len = max_gen_len.long().view(bsz,)
        max_gen_len = torch.clamp(max_gen_len, min=0, max=max_len)
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
            else:
                q_scores = scores.new_zeros(bsz)
            q_scores = q_scores.view(bsz)
            assert q_scores.size() == (bsz,), q_scores
            
            query_scores = query_scores + unfinished_sents.float() * q_scores
            
            next_words = torch.topk(scores, 1)[1].squeeze()

            next_words = next_words.view(bsz)
            assert next_words.size()  == (bsz,)
            generated[:, cur_len] = next_words * unfinished_sents + tgt_vocab.stoi[config.PAD] * (1 - unfinished_sents)
            gen_len.add_(unfinished_sents)
            unfinished_sents.mul_(next_words.ne(tgt_vocab.stoi[config.EOS]).long())
            unfinished_sents.mul_(gen_len.le(max_gen_len).long())
            cur_len = cur_len + 1
            cache['cur_len'] = cur_len - 1

            if unfinished_sents.max() == 0:
                break

        if cur_len == max_len:
            generated[:, -1].masked_fill_(unfinished_sents.bool(), tgt_vocab.stoi[config.EOS])
        
        translated = gen_batch2str(generated[:, :cur_len], gen_len, tgt_vocab)

        if average_by_length:
            query_scores = query_scores / gen_len.float()
        query_scores = query_scores.cpu().numpy().tolist()
        indices = indices.tolist()
        assert len(query_scores) == len(indices)
        assert len(query_scores) == len(translated)
        for i in range(len(query_scores)):
            result.append((query_scores[i], indices[i], translated[i]))
    return result


def split_batch(src, indices, max_batch_size=500):
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


def query_instances(args, unlabeled_dataset, oracle, active_func="random"):
    # lc stands for least confident
    # te stands for token entropy
    # tte stands for total token entropy
    assert active_func in ["random", "longest", "shortest", "lc", "margin", "te", "tte"]

    # lengths represents number of tokens, so BPE should be removed
    lengths = np.array([len(remove_special_tok(remove_bpe(s)).split()) for s in unlabeled_dataset])
    
    # Preparations before querying instances
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
        result = get_scores(args, net, active_func, infer_dataiter, src_vocab, tgt_vocab)
        random.shuffle(result)
        indices = [item[1] for item in result]
        indices = np.array(indices).astype('int')
        for idx in indices:
            print("S:", unlabeled_dataset[idx])
            print("H:", result[idx][2])
            print("T:", oracle[idx])
            print("V:", result[idx][0])
            print("I:", args.input, args.reference, idx)
    elif active_func == "longest":
        result = get_scores(args, net, active_func, infer_dataiter, src_vocab, tgt_vocab)
        result = [(
            len(remove_special_tok(remove_bpe(unlabeled_dataset[item[1]])).split(' ')),
            item[1], item[2] 
            ) for item in result]
        result = sorted(result, key=lambda item:-item[0])
        indices = [item[1] for item in result]
        indices = np.array(indices).astype('int')
        for idx in indices:
            print("S:", unlabeled_dataset[idx])
            print("H:", result[idx][2])
            print("T:", oracle[idx])
            print("V:", -result[idx][0])
            print("I:", args.input, args.reference, idx)
    elif active_func == "shortest":
        result = get_scores(args, net, active_func, infer_dataiter, src_vocab, tgt_vocab)
        result = [(
            len(remove_special_tok(remove_bpe(unlabeled_dataset[item[1]])).split(' ')),
            item[1], item[2] 
            ) for item in result]
        result = sorted(result, key=lambda item:item[0])
        indices = [item[1] for item in result]
        indices = np.array(indices).astype('int')
        for idx in indices:
            print("S:", unlabeled_dataset[idx])
            print("H:", result[idx][2])
            print("T:", oracle[idx])
            print("V:", result[idx][0])
            print("I:", args.input, args.reference, idx)
        indices = indices[np.argsort(lengths[indices])]
    elif active_func in ["lc", "margin", "te", "tte"]:
        result = get_scores(args, net, active_func, infer_dataiter, src_vocab, tgt_vocab)
        result = sorted(result, key=lambda item:item[0])
        indices = [item[1] for item in result]
        indices = np.array(indices).astype('int')

        for idx in range(len(result)):
            print("S:", unlabeled_dataset[result[idx][1]])
            print("H:", result[idx][2])
            print("T:", oracle[result[idx][1]])
            print("V:", result[idx][0])
            print("I:", args.input, args.reference, result[idx][1])


def supvised_learning_modify(args):
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
    random.shuffle(active_out)
    active_out = sorted(active_out, key=lambda item: item[2])

    # Change datasets
    indices = np.arange(len(active_out))
    lengths = np.array([len(remove_special_tok(remove_bpe(item[0][len("S: "):])).split(' ')) for item in active_out])
    include = np.cumsum(lengths) <= args.tok_budget
    not_include = (1 - include).astype('bool')
    include = indices[include]
    not_include = indices[not_include]
    
    for idx in include:
        labeled_dataset[0].append(active_out[idx][0][len("S: "):].strip())
        labeled_dataset[1].append(active_out[idx][1][len("T: "):].strip())
    
    unlabeled_dataset = []
    oracle = []
    for idx in not_include:
        unlabeled_dataset.append(active_out[idx][0][len("S: "):].strip())
        oracle.append(active_out[idx][1][len("T: "):].strip())

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


def back_translation_modify(args):
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
    assert len(active_out) % 5 == 0
    assert len(active_out) / 5 == len(oracle)
    active_out = [[active_out[i], active_out[i+1], active_out[i+2], float(active_out[i+3].split(' ')[-1]), active_out[i+4]] for i in range(0, len(active_out), 5)]
    active_out = sorted(active_out, key=lambda item: item[3])
    
    # Change datasets
    indices = np.arange(len(active_out))
    lengths = np.array([len(remove_special_tok(remove_bpe(item[0][len("S: "):])).split(' ')) for item in active_out])
    include_oracle = np.cumsum(lengths) <= args.tok_budget

    for idx in indices[include_oracle]:
        labeled_dataset[0].append(active_out[idx][0][len("S: "):].strip())
        labeled_dataset[1].append(active_out[idx][2][len("T: "):].strip())

    unlabeled_dataset = []
    oracle = []
    not_include = (1 - include_oracle).astype('bool')
    not_include = indices[not_include]
    for idx in not_include:
        unlabeled_dataset.append(active_out[idx][0][len("S: "):].strip())
        oracle.append(active_out[idx][2][len("T: "):].strip())

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

    output_new_queries_src, output_new_queries_tgt = args.output_new_queries.split(',')
    
    f = open(output_new_queries_src, 'w')
    f.write("\n".join([active_out[idx][0][len("S: "):].strip() for idx in range(len(include_oracle)) if include_oracle[idx]]) + '\n')
    f.close()
    
    f = open(output_new_queries_tgt, 'w')
    f.write("\n".join([active_out[idx][2][len("T: "):].strip() for idx in range(len(include_oracle)) if include_oracle[idx]]) + '\n')
    f.close()

    output_train_src, output_train_tgt = args.output_train.split(',')
    include_pseudo = np.cumsum(lengths) <= ( args.tok_budget + args.back_translation_tok_budget )
    include_pseudo = np.logical_xor(include_pseudo, include_oracle)
    include_pseudo = indices[include_pseudo]
    labeled_dataset[0].extend([active_out[idx][0][len("S: "):].strip() for idx in include_pseudo])
    labeled_dataset[1].extend([active_out[idx][1][len("H: "):].strip() for idx in include_pseudo])
    ots = labeled_dataset[0]
    ott = labeled_dataset[1]

    f = open(output_train_src, 'w')
    f.write("\n".join(ots) + "\n")
    f.close()

    f = open(output_train_tgt, 'w')
    f.write("\n".join(ott) + "\n")
    f.close()


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
            '-ga', '--gen_a', type=float, default=1.3,
            help="Maximum generated length is gen_a * src_l + gen_b"
            )
    parser_score.add_argument(
            '-gb', '--gen_b', type=float, default=5,
            help="Maximum generated length is gen_a * src_l + gen_b"
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
            '-bttb', '--back_translation_tok_budget', type=int, default=None,
            help="Back translation token budget, if None, the same as tok_budget"
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
    parser_modify.add_argument('-bt', '--back_translation', action='store_true',
            help="For back translation or for supervised learning"
            )
    parser_modify.add_argument('-onq', '--output_new_queries', type=str, 
            help="Path to store new queries in this round, separated by comma"
            )
    parser_modify.add_argument('-OT', '--output_train', type=str, default=None,
            help="Path to store new generated train data, separated by comma"
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
        query_instances(args, text, ref_text, args.active_func)

    elif args.mode == "modify":
        if args.back_translation_tok_budget is None:
            args.back_translation_tok_budget = args.tok_budget
        # Read labeled and unlabeled datasets
        if args.back_translation:
            back_translation_modify(args)
        else:
            supvised_learning_modify(args)
    else:
        raise("Invalid mode! Only two modes, score or modify")


if __name__ == "__main__":
    main()
