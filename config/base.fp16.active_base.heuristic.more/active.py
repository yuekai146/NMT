# Implement random, longest, shortest baselines
import argparse
import numpy as np
import math

punc = [".", ",", "?", "!", "'", "<", ">", ":", ";", "(", ")", "{", "}", "[", "]", "-"]

def query_instances(labeled_dataset, unlabeled_dataset, active_func="random", tok_budget=None):
    assert active_func in ["random", "longest", "shortest", "dden", "div"]
    assert isinstance(tok_budget, int)
    lengths = np.array([len(s.split()) for s in unlabeled_dataset])
    total_num = sum(lengths)
    if total_num < tok_budget:
        tok_budget = total_num
    
    # Start ranking unlabeled dataset
    indices = np.arange(len(unlabeled_dataset))
    if active_func == "random":
        np.random.shuffle(indices)
    elif active_func == "longest":
        indices = indices[np.argsort(-lengths[indices])]
    elif active_func == "shortest":
        indices = indices[np.argsort(lengths[indices])]
    elif active_func == "dden":
        lamb1 = 1
        lamb2 = 3
        p_u = {}
        total_num_without_punc = 0
        for s in unlabeled_dataset:
            sentence = s.split()
            for token in sentence:
                if token not in punc:
                    if token in p_u.keys():
                        p_u[token] += 1
                    else:
                        p_u[token] = 1
                    total_num_without_punc += 1
        #f = open('tmp.txt', 'w')
        #for tokens in p_u.keys():
        #    f.write(tokens + '    ' + str(p_u[tokens]) + '\n')
        #f.close()

        for token in p_u.keys():
            p_u[token] /= total_num_without_punc
        sum_for_p = 0
        for token in p_u.keys():
            sum_for_p += p_u[token]
        print(sum_for_p)
        
        count_l = {}
        for s in labeled_dataset[0]:
            sentence = s.split()
            for token in sentence:
                if token not in punc:
                    if token in count_l.keys():
                        count_l[token] += 1
                    else:
                        count_l[token] = 1

        dden = []
        for s in unlabeled_dataset:
            sentence = s.split()
            len_for_sentence = 0
            sum_for_sentence = 0
            for token in sentence:
                if token not in punc:
                    if token in count_l.keys():
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
            sentence = unlabeled_dataset[i].split()
            len_for_sentence = 0
            sum_for_sentence = 0
            for token in sentence:
                if token not in punc:
                    p_tmp = p_u[token]
                    if token in count_batch.keys():
                        p_tmp = 0
                        p_tmp *= math.exp(-lamb2 * count_batch[token])
                    if token in count_l.keys():
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

    elif active_func == "div":
        tokens_l = []
        for s in labeled_dataset[0]:
            sentence = s.split()
            tokens_l.extend(sentence)
        div = []
        for s in unlabeled_dataset:
            sentence = s.split()
            sum_for_sentence = 0
            for token in sentence:
                if token in tokens_l:
                    sum_for_sentence += 1
            sum_for_sentence /= len(sentence)
            div.append(sum_for_sentence)
        divs = np.array(div)
        indices = indices[np.argsort(-divs)]

    elif active_func == "dden_div":
        lamb = 1  
        p_u = {}
        for s in unlabeled_dataset:
            sentence = s.split()
            for token in sentence:
                if token in p_u.keys():
                    p_u[token] += 1
                else:
                    p_u[token] = 1
        for token in p_u.keys():
            p_u[token] /= total_num
        sum_for_p = 0
        for token in p_u.keys():
            sum_for_p += p_u[token]
        print(sum_for_p)
        
        count_l = {}
        for s in labeled_dataset[0]:
            sentence = s.split()
            for token in sentence:
                if token in count_l.keys():
                    count_l[token] += 1
                else:
                    count_l[token] = 1

        dden = []
        for s in unlabeled_dataset:
            sentence = s.split()
            sum_for_sentence = 0
            for token in sentence:
                if token in count_l.keys():
                    sum_for_sentence += p_u[token] * math.exp(-lamb * count_l[token])
                else:
                    sum_for_sentence += p_u[token]
            sum_for_sentence /= len(sentence)
            dden.append(sum_for_sentence)

        tokens_l = []
        for s in labeled_dataset[0]:
            sentence = s.split()
            tokens_l.extend(sentence)
        div = []
        for s in unlabeled_dataset:
            sentence = s.split()
            sum_for_sentence = 0
            for token in sentence:
                if token in tokens_l:
                    sum_for_sentence += 1
            sum_for_sentence /= len(sentence)
            div.append(sum_for_sentence)

        dden_div = []
        beta = 1
        for i in range(len(dden)):
            dden_div.append(((1 + beta * beta) * dden[i] * div[i]) / (beta * beta * dden[i] + div[i]))
        indices = indices[np.argsort(-dden_div)]

    include = np.cumsum(lengths[indices]) <= tok_budget
    include = indices[include]
    return [unlabeled_dataset[idx] for idx in include], include


def label_queries(queries, oracle):
    assert isinstance(queries, np.ndarray)
    queries = queries.astype('int').tolist()
    return [oracle[idx] for idx in queries]


def change_datasets(unlabeled_dataset, labeled_dataset, labeled_queries, query_indices):
    assert len(labeled_queries[0]) == len(query_indices)
    assert len(labeled_queries[1]) == len(labeled_queries[1])
    unlabeled_dataset = [unlabeled_dataset[idx] for idx in range(len(unlabeled_dataset)) if idx not in query_indices]
    labeled_dataset[0].extend(labeled_queries[0])
    labeled_dataset[1].extend(labeled_queries[1])

    return unlabeled_dataset, labeled_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "-U", "--unlabeled_dataset", type=str,
            help="where to read unlabelded dataset", required=True
            )
    parser.add_argument(
            "-L", "--labeled_dataset", type=str,
            help="where to read labeled dataset, split by comma, e.g. l.de,l.en", required=True
            )
    parser.add_argument(
            "--oracle", type=str,
            help="where to read oracle dataset",
            required=True
            )
    parser.add_argument(
            "-tb", "--tok_budget", type=int,
            help="Token budget", required=True
            )
    parser.add_argument(
            "-OU", "--output_unlabeled_dataset", type=str,
            help="path to store new unlabeled dataset", required=True
            )
    parser.add_argument(
            "-OL", "--output_labeled_dataset", type=str,
            help="path to store new labeled dataset", required=True
            )
    parser.add_argument(
            "-OO", "--output_oracle", type=str,
            help="path to oracle", required=True
            )
    parser.add_argument(
            "-a", "--active_func", type=str,
            help="Active query function type", required=True
            )
    args = parser.parse_args()

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

    # Query instances
    queries, query_indices = query_instances(labeled_dataset, unlabeled_dataset, args.active_func, args.tok_budget)

    # Label instances
    labeled_queries = [queries]
    labeled_queries.append( label_queries(query_indices, oracle) )

    # Change datasets
    unlabeled_dataset, labeled_dataset = change_datasets(
            unlabeled_dataset, labeled_dataset, labeled_queries, query_indices
            )
    
    oracle = [oracle[idx] for idx in range(len(oracle)) if idx not in query_indices]
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
