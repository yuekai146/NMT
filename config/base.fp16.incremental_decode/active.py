# Implement random, longest, shortest baselines
import argparse
import numpy as np


def query_instances(unlabeled_dataset, active_func="random", tok_budget=None):
    assert active_func in ["random", "longest", "shortest"]
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
    queries, query_indices = query_instances(unlabeled_dataset, args.active_func, args.tok_budget)

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
