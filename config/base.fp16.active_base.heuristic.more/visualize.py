import argparse
import matplotlib.pyplot as plt
import numpy as np


def read_res(fpath):
    f = open(fpath, 'r')
    res = f.read().split('\n')[:-1]
    res = [float(v) for v in res]
    res = np.array(res)

    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "-f", "--result_files", type=str,
            help="result files, split by comma",
            required=True
            )
    parser.add_argument(
            "-n", "--num_toks", type=int, 
            help="Number of tokens added to parallel corpus each round",
            default=279315
            )
    parser.add_argument(
            "-m", "--metric", type=str, 
            help="Which metric is used for measuring model performance",
            default="ppl"
            )
    args = parser.parse_args()
    fpaths = args.result_files.split(',')
    res = {}
    for fpath in fpaths:
        method = fpath.split('.')[0].split('_')[0]
        res[method] = read_res(fpath)

    rounds = range(1, len(res[method])+1)
    plt.xlabel("Number of tokens")
    plt.ylabel(args.metric)

    for method in res:
        plt.plot(np.array(rounds) * args.num_toks, res[method], '-', label=method)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
