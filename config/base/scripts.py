from utils import average_checkpoints, last_n_checkpoints
import argparse
import torch


def avg_ckpts(path, n, output, upper_bound=None):
    inputs = last_n_checkpoints(path, n, upper_bound)
    new_state = average_checkpoints(inputs)
    torch.save(new_state, output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--type', type=str, help="which script you want to execute")

    parser.add_argument('-p', '--path', type=str, default="./checkpoints")
    parser.add_argument('-n', '--num_ckpts', type=int, default=None)
    parser.add_argument('-o', '--output', type=str, default=None)
    parser.add_argument('-ub', '--upper_bound', type=int, default=None)

    args = parser.parse_args()

    if args.type == "avg_ckpts":
        avg_ckpts(args.path, args.num_ckpts, args.output, args.upper_bound)


if __name__ == "__main__":
    main()
