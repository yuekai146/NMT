import numpy as np
import torch


def test_mask():
    lengths = np.random.randint(low=10, high=15, size=10)
    src = []
    for l in lengths:
        sent = np.random.randint(10000, size=l)
        pad = np.zeros(16-l).astype(np.int)
        sent = np.concatenate((sent, pad), axis=None)
        src.append(sent)

    src = torch.from_numpy(np.stack(src)).long()
    mask = (src != 0).unsqueeze(-2)
    ntokens = mask.sum()

    print(ntokens)
    print(np.sum(lengths))


def test():
    test_mask()


if __name__ == "__main__":
    test()
