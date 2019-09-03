import numpy as np
import torch
import math


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


def test_model_numel():
    from model import get
    net, _ = get()
    n = 0
    for p in net.parameters():
        n += p.numel()
        print(p.size())
    print(n)


def test_early_stop():
    exit()


def test_incremental():
    from common import config
    import model
    from utils import get_batch
    net, _ = model.get()
    net.eval()

    import dataset
    train_iter, _, SRC_TEXT, TGT_TEXT = dataset.get()
    data_iter = iter(train_iter.get_iterator(True, True))
    raw_batch = next(data_iter)
    batch = get_batch(
            raw_batch.src, raw_batch.tgt,
            SRC_TEXT.vocab, TGT_TEXT.vocab
            )
    for k, v in batch.items():
        try:
            print(k, v.size())
        except AttributeError:
            pass

    with torch.no_grad():
        # No incremental
        logits1 = net(src=batch['src'], tgt=batch['tgt'], src_mask=batch['src_mask'], tgt_mask=batch['tgt_mask'])

        # Incremental
        enc_out = net.encode(src=batch['src'], src_mask=batch['src_mask'])
        print("Incremental encoding finished!")
        tlen = batch['tgt'].size(1)
        cache = {}
        logits2 = []
        for i in range(tlen):
            x = batch['tgt'][:, i].unsqueeze(-1)
            logit = net.decoder(net.tgt_emb(x), enc_out, batch['src_mask'], batch['tgt_mask'][:, i, :(i+1)].unsqueeze(-2), cache)
            logit = net.generator(logit)
            
            if i >= 0:
                ref = logits1[:, i, :].exp()
                sys = logit.squeeze().exp()

                diff = torch.abs(sys - ref).cpu().numpy().tolist()
                print("Diff  = {}".format(torch.sum(torch.abs(ref - sys)).item()))
                print("Logits sys size : {}, Logits sys : {}".format(sys.size(), sys.sum().item()))
                print("Logits ref size : {}, Logits ref : {}".format(ref.size(), ref.sum().item()))
                print("\n")
            logits2.append(logit)
        logits2 = torch.cat(logits2, dim=1).contiguous()

        print("Logits1: {}".format(torch.sum(logits1.exp()).item()))
        print("Logits2: {}".format(torch.sum(logits2.exp()).item()))


def test():
    #test_mask()
    #test_model_numel()
    #test_early_stop()
    test_incremental()


if __name__ == "__main__":
    test()
