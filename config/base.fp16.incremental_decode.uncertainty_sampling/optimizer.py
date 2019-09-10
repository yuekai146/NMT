from common import config
import torch


class Noam_Opt:

    def __init__(self, lr, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self._rate = 0
        self.lr_step = (lr - 0.0) / self.warmup
        self.decay_factor = lr * self.warmup ** 0.5

    def step(self):
        # Update network parameters and rate
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        # Implement of lr scheduler
        if step is None:
            step = self._step

        if step < self.warmup:
            return step * self.lr_step
        else:
            return self.decay_factor * step ** (-0.5)


def get(net):
    return Noam_Opt(
            config.lr, config.opt_warmup,
            torch.optim.Adam(net.parameters(), lr=config.lr, betas=(config.beta1, config.beta2), eps=config.opt_eps, weight_decay=config.weight_decay)
            )
