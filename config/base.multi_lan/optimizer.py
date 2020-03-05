from common import config
import torch


class Noam_Opt:

    def __init__(self, lr, warmup, optimizer, init_lr=None):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        if init_lr is None:
            self.lr_step = (lr - 0.0) / self.warmup
            self._rate = 0.0
            self.init_lr = 0.0
        else:
            self.lr_step = (lr - init_lr) / warmup
            self._rate = init_lr
            self.init_lr = init_lr
        self.decay_factor = lr * self.warmup ** 0.5

    def step(self):
        # Update network parameters and rate
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        self._step += 1

    def rate(self, step=None):
        # Implement of lr scheduler
        if step is None:
            step = self._step

        if step < self.warmup:
            return self.init_lr + step * self.lr_step
        else:
            return self.decay_factor * step ** (-0.5)


def get(net):
    return Noam_Opt(
            config.lr, config.opt_warmup,
            torch.optim.Adam(net.parameters(), lr=config.lr, betas=(config.beta1, config.beta2), eps=config.opt_eps, weight_decay=config.weight_decay), init_lr=config.init_lr
            )
