from common import config
import torch


class Noam_Opt:

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

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

        return self.factor * ( self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))


def get(net):
    return Noam_Opt(
            config.d_model, config.opt_factor, config.opt_warmup,
            torch.optim.Adam(net.parameters(), lr=config.opt_init_lr, betas=(config.beta1, config.beta2), eps=config.opt_eps, weight_decay=config.weight_decay)
            )
