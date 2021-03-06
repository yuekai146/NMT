# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import time
from collections import OrderedDict
from logging import getLogger

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_

from common import config
from utils import get_batch
import optimizer
import model
import dataset


logger = getLogger()


class Trainer(object):

    def __init__(self, params):
        """
        Initialize trainer.
        """
        self.params = params

        # epoch / iteration size
        assert isinstance(config.epoch_size, int)
        assert config.epoch_size >= 1
        self.epoch_size = config.epoch_size

        # network and criterion
        net, criterion = model.get()
        self.net = net
        self.criterion = criterion

        # data iterators
        self.iterators = {}
        train_iter, valid_iter, SRC_TEXT, TGT_TEXT = dataset.get()
        self.iterators["train"] = train_iter
        self.iterators["valid"] = valid_iter
        self.num_train = len(train_iter)
        self.SRC_TEXT = SRC_TEXT
        self.TGT_TEXT = TGT_TEXT

        # Multi-GPU
        if config.multi_gpu:
            logger.info("Using nn.parallel.DistributedDataParallel ...")
            self.net = nn.parallel.DistributedDataParallel(
                    self.net, device_ids=[params.local_rank], output_device=params.local_rank
                    )
            """
            self.criterion = nn.parallel.DistributedDataParallel(
                    self.criterion, device_ids=[params.local_rank], output_device=params.local_rank
                    )
            """

        # set optimizers
        self.opt = optimizer.get(self.net)

        # validation metrics
        self.best_metrics = {}
        for k in config.valid_metrics.keys():
            factor = config.valid_metrics[k]
            self.best_metrics[k] = [config.init_metric * factor, factor]

        # training statistics
        self.epoch = 0
        self.n_iter = 0
        self.n_total_iter = 0
        self.n_sentences = 0
        self.stats = OrderedDict(
            [('processed_s', 0), ('processed_w', 0)] +
            [('MT-%s-%s-loss' % (config.SRC_LAN, config.TGT_LAN), [])] +
            [('MT-%s-%s-ppl' % (config.SRC_LAN, config.TGT_LAN), [])]
        )
        self.last_time = time.time()

        # reload potential checkpoints
        self.reload_checkpoint()



    def optimize(self, loss):
        """
        Optimize.
        """
        # check NaN
        if (loss != loss).data.any():
            logger.warning("NaN detected")
            # exit()

        assert isinstance(config.accumulate_gradients, int)
        if config.accumulate_gradients > 1:
            loss = loss / config.accumulate_gradients
        # regular optimization
        if self.n_iter % config.accumulate_gradients == 0:
            loss.backward()
            if config.clip_grad_norm > 0:
                for name in names:
                    # norm_check_a = (sum([p.grad.norm(p=2).item() ** 2 for p in self.parameters[name]])) ** 0.5
                    clip_grad_norm_(self.net.parameters(), config.clip_grad_norm)
                    # norm_check_b = (sum([p.grad.norm(p=2).item() ** 2 for p in self.parameters[name]])) ** 0.5
                    # print(name, norm_check_a, norm_check_b)
            self.opt.step()
            self.opt.optimizer.zero_grad()
        else:
            loss.backward()


    def iter(self):
        """
        End of iteration.
        """
        self.n_iter += 1
        self.n_total_iter += 1
        self.print_stats()


    def print_stats(self):
        """
        Print statistics about the training.
        """
        if self.n_total_iter % config.print_interval != 0:
            return

        num_sents_trained_within_epoch = self.n_sentences % self.num_train
        s_iter = "{} - epoch_{}:{}% - ".format(
                self.n_total_iter,
                self.epoch,
                int(100 * num_sents_trained_within_epoch / self.num_train)
                )
        s_stat = ' || '.join([
            '{}: {:7.4f}'.format(k, np.mean(v)) for k, v in self.stats.items()
            if type(v) is list and len(v) > 0
        ])
        for k in self.stats.keys():
            if type(self.stats[k]) is list:
                del self.stats[k][:]

        # learning rates
        s_lr = " - LR: {:.4e}".format(self.opt._rate)

        # processing speed
        new_time = time.time()
        diff = new_time - self.last_time
        s_speed = "{:7.2f} sent/s - {:8.2f} words/s - ".format(
            self.stats['processed_s'] * 1.0 / diff,
            self.stats['processed_w'] * 1.0 / diff
        )
        self.stats['processed_s'] = 0
        self.stats['processed_w'] = 0
        self.last_time = new_time

        # log speed + stats + learning rate
        logger.info(s_iter + s_speed + s_stat + s_lr)


    def save_checkpoint(self, name):
        """
        done
        Save the model / checkpoints.
        """
        if self.params.local_rank != 0:
            return

        path = os.path.join(config.dump_path, '%s.pth' % name)
        logger.info("Saving %s to %s ..." % (name, path))

        ckpt = {
            'epoch': self.epoch,
            'n_total_iter': self.n_total_iter,
            'best_metrics': self.best_metrics
        }

        logger.warning(f"Saving network parameters ...")
        ckpt["net"] = self.net.state_dict()

        logger.warning(f"Saving optimizer state_dict...")
        ckpt["opt"] = {}
        for k in vars(self.opt).keys():
            if k != "optimizer":
                ckpt["opt"][k] = getattr(self.opt, k)
            else:
                ckpt["opt"]["optimizer_state_dict"] = self.opt.optimizer.state_dict()
        ckpt["src_vocab"] = self.SRC_TEXT.vocab
        ckpt["tgt_vocab"] = self.TGT_TEXT.vocab

        torch.save(ckpt, path)


    def reload_checkpoint(self):
        """
        done
        Reload a checkpoint if we find one.
        """
        if config.continue_path is not None:
            checkpoint_path = config.continue_path 
            assert os.path.isfile(checkpoint_path)
            logger.warning(f"Reloading checkpoint from {checkpoint_path} ...")
            ckpt = torch.load(checkpoint_path, map_location='cpu')

            # reload model parameters
            self.net.load_state_dict(ckpt["net"])

            # reload optimizers
            self.opt.optimizer.load_state_dict(ckpt["opt"]["optimizer_state_dict"])
            for k in ckpt["opt"].keys():
                if k != "optimizer_state_dict":
                    setattr(self.opt, k, ckpt["opt"][k])

            # reload main metrics
            self.epoch = ckpt['epoch'] + 1
            self.n_total_iter = ckpt['n_total_iter']
            self.best_metrics = ckpt['best_metrics']
            #self.best_stopping_criterion = data['best_stopping_criterion']
            logger.warning(f"Checkpoint reloaded. Resuming at epoch {self.epoch} / iteration {self.n_total_iter} ...")


    def save_periodic(self):
        """
        Save the models periodically.
        """
        if self.params.local_rank != 0:
            return

        if config.save_periodic > 0 and self.epoch % config.save_periodic == 0:
            self.save_checkpoint('checkpoint_%i' % self.epoch)


    def save_best_model(self, scores):
        """
        Save best models according to given validation metrics.
        """
        if self.params.local_rank != 0:
            return 

        for metric_name in self.best_metrics.keys():
            if metric_name not in scores:
                logger.warning("Metric \"%s\" not found in scores!" % metric_name)
                continue
            factor = self.best_metrics[metric_name][1]
            val = self.best_metrics[metric_name][0]
            if factor * scores[metric_name] > factor * val:
                self.best_metrics[metric_name] = [scores[metric_name], factor]
                logger.info('New best score for %s: %.6f' % (metric_name, scores[metric_name]))
                self.save_checkpoint('checkpoint_best_%s' % metric_name)

    
    def end_epoch(self):
        self.epoch += 1


class Enc_Dec_Trainer(Trainer):

    def train_step(self, raw_batch):
        """
        Machine translation training step.
        Can also be used for denoising auto-encoding.
        """
        self.net.train()
        if config.multi_gpu:
            self.net.module.train()
        batch = get_batch(
                raw_batch.src, raw_batch.tgt,
                self.SRC_TEXT.vocab, self.TGT_TEXT.vocab
                )
        # Get a batch of input data

        # Network forward step
        inputs = {"src":None, "tgt":None, "src_mask":None, "tgt_mask":None}
        for k in inputs.keys():
            assert k in batch
            inputs[k] = batch[k]
        logits = self.net(**inputs)

        # loss
        loss_inputs = {"logits":None, "target":None, "target_mask":None}
        for k in loss_inputs.keys():
            assert ( k in batch ) or ( k == "logits" )
            if k in batch:
                loss_inputs[k] = batch[k]
            else:
                loss_inputs[k] = logits
        loss, nll_loss = self.criterion(**loss_inputs) 
        self.stats[('MT-%s-%s-loss' % (config.SRC_LAN, config.TGT_LAN))].append(loss.item())
        self.stats[('MT-%s-%s-ppl' % (config.SRC_LAN, config.TGT_LAN))].append(nll_loss.exp().item())

        # optimize
        self.optimize(loss)

        # number of processed sentences / words
        batch_size = batch["src"].size(0)
        n_tokens = batch["n_tokens"]
        self.n_sentences += batch_size
        self.stats['processed_s'] += batch_size
        self.stats['processed_w'] += n_tokens


    def valid_step(self):
        """
        Evaluate perplexity and next word prediction accuracy.
        """
        self.net.eval()

        data_iter = iter(self.iterators["valid"].get_iterator(True, True))

        n_words = 0
        xe_loss = 0
        n_valid = 0

        for raw_batch in data_iter:
            # generate batch
            batch = get_batch(
                    raw_batch.src, raw_batch.tgt,
                    self.SRC_TEXT.vocab, self.TGT_TEXT.vocab
                    )


            # Network forward step
            inputs = {"src":None, "tgt":None, "src_mask":None, "tgt_mask":None}
            for k in inputs.keys():
                assert k in batch
                inputs[k] = batch[k]
            logits = self.net(**inputs)

            # loss
            loss_inputs = {"logits":None, "target":None, "target_mask":None}
            for k in loss_inputs.keys():
                assert ( k in batch ) or ( k == "logits" )
                if k in batch:
                    loss_inputs[k] = batch[k]
                else:
                    loss_inputs[k] = logits
            loss, nll_loss = self.criterion(**loss_inputs) 

            # update stats
            n_words += batch["n_tokens"]
            xe_loss += nll_loss.item() * batch["n_tokens"]
            n_valid += (logits.max(-1)[1] == batch["target"]).sum().item()
            
            """
            def print_valid_trans():
                trans = logits.max(-1)[1].cpu().numpy().tolist()
                sents = []
                for s in trans:
                    sents.append(
                            " ".join(self.TGT_TEXT.vocab.itos[x] for x in s).replace("@@ ", "")
                            )
                return sents


            trans_sents = print_valid_trans()
            for idx in range(len(trans_sents)):
                print(trans_sents[idx].replace("<pad>", ""))
            """


        # compute perplexity and prediction accuracy
        scores = {}
        scores['ppl'] = np.exp(xe_loss / n_words)
        scores['acc'] = 100. * n_valid / n_words
        ppl_info = '{}-{} validation ppl:{} '.format(config.SRC_LAN, config.TGT_LAN, scores['ppl'])
        acc_info = '{}-{} validation accuracy:{} '.format(config.SRC_LAN, config.TGT_LAN, scores['acc'])
        logger.info(ppl_info + '|' + acc_info)
        
        return scores



