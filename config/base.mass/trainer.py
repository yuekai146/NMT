# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import apex
import os
import random
import time
from collections import OrderedDict
from logging import getLogger

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_

from common import config, MODE
from utils import *
from torch.autograd import Variable
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
        if MODE == "MT":
            train_iter, valid_iter, SRC_TEXT, TGT_TEXT = dataset.load()
            torch.distributed.barrier()
            print("Process {}, dataset loaded.".format(params.local_rank))
            self.iterators["train"] = train_iter
            self.iterators["valid"] = valid_iter
            self.num_train = len(train_iter)
            self.SRC_TEXT = SRC_TEXT
            self.TGT_TEXT = TGT_TEXT
        elif MODE == "MASS":
            train_iter, valid_iter, TOTAL_TEXT = dataset.load()
            torch.distributed.barrier()
            print("Process {}, dataset loaded.".format(params.local_rank))
            self.iterators["train"] = train_iter
            self.iterators["valid"] = valid_iter
            self.num_train = len(train_iter)
            self.TOTAL_TEXT = TOTAL_TEXT

        torch.distributed.barrier()

        # Multi-GPU
        assert config.amp >= 1 or not config.fp16
        if config.multi_gpu and config.fp16 == False:
            logger.info("Using nn.parallel.DistributedDataParallel ...")
            self.net = nn.parallel.DistributedDataParallel(
                    self.net, device_ids=[params.local_rank], output_device=params.local_rank
                    )

        # set optimizers
        self.opt = optimizer.get(self.net)

        torch.distributed.barrier()
        # Float16 / distributed
        if config.fp16:
            self.init_amp()
            if config.multi_gpu:
                logger.info("Using apex.parallel.DistributedDataParallel ...")
                self.net = apex.parallel.DistributedDataParallel(self.net, delay_allreduce=True)

        # validation metrics
        self.best_metrics = {}
        for k in config.valid_metrics.keys():
            factor = config.valid_metrics[k]
            self.best_metrics[k] = [config.init_metric * factor, factor]

        # early stopping metrics
        self.early_stopping_metrics = {}
        for k in self.best_metrics:
            self.early_stopping_metrics[k] = self.best_metrics[k]

        self.decrease_counts = 0
        self.decrease_counts_max = config.decrease_counts_max
        if self.decrease_counts_max is None:
            self.decrease_counts_max = self.epoch_size
        self.stopping_criterion = config.stopping_criterion
        if config.multi_gpu:
            self.should_terminate = torch.tensor(0).byte()
            self.should_terminate = self.should_terminate.cuda()
        else:
            self.should_terminate = False
        assert ( self.stopping_criterion in self.best_metrics ) or ( self.stopping_criterion is None )

        # training statistics
        self.epoch = 0
        self.n_iter = 0
        self.n_total_iter = 0
        self.n_sentences = 0

        if MODE == "MT":
            self.stats = OrderedDict(
                [('processed_s', 0), ('processed_w', 0)] +
                [('MT-%s-%s-loss' % (config.SRC_LAN, config.TGT_LAN), [])] +
                [('MT-%s-%s-ppl' % (config.SRC_LAN, config.TGT_LAN), [])]
            )
        elif MODE == "MASS":
            self.stats = OrderedDict(
                [('processed_s', 0), ('processed_w', 0)] +
                [('MASS-%s-loss' % (lan.lower()), []) for lan in config.LANS] +
                [('MASS-%s-ppl' % (lan.lower()), []) for lan in config.LANS]
            )
        self.last_time = time.time()

        # reload potential checkpoints
        self.reload_checkpoint(network_only=config.reload_network_only, optimizer_only=config.optimizer_only)
        print("Process {}, trainer initialized.".format(params.local_rank))


    def optimize(self, loss):
        """
        Optimize.
        """
        # check NaN
        if (loss != loss).data.any():
            logger.warning("NaN detected")
            # exit()

        assert isinstance(config.accumulate_gradients, int)

        if config.fp16 == False:
            # regular optimization
            loss.backward()
            if self.n_iter % config.accumulate_gradients == 0:
                if config.clip_grad_norm > 0:
                    for name in names:
                        clip_grad_norm_(self.net.parameters(), config.clip_grad_norm)
                self.opt.step()
                self.opt.optimizer.zero_grad()
        else:
            if self.n_iter % config.accumulate_gradients == 0:
                with apex.amp.scale_loss(loss, self.opt.optimizer) as scaled_loss:
                    scaled_loss.backward()
                if config.clip_grad_norm > 0:
                    clip_grad_norm_(apex.amp.master_params(self.opt.optimizer), config.clip_grad_norm)
                self.opt.step()
                self.opt.optimizer.zero_grad()
            else:
                with apex.amp.scale_loss(loss, self.opt.optimizer, delay_unscale=True) as scaled_loss:
                    scaled_loss.backward()
    

    def init_amp(self):
        assert ( config.amp == 0 and config.fp16 is False ) or  ( config.amp in [1, 2, 3] and config.fp16 is True )
        self.net, self.opt.optimizer = apex.amp.initialize(
                self.net, self.opt.optimizer,
                opt_level='O{}'.format(config.amp)
                )

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

        num_sents_trained_within_epoch = self.n_sentences
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
            'best_metrics': self.best_metrics,
            'early_stopping_metrics': self.early_stopping_metrics
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
        
        if MODE == "MT":
            ckpt["src_vocab"] = self.SRC_TEXT.vocab
            ckpt["tgt_vocab"] = self.TGT_TEXT.vocab
        elif MODE == "MASS":
            ckpt['total_vocab'] = self.TOTAL_TEXT.vocab

        torch.save(ckpt, path)


    def reload_checkpoint(self, network_only=False, optimizer_only=False):
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

            if network_only == False:
                # reload optimizers
                # self.opt.optimizer.load_state_dict(ckpt["opt"]["optimizer_state_dict"])
                for k in ckpt["opt"].keys():
                    if k != "optimizer_state_dict":
                        setattr(self.opt, k, ckpt["opt"][k])
                if optimizer_only == False:
                    # reload main metrics
                    self.epoch = ckpt['epoch'] + 1
                    self.n_total_iter = ckpt['n_total_iter']
                    self.best_metrics = ckpt['best_metrics']
                    self.early_stopping_metrics = ckpt['early_stopping_metrics']
            
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

    
    def early_stop(self, scores):
        assert isinstance(scores, dict)

        if self.params.local_rank == 0: 
            if self.stopping_criterion is not None:
                assert self.stopping_criterion in scores
                factor = self.early_stopping_metrics[self.stopping_criterion][1]
                val = self.early_stopping_metrics[self.stopping_criterion][0]

                print("New score is {}".format(factor * scores[self.stopping_criterion]))
                print("Old score is {}".format(factor * val))
                if factor * scores[self.stopping_criterion] > factor * val:
                    self.decrease_counts = 0
                    self.early_stopping_metrics[self.stopping_criterion][0] = scores[self.stopping_criterion]
                else:
                    self.decrease_counts += 1
                
                if self.decrease_counts >= self.decrease_counts_max:
                    if config.multi_gpu:
                        self.should_terminate.data = torch.tensor(1).byte().cuda()
                    else:
                        self.should_terminate = True
                
        torch.distributed.broadcast(
                tensor=self.should_terminate, src=0
                )


    def end_epoch(self, scores=None):
        if scores is not None:
            self.early_stop(scores) 
            print("Process {}, should terminate: {}".format(self.params.local_rank, self.should_terminate.item()))
            
            if config.multi_gpu:
                if self.should_terminate.item() == True:
                    exit()
            else:
                if self.should_terminate == True:
                    exit()

        self.epoch += 1
        self.n_sentences = 0


class Enc_Dec_Trainer(Trainer):
        
    
    def mt_step(self, raw_batch):
        """
        Machine translation training step.
        Can also be used for denoising auto-encoding.
        """
        self.net.train()
        self.criterion.train()
        if config.multi_gpu:
            self.net.module.train()
        batch = get_batch(
                raw_batch.src, raw_batch.tgt,
                self.SRC_TEXT.vocab, self.TGT_TEXT.vocab
                )
        batch_size = batch["src"].size(0)
        del raw_batch
        # Get a batch of input data
        
        # Network forward step
        tensor = self.net(
                batch['src'], batch['src_mask'], batch['tgt'], batch['tgt_mask'],
                src_lang=self.SRC_TEXT.vocab.stoi['<' + config.SRC_LAN.upper() + '>'],
                tgt_lang=self.TGT_TEXT.vocab.stoi['<' + config.TGT_LAN.upper() + '>']
                )

        # loss
        loss, nll_loss = self.criterion(tensor, batch['target'], batch['target_mask'])
        self.stats[('MT-%s-%s-loss' % (config.SRC_LAN, config.TGT_LAN))].append(loss.item())
        self.stats[('MT-%s-%s-ppl' % (config.SRC_LAN, config.TGT_LAN))].append(nll_loss.exp().item())
        # optimize
        self.optimize(loss)

        # number of processed sentences / words
        n_tokens = batch["n_tokens"]
        self.n_sentences += batch_size
        self.stats['processed_s'] += batch_size
        self.stats['processed_w'] += n_tokens
        
        del batch
        del loss
        del nll_loss
        del tensor
	
    
    def mass_step(self, raw_batch, lan):
        """
        Machine translation training step.
        Can also be used for denoising auto-encoding.
        """
        self.net.train()
        self.criterion.train()
        if config.multi_gpu:
            self.net.module.train()
        bsz = raw_batch.size(0)
        l = ( raw_batch != self.TOTAL_TEXT.vocab.stoi[config.PAD] ).sum(-1).view(bsz).long().numpy().tolist()
        batch = self.restricted_mask_sent(raw_batch, l)

        batch_size = batch["src"].size(0)
        del raw_batch
        # Get a batch of input data
        
        # Network forward step
        tensor = self.net(
                batch["src"], batch["src_mask"], batch["tgt"], batch["tgt_mask"],
                src_lang=self.TOTAL_TEXT.vocab.stoi['<' + lan.upper() + '>'],
                tgt_lang=self.TOTAL_TEXT.vocab.stoi['<' + lan.upper() + '>'],
                tgt_pos=batch["tgt_pos"], enc_mask=batch["enc_mask"],
                decoder_lang_id=config.LANG2IDS[lan.upper()]
                )

        # loss
        loss, nll_loss = self.criterion(tensor, batch['target'], batch['target_mask'])
        self.stats[('MASS-%s-loss' % (lan))].append(loss.item())
        self.stats[('MASS-%s-ppl' % (lan))].append(nll_loss.exp().item())
        # optimize
        self.optimize(loss)

        # number of processed sentences / words
        n_tokens = batch["n_tokens"]
        self.n_sentences += batch_size
        self.stats['processed_s'] += batch_size
        self.stats['processed_w'] += n_tokens
        
        del batch
        del loss
        del nll_loss
        del tensor
    
    def mask_word(self, w):
        mask_index = self.TOTAL_TEXT.vocab.stoi[config.MASK]
        _w_real = w
        _w_rand = np.random.randint(low=config.N_SPECIAL_TOKENS, high=config.total_n_vocab, size=w.shape)
        _w_mask = np.full(w.shape, mask_index)

        probs = torch.multinomial(
                torch.from_numpy(np.array(config.mask_probs)),
                len(_w_real), replacement=True
                )

        _w = _w_mask * (probs == 0).numpy() + _w_real * (probs == 1).numpy() + _w_rand * (probs == 2).numpy()
        return _w

    
    def unfold_segments(self, segs):
        """Unfold the random mask segments, for example:
           The shuffle segment is [2, 0, 0, 2, 0], 
           so the masked segment is like:
           [1, 1, 0, 0, 1, 1, 0]
           [1, 2, 3, 4, 5, 6, 7] (positions)
           (1 means this token will be masked, otherwise not)
           We return the position of the masked tokens like:
           [1, 2, 5, 6]
        """
        pos = []
        curr = 1   # We do not mask the start token
        for l in segs:
            if l >= 1:
                pos.extend([curr + i for i in range(l)])
                curr += l
            else:
                curr += 1
        return np.array(pos)

    
    def shuffle_segments(self, segs, unmasked_tokens):
        """
        We control 20% mask segment is at the start of sentences
                   20% mask segment is at the end   of sentences
                   60% mask segment is at random positions,
        """

        p = np.random.random()
        if p >= 0.8:
            shuf_segs = segs[1:] + unmasked_tokens
        elif p >= 0.6:
            shuf_segs = segs[:-1] + unmasked_tokens
        else:
            shuf_segs = segs + unmasked_tokens
            #while shuf_segs[0] != 0 or shuf_segs[-1]!= 0:
            #    random.shuffle(shuf_segs)
            random.shuffle(shuf_segs)
        
        if p >= 0.8:
            shuf_segs = segs[0:1] + shuf_segs
        elif p >= 0.6:
            shuf_segs = shuf_segs + segs[-1:]
        return shuf_segs

    
    def get_segments(self, mask_len, span_len):
        segs = []
        for idx in range(len(mask_len)):
            _segs = []
            while mask_len[idx] >= span_len:
                _segs.append(span_len)
                mask_len[idx] -= span_len
            if mask_len[idx] != 0:
                _segs.append(mask_len[idx])
            segs.append(_segs)
        return segs

    
    def restricted_mask_sent(self, x, l, span_len=100000):
        """ Restricted mask sents
            if span_len is equal to 1, it can be viewed as
            discrete mask;
            if span_len -> inf, it can be viewed as 
            pure sentence mask

            params: x, torch.LongTensor (bsz, slen)
                    l, List of integers (bsz)
                    span_len: int
        """

        pad_index = self.TOTAL_TEXT.vocab.stoi[config.PAD]
        mask_index = self.TOTAL_TEXT.vocab.stoi[config.MASK]

        if span_len <= 0:
            span_len = 1
        max_len = 0
        positions, inputs, targets, outputs, = [], [], [], []
        mask_len = [round(l[idx] * config.word_mass) for idx in range(len(l))]
        
        unmasked_tokens = [[0 for i in range(l[idx] - mask_len[idx] - 1)] for idx in range(len(l))]
        segs = self.get_segments(mask_len, span_len)
        
        for i in range(len(l)):
            words = x[i, :l[i]].cpu().numpy()
            shuf_segs = self.shuffle_segments(segs[i], unmasked_tokens[i])
            pos_i = self.unfold_segments(shuf_segs)
            output_i = words[pos_i].copy()
            target_i = words[pos_i - 1].copy()
            words[pos_i] = self.mask_word(words[pos_i])

            inputs.append(words)
            targets.append(target_i)
            outputs.append(output_i)
            positions.append(pos_i - 1)

        x1  = torch.LongTensor(len(l), max(l)).fill_(pad_index)
        x2  = torch.LongTensor(len(l), max(mask_len)).fill_(pad_index)
        y   = torch.LongTensor(len(l), max(mask_len)).fill_(pad_index)
        pos = torch.LongTensor(len(l), max(mask_len)).fill_(pad_index)
        l1  = l
        l2 = mask_len
        for i in range(len(l)):
            x1[i, :l1[i]].copy_(torch.LongTensor(inputs[i]))
            x2[i, :l2[i]].copy_(torch.LongTensor(targets[i]))
            y[i, :l2[i]].copy_(torch.LongTensor(outputs[i]))
            pos[i, :l2[i]].copy_(torch.LongTensor(positions[i]))
        src, tgt, target = x1, x2, y
        
        # Create network inputs
        src_mask = (src != pad_index)
        enc_mask = (src != mask_index) & src_mask
        src_mask = src_mask.unsqueeze(-2)
        enc_mask = enc_mask.unsqueeze(-2)
        bsz = tgt.size(0)

        # Create target mask
        tgt_mask = ( tgt != pad_index ).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
                subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
                )
        target_mask = ( target != pad_index )
        n_tokens = torch.sum(target_mask).long().item()

        batch = {"src":src, "tgt":tgt,
                "src_mask":src_mask, "tgt_mask":tgt_mask,
                "target":target, "target_mask":target_mask,
                "n_tokens":n_tokens, "tgt_pos":pos,
                "enc_mask":enc_mask
                } 

        # check
        if config.use_cuda:
            batch = to_cuda(batch)
        
        return batch 


    def valid_step(self):
        """
        Evaluate perplexity and next word prediction accuracy.
        """
        if MODE == "MT":
            self.iterators["valid"].tokens_per_batch = -1
            self.iterators["valid"].batch_size = 32
            data_iter = iter(self.iterators["valid"].get_iterator(True, True))

            n_words = 0
            xe_loss = 0
            n_valid = 0
            torch.cuda.empty_cache()
            
            with torch.no_grad():
                self.net.eval()
                self.criterion.eval()
                if config.multi_gpu:
                    net = self.net.module
                else:
                    net = self.net

                for i_batch, raw_batch in enumerate(data_iter):
                    # generate batch
                    batch = get_batch(
                            raw_batch.src, raw_batch.tgt,
                            self.SRC_TEXT.vocab, self.TGT_TEXT.vocab
                            )


                    # Network forward step
                    logits = net(
                            src=batch["src"],
                            src_mask=batch["src_mask"],
                            tgt=batch["tgt"],
                            tgt_mask=batch["tgt_mask"],
                            src_lang=self.SRC_TEXT.vocab.stoi["<" + config.SRC_LAN.upper() + ">"],
                            tgt_lang=self.TGT_TEXT.vocab.stoi["<" + config.TGT_LAN.upper()  + ">"]
                            )
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
                    
                    del logits
                    del loss
                    del nll_loss
                    del loss_inputs
                    del batch
                    #torch.cuda.empty_cache()

            # compute perplexity and prediction accuracy
            scores = {}
            scores['ppl'] = np.exp(xe_loss / n_words)
            scores['acc'] = 100. * n_valid / n_words
            ppl_info = '{}-{} validation ppl:{} '.format(config.SRC_LAN, config.TGT_LAN, scores['ppl'])
            acc_info = '{}-{} validation accuracy:{} '.format(config.SRC_LAN, config.TGT_LAN, scores['acc'])
            logger.info(ppl_info + '|' + acc_info)
            
            return scores
        
        elif MODE == "MASS":

            n_words = 0
            xe_loss = 0
            n_valid = 0
            torch.cuda.empty_cache()
            scores = {}
            
            with torch.no_grad():
                net = self.net.eval()
                self.criterion.eval()
                if config.multi_gpu:
                    net = self.net.module
                
                for direction, valid_iter in self.iterators["valid"].items():
                    src_lan, tgt_lan = direction.split('-')
                    valid_iter.tokens_per_batch = -1
                    valid_iter.batch_size = 32
                    data_iter = iter(valid_iter.get_iterator(True, True))
                    
                    for i_batch, raw_batch in enumerate(data_iter):
                        # generate batch
                        batch = get_batch(
                                raw_batch.src, raw_batch.tgt,
                                self.TOTAL_TEXT.vocab, self.TOTAL_TEXT.vocab
                                )

                        # Network forward step
                        logits = net(
                                src=batch["src"],
                                src_mask=batch["src_mask"],
                                tgt=batch["tgt"],
                                tgt_mask=batch["tgt_mask"],
                                src_lang=self.TOTAL_TEXT.vocab.stoi["<" + src_lan.upper() + ">"],
                                tgt_lang=self.TOTAL_TEXT.vocab.stoi["<" + tgt_lan.upper() + ">"],
                                decoder_lang_id=config.LANG2IDS[tgt_lan.upper()]
                                )

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
                        n_valid += (logits.max(-1)[1] == batch["target"]).sum().long().item()

                        del batch, logits, loss, loss_inputs, nll_loss
                        torch.cuda.empty_cache()
                        
                    # compute perplexity and prediction accuracy
                    scores[direction.replace('-', '_') + '_ppl'] = np.exp(xe_loss / n_words)
                    scores[direction.replace('-', '_') + '_acc'] = 100. * n_valid / n_words
                    ppl_info = '{}-{} validation ppl:{} '.format(src_lan, tgt_lan, scores[direction.replace('-', '_') + '_ppl'])
                    acc_info = '{}-{} validation accuracy:{} '.format(src_lan, tgt_lan, scores[direction.replace('-', '_') + '_acc'])
                    logger.info(ppl_info + '|' + acc_info)
            
            return scores


if __name__ == "__main__":
    from dataset import MultiLingualDataset, ParallelDataset, Dataset, Text, Batch, Vocab 
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0,
                        help="Multi-GPU - Local rank"
            )
    parser.add_argument("--continue_path", type=str, default=None,
            help="Where to reload checkpoint"
            )
    parser.add_argument("--dump_path", type=str, default=None,
            help="Where to store checkpoints"
            )
    parser.add_argument('--data_bin', default=None, type=str,
            help="Path to store binarized data"
            )
    parser.add_argument('--epoch_size', default=None, type=int,
            help="Maximum train epochs"
            )
    params = parser.parse_args()
    if params.local_rank != -1:
        torch.cuda.set_device(params.local_rank)
        torch.distributed.init_process_group(backend="nccl",  init_method='env://')
    trainer = Enc_Dec_Trainer(params)

    data_iter = iter(
            trainer.iterators["train"].get_iterator(
                shuffle=True, group_by_size=True,
                )
            )
    
    vocab = trainer.TOTAL_TEXT.vocab 
    '''
    raw_batch = next(data_iter)
    mask = ( raw_batch['en'] != vocab.stoi[config.PAD] )
    bsz = mask.size(0)
    batch = trainer.restricted_mask_sent(
            raw_batch['en'], mask.sum(-1).view(bsz).long().cpu().numpy().tolist(), config.span_len
            )
    idx = np.random.randint(bsz, size=(1))[0]
    src = batch['src'][idx].cpu().numpy().tolist()
    tgt = batch['tgt'][idx].cpu().numpy().tolist()
    target = batch['target'][idx].cpu().numpy().tolist()
    tgt_pos = batch['tgt_pos'][idx].cpu().numpy().tolist()
    assert len(target) == len(tgt)
    assert len(tgt) == len(tgt_pos)

    print("SRC:", ' '.join([vocab.itos[idx] for idx in src]).replace("<pad>", ""))
    print("TGT:", ' '.join([vocab.itos[idx] for idx in tgt]).replace("<pad>", ""))
    print("TARGET:", ' '.join([vocab.itos[idx] for idx in target]).replace("<pad>", ""))
    print(tgt_pos)
    '''

    for direction, valid_iter in trainer.iterators["valid"].items():
        src_lan, tgt_lan = direction.split('-')
        valid_iter.tokens_per_batch = -1
        valid_iter.batch_size = 32
        data_iter = iter(valid_iter.get_iterator(True, True))
        print(direction)
        src_lang=vocab.stoi["<" + src_lan.upper() + ">"]
        tgt_lang=vocab.stoi["<" + tgt_lan.upper() + ">"]
        print(src_lan, src_lang)
        print(tgt_lan, tgt_lang)
        
        for i_batch, raw_batch in enumerate(data_iter):
            # generate batch
            src, tgt = raw_batch.src, raw_batch.tgt
            bsz = src.size(0)
            idx = np.random.randint(bsz, size=(1))[0]
            src = src[idx].cpu().numpy().tolist()
            tgt = tgt[idx].cpu().numpy().tolist()
            print("{}:".format(src_lan), ' '.join([vocab.itos[idx] for idx in src]).replace("<pad>", ""))
            print("{}:".format(tgt_lan), ' '.join([vocab.itos[idx] for idx in tgt]).replace("<pad>", ""))
            break
