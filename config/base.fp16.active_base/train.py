from common import config
from trainer import Enc_Dec_Trainer
from utils import create_logger

import argparse
import os
import pickle
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Multi-GPU - Local rank"
            )
    parser.add_argument("--raw_src", type=str, default=None
            help="Tokenized source train file"
            )
    parser.add_argument("--raw_tgt", type=str, default=None
            help="Tokenized target train file"
            )
    parser.add_argument("--continue_path", type=str, default=None,
            help="Where to reload checkpoint"
            )
    parser.add_argument("--dump_path", type=str, default=None,
            help="Where to store checkpoints"
            )
    params = parser.parse_args()

    if args.raw_src is not None:
        config.SRC_RAW_TRAIN_PATH = args.raw_src
    if args.raw_tgt is not None:
        config.TGT_RAW_TRAIN_PATH = args.raw_tgt
    if args.continue_path is not None:
        config.continue_path = args.continue_path
    if args.dump_path is not None:
        config.dump_path = args.dump_path

    # Initialize distributed training
    if params.local_rank != -1:
        torch.cuda.set_device(params.local_rank)
        torch.distributed.init_process_group(backend="nccl",  init_method='env://')
    trainer = Enc_Dec_Trainer(params)

    # Check whether dump_path exists, if not create one
    if os.path.exists(config.dump_path) == False:
        os.makedirs(config.dump_path)

    # Save config in dump_path
    f = open(os.path.join(config.dump_path, "config.pkl"), 'wb')
    pickle.dump(config, f)
    f.close()

    # Create logger for each process
    logger = create_logger(
            os.path.join(config.dump_path, 'train.log'),
            rank=getattr(params, 'local_rank', 0)
            )

    # Start epoch training
    for i_epoch in range(trainer.epoch_size):
        if trainer.epoch > trainer.epoch_size:
            break

        if config.multi_gpu == False or int(os.environ["NGPUS"]) == 1:
            # Single GPU, do not need to split dataset
            data_iter = iter(trainer.iterators["train"].get_iterator(True, True))
        else:
            if params.local_rank == 0:
                if os.path.exists(config.data_bin) == False:
                    os.makedirs(config.data_bin)

                # Split dataset into NGPUS subsets, with the same number of batches
                # Store NGPUS subsets in config.data_bin
                subset_batches = trainer.iterators["train"].get_batch_ids(
                        shuffle=True, group_by_size=True,
                        num_subsets=int(os.environ["NGPUS"])
                        )
                
                for i_sub in range(len(subset_batches)):
                    f = open(os.path.join(config.data_bin, "batches_" + str(i_sub)), 'wb')
                    pickle.dump(subset_batches[i_sub], f)
                    f.close()

            torch.distributed.barrier()
            # Each process reads its own subset 
            f = open(os.path.join(config.data_bin, "batches_" + str(params.local_rank)), 'rb')
            subset_batches = pickle.load(f)
            f.close()
            data_iter = iter(trainer.iterators["train"].get_batches_iterator(subset_batches))
            num_train = sum([len(b) for b in subset_batches])
            trainer.num_train = num_train

        for i_batch, raw_batch in enumerate(data_iter):
            try:
                trainer.train_step(raw_batch)
                trainer.iter()
            except RuntimeError:
                continue
        
        scores = trainer.valid_step()
        trainer.save_best_model(scores)
        trainer.save_periodic()
        trainer.end_epoch(scores)
        torch.distributed.barrier()


if __name__ == "__main__":
    main()
