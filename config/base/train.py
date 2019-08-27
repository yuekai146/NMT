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
    params = parser.parse_args()

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
        data_iter = iter(trainer.iterators["train"].get_iterator(True, True))
        for i_batch, raw_batch in enumerate(data_iter):
            try:
                trainer.train_step(raw_batch)
                trainer.iter()
            except RuntimeError:
                continue
        print("Epoch {} finished!".format(i_epoch))
        scores = trainer.valid_step()
        trainer.save_best_model(scores)
        trainer.save_periodic()
        trainer.end_epoch()


if __name__ == "__main__":
    main()
