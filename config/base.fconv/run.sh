NGPUS=1
python -m torch.distributed.launch --nproc_per_node=$NGPUS train.py
