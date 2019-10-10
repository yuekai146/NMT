export NGPUS=8
rlaunch --cpu=16 --memory=32768 --gpu=8 -- python3 -m torch.distributed.launch --nproc_per_node=$NGPUS train.py 
