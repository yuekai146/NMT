export NGPUS=8
if [ ! -d data_bin ]; then
	python3 dataset.py --store
fi
python3 -m torch.distributed.launch --nproc_per_node=$NGPUS train.py 
