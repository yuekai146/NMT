export NGPUS=8
DATA_BASE=/data/NMT/data/de-en/news_crawl

if [ ! -d ./data_bin ]; then
	python3 dataset.py --store --DATA_PATH $DATA_BASE
fi
python3 -m torch.distributed.launch --nproc_per_node=$NGPUS train.py \
	--dump_path checkpoints/
