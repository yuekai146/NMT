export NGPUS=8

src=de
tgt=en

SRC_RAW_TRAIN_PATH=init/labeled_1.$src
TGT_RAW_TRAIN_PATH=init/labeled_1.$tgt
SRC_RAW_VALID_PATH=../../data/de-en/wmt17_de_en/valid.$src
TGT_RAW_VALID_PATH=../../data/de-en/wmt17_de_en/valid.$tgt
python3 dataset.py --store \
	--SRC_RAW_TRAIN_PATH $SRC_RAW_TRAIN_PATH \
	--TGT_RAW_TRAIN_PATH $TGT_RAW_TRAIN_PATH \
	--SRC_RAW_VALID_PATH $SRC_RAW_VALID_PATH \
	--TGT_RAW_VALID_PATH $TGT_RAW_VALID_PATH \
	--data_bin data_bin/
	
python3 -m torch.distributed.launch --nproc_per_node=$NGPUS train.py \
	--data_bin data_bin/ \
	--dump_path checkpoints/ \
	--continue_path ../base.transfer.BT/checkpoints/pivot_${src}2${tgt}.pth
