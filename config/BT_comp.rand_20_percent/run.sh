function run () {
	
	local SRC=$1
	local TGT=$2
	
	FAIRSEQ_PATH=/data/fairseq
	export NGPUS=8
	rm -rf data_bin/$SRC-$TGT

	SRC_RAW_TRAIN_PATH=init/labeled_1.$SRC
	TGT_RAW_TRAIN_PATH=init/labeled_1.$TGT
	SRC_RAW_VALID_PATH=../../data/de-en/wmt17_de_en/valid.$SRC
	TGT_RAW_VALID_PATH=../../data/de-en/wmt17_de_en/valid.$TGT
	python3 dataset.py --store \
		--SRC_RAW_TRAIN_PATH $SRC_RAW_TRAIN_PATH \
		--TGT_RAW_TRAIN_PATH $TGT_RAW_TRAIN_PATH \
		--SRC_RAW_VALID_PATH $SRC_RAW_VALID_PATH \
		--TGT_RAW_VALID_PATH $TGT_RAW_VALID_PATH \
		--data_bin data_bin/$SRC-$TGT/

	python3 -m torch.distributed.launch --nproc_per_node=$NGPUS train.py \
		--data_bin data_bin/$SRC-$TGT/ \
		--dump_path checkpoints/$SRC-$TGT/

	rm -rf checkpoints/$SRC-$TGT/checkpoint_?.pth
	rm -rf checkpoints/$SRC-$TGT/checkpoint_??.pth
	rm -rf checkpoints/$SRC-$TGT/checkpoint_???.pth
		

	TEST_SRC=../../data/de-en/wmt17_de_en/test.$SRC
	TEST_TGT=../../data/de-en/wmt17_de_en/test.$TGT

	python3 translate.py -ckpt checkpoints/$SRC-$TGT/checkpoint_best_ppl.pth \
		-text $TEST_SRC -ref_text $TEST_TGT \
		--max_batch_size 0 --tokens_per_batch 2000 -k 5 -max_len 200 > checkpoints/$SRC-$TGT/total.out

	cat checkpoints/$SRC-$TGT/total.out | grep ^H | cut -d " " -f3- > checkpoints/$SRC-$TGT/sys.out
	cat checkpoints/$SRC-$TGT/total.out | grep ^T | cut -d " " -f3- > checkpoints/$SRC-$TGT/ref.out

	cat checkpoints/$SRC-$TGT/sys.out | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > checkpoints/$SRC-$TGT/generate.sys
	cat checkpoints/$SRC-$TGT/ref.out | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > checkpoints/$SRC-$TGT/generate.ref

	python3 $FAIRSEQ_PATH/score.py --sys checkpoints/$SRC-$TGT/generate.sys \
		--ref checkpoints/$SRC-$TGT/generate.ref > checkpoints/$SRC-$TGT/bleu.txt
}


run de en
run en de
