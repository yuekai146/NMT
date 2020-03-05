function Init_Train () {
	local src=$1
	local tgt=$2

	export NGPUS=8
	python3 dataset.py --store \
		--SRC_RAW_TRAIN_PATH /data/ru-en/init/labeled_1.$src \
		--TGT_RAW_TRAIN_PATH /data/ru-en/init/labeled_1.$tgt \
		--SRC_RAW_VALID_PATH /data/ru-en/init/valid.$src \
		--TGT_RAW_VALID_PATH /data/ru-en/init/valid.$tgt \
		--SRC_VOCAB_PATH /data/ru-en/vocab.$src \
		--TGT_VOCAB_PATH /data/ru-en/vocab.$tgt \
		--data_bin data_bin/$src-$tgt/

	python3 -m torch.distributed.launch --nproc_per_node=$NGPUS train.py \
			--data_bin data_bin/$src-$tgt/
}

function test() {
	local src=$1
	local tgt=$2
	FAIRSEQ_PATH=/data/fairseq
	python3 translate.py -ckpt checkpoints/checkpoint_best_ppl.pth \
		-text /data/ru-en/test.$src \
		-ref_text /data/ru-en/test.$tgt \
		--max_batch_size 0 \
		--tokens_per_batch 2000 \
		-max_len 200 > total.out

	cat total.out | grep ^H | cut -d " " -f2- > sys.out
	cat total.out | grep ^T | cut -d " " -f2- > ref.out

	cat sys.out | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > generate.sys
	cat ref.out | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > generate.ref

	python3 $FAIRSEQ_PATH/score.py --sys generate.sys --ref generate.ref > bleu.out
	mv *.out generate.* checkpoints
	mkdir -p BT_init_ckpt
	mv checkpoints BT_init_ckpt/$src-$tgt
}


function main () {
	Init_Train ru en
	test ru en
}

main
