ACTIVE=./active_data
tmp=$ACTIVE/tmp
prep=$ACTIVE/train
U=$ACTIVE/unlabeled_
L=$ACTIVE/labeled_
ORACLE=$ACTIVE/oracle_
ACTIVE_FUNC=te
N_ROUNDS=11
SRC=de
TGT=en

ORIG_SRC=../../data/de-en/iwslt14_de_en/orig.$SRC
ORIG_TGT=../../data/de-en/iwslt14_de_en/orig.$TGT

BPEROOT=/data/fastBPE
BPE_CODE=../../data/de-en/iwslt14_de_en/code

SCRIPTS=/data/mosesdecoder/scripts
CLEAN=$SCRIPTS/training/clean-corpus-n.perl

function main () {
	# Initialize labeled and unlabeled dataset
	mkdir -p $ACTIVE $tmp $prep

	for i in $( seq 0 $N_ROUNDS )
	do	
		# Train network on new labeled dataset
		export NGPUS=8
		if [ $i -eq 0 ]; then
			python3 -m torch.distributed.launch --nproc_per_node=$NGPUS train.py \
				--raw_src $L$i.$SRC \
				--raw_tgt $L$i.$TGT \
				--dump_path checkpoints/$((i+1))/
		else
			continue_path=checkpoints/$i/checkpoint_best_ppl.pth
			python3 -m torch.distributed.launch --nproc_per_node=$NGPUS train.py \
				--raw_src $L$i.$SRC \
				--raw_tgt $L$i.$TGT \
				--continue_path $continue_path \
				--dump_path checkpoints/$((i+1))
		fi
		rm -rf checkpoints/$((i+1))/checkpoint_?.pth
		rm -rf checkpoints/$((i+1))/checkpoint_??.pth
		rm -rf checkpoints/$((i+1))/checkpoint_???.pth
		
		# Do active learning
		python3 active.py -U $U$i \
			-L $L$i.$SRC,$L$i.$TGT \
			--oracle $ORACLE$i -tb 279315 \
			-OU $U$((i+1)) \
			-OL $L$((i+1)).$SRC,$L$((i+1)).$TGT \
			-OO $ORACLE$((i+1)) -a $ACTIVE_FUNC \
			-ckpt checkpoints/$((i+1))/checkpoint_best_ppl.pth \
			--max_batch_size 0 --tokens_per_batch 16384
	done
	
}

main
