ACTIVE=./active_data
tmp=$ACTIVE/tmp
prep=$ACTIVE/train
U=$ACTIVE/unlabeled_
L=$ACTIVE/labeled_
ORACLE=$ACTIVE/oracle_
ACTIVE_FUNC=dden
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
	cp $ORIG_SRC ${U}0
	cp $ORIG_TGT ${ORACLE}0
	touch ${L}0.$SRC
	touch ${L}0.$TGT

	for i in $( seq 0 $N_ROUNDS )
	do	
		# Do active learning
		python3 active.py -U $U$i -L $L$i.$SRC,$L$i.$TGT --oracle $ORACLE$i \
			-OU $U$((i+1)) -OL $L$((i+1)).$SRC,$L$((i+1)).$TGT -OO $ORACLE$((i+1)) \
			-a $ACTIVE_FUNC -tb 279315

		# Apply BPE and filter by length
		$BPEROOT/fast applybpe $tmp/bpe.labeled_$((i+1)).$SRC $L$((i+1)).$SRC $BPE_CODE
		$BPEROOT/fast applybpe $tmp/bpe.labeled_$((i+1)).$TGT $L$((i+1)).$TGT $BPE_CODE
		perl $CLEAN -ratio 1.5 $tmp/bpe.labeled_$((i+1)) $SRC $TGT $prep/train_$((i+1)) 1 250

		# Train network on new labeled dataset
		export NGPUS=8
		if [ $i -eq 0 ]; then
			python3 -m torch.distributed.launch --nproc_per_node=$NGPUS train.py \
				--raw_src $prep/train_$((i+1)).$SRC \
				--raw_tgt $prep/train_$((i+1)).$TGT \
				--dump_path checkpoints/$((i+1))/
		else
			continue_path=checkpoints/$i/checkpoint_best_ppl.pth
			python3 -m torch.distributed.launch --nproc_per_node=$NGPUS train.py \
				--raw_src $prep/train_$((i+1)).$SRC \
				--raw_tgt $prep/train_$((i+1)).$TGT \
				--continue_path $continue_path \
				--dump_path checkpoints/$((i+1))/
		fi
		rm -rf checkpoints/$((i+1))/checkpoint_?.pth
		rm -rf checkpoints/$((i+1))/checkpoint_??.pth
		rm -rf checkpoints/$((i+1))/checkpoint_???.pth
	done
}

main
