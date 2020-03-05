ACTIVE=./active_data
ACTIVE_OUT=$ACTIVE/test_active.out_
U=$ACTIVE/unlabeled_
L=$ACTIVE/labeled_
ORACLE=$ACTIVE/oracle_
N_ROUNDS=11
NGPUS=8
TOK_BUDGET=1800000
SRC=ru
TGT=en


SCRIPTS=/data/mosesdecoder/scripts
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
FAIRSEQ_PATH=/data/fairseq
BPEROOT=/data/fastBPE

INIT_SRC=../base.wmt14.ru-en.active_base/result/random/active_data/labeled_1.$SRC
INIT_TGT=../base.wmt14.ru-en.active_base/result/random/active_data/labeled_1.$TGT
INIT_UNLABELED=../base.wmt14.ru-en.active_base/result/random/active_data/unlabeled_1
INIT_ORACLE=../base.wmt14.ru-en.active_base/result/random/active_data/oracle_1
INIT_CKPT=../base.wmt14.ru-en.active_base/result/random/checkpoints/1
TEST_INPUT=/data/ru-en/test.ru
TEST_REF=/data/ru-en/test.en

function Active_NMT () {
	local ACTIVE_FUNC=$1
	local START_ROUND=$2
	# Initialize labeled and unlabeled dataset
	mkdir -p $ACTIVE
	cp $INIT_SRC ${L}1.$SRC
	cp $INIT_TGT ${L}1.$TGT
	cp $INIT_UNLABELED ${U}1
	cp $INIT_ORACLE ${ORACLE}1

	for i in $( seq $START_ROUND $N_ROUNDS )
	do	

		# Train network on new labeled dataset
		export NGPUS=8
		rm -rf data_bin
		python3 dataset.py --store \
			--SRC_RAW_TRAIN_PATH $L$((i+1)).$SRC \
			--TGT_RAW_TRAIN_PATH $L$((i+1)).$TGT \
			--SRC_VOCAB_PATH /data/ru-en/vocab.$SRC \
			--TGT_VOCAB_PATH /data/ru-en/vocab.$TGT 
		if [ $i -eq 0 ]; then
			python3 -m torch.distributed.launch --nproc_per_node=$NGPUS train.py \
				--raw_src $L$((i+1)).$SRC \
				--raw_tgt $L$((i+1)).$TGT \
				--decrease_counts_max 25 \
				--dump_path checkpoints/$((i+1))/
		else
			continue_path=checkpoints/$i/checkpoint_best_ppl.pth
			python3 -m torch.distributed.launch --nproc_per_node=$NGPUS train.py \
				--raw_src $L$((i+1)).$SRC \
				--raw_tgt $L$((i+1)).$TGT \
				--continue_path $continue_path \
				--dump_path checkpoints/$((i+1))
		fi
		rm -rf checkpoints/$((i+1))/checkpoint_?.pth
		rm -rf checkpoints/$((i+1))/checkpoint_??.pth
		rm -rf checkpoints/$((i+1))/checkpoint_???.pth

		python3 translate.py -ckpt checkpoints/$((i+1))/checkpoint_best_ppl.pth \
			-text $TEST_INPUT -ref_text $TEST_REF \
			--max_batch_size 0 --tokens_per_batch 2000 -k 5 -max_len 200 > checkpoints/$((i+1))/total.out

		cat checkpoints/$((i+1))/total.out | grep ^H | cut -d " " -f3- > checkpoints/$((i+1))/sys.out
		cat checkpoints/$((i+1))/total.out | grep ^T | cut -d " " -f3- > checkpoints/$((i+1))/ref.out

		cat checkpoints/$((i+1))/sys.out | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > checkpoints/$((i+1))/generate.sys
		cat checkpoints/$((i+1))/ref.out | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > checkpoints/$((i+1))/generate.ref

		python3 $FAIRSEQ_PATH/score.py --sys checkpoints/$((i+1))/generate.sys \
			--ref checkpoints/$((i+1))/generate.ref > checkpoints/$((i+1))/bleu.txt
		
		# Do active learning
		if [ $i -ne $N_ROUNDS ]; then
			cd active_data
			num_U=$(cat unlabeled_$((i+1)) | wc -l)
			num_chunk=$(($num_U / $NGPUS + 1))
			split -l $num_chunk unlabeled_$((i+1)) unlabeled_$((i+1))_ -da 1
			split -l $num_chunk oracle_$((i+1)) oracle_$((i+1))_ -da 1
			cd ..	

			for j in $( seq 0 $((NGPUS - 1)) )
			do
				echo "CUDA_VISIBLE_DEVICES=$j python3 active.py score -a $ACTIVE_FUNC \
					-i ${U}$((i+1))_$j -ref ${ORACLE}$((i+1))_$j \
					-ckpt checkpoints/$((i+1))/checkpoint_best_ppl.pth \
					--max_batch_size 0 \
					--tokens_per_batch 16384 > test_active.out_$((i+1))_${j}" >> parallel_active.sh
			done
			parallel -j $NGPUS < parallel_active.sh
			rm parallel_active.sh
			mv test_active.out_$((i+1))_? active_data/
			cat active_data/test_active.out_$((i+1))_? >> active_data/test_active.out_$((i+1))
			python3 active.py modify -U $U$((i+1)) \
				-L $L$((i+1)).$SRC,$L$((i+1)).$TGT \
				--oracle $ORACLE$((i+1)) -tb $TOK_BUDGET \
				-OU $U$((i+2)) -OL $L$((i+2)).$SRC,$L$((i+2)).$TGT \
				-OO $ORACLE$((i+2)) -AO $ACTIVE_OUT$((i+1)) 
			cd active_data
			rm test_active.out*
			rm *_$((i+1))_?
			cd ..
		fi
		
	
	done
}


function main () {
	mkdir -p ./result
	for ACTIVE_FUNC in lc margin te tte; do
		Active_NMT $ACTIVE_FUNC 0
		mkdir -p ./result/$ACTIVE_FUNC
		mv checkpoints active_data ./result/$ACTIVE_FUNC/
		rm -rf data_bin
	done
}


main
