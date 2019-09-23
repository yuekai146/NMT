ACTIVE=./active_data
ACTIVE_OUT=$ACTIVE/test_active.out_
U=$ACTIVE/unlabeled_
L=$ACTIVE/labeled_
ORACLE=$ACTIVE/oracle_
ACTIVE_FUNC=random
START_ROUND=2
N_ROUNDS=11
NGPUS=8
TOK_BUDGET=1600000
SRC=de
TGT=en


SCRIPTS=/data/mosesdecoder/scripts
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
FAIRSEQ_PATH=/data/fairseq
BPEROOT=/data/fastBPE

ORIG_SRC=../../data/de-en/wmt17_de_en/train.de
ORIG_TGT=../../data/de-en/wmt17_de_en/train.en
TEST_INPUT=../../data/de-en/wmt17_de_en/test.de
TEST_REF=../../data/de-en/wmt17_de_en/test.en

function main () {
	# Initialize labeled and unlabeled dataset
	mkdir -p $ACTIVE
	cp $ORIG_SRC ${U}0
	cp $ORIG_TGT ${ORACLE}0
	touch ${L}0.$SRC ${L}0.$TGT

	for i in $( seq $START_ROUND $N_ROUNDS )
	do	
		# Do active learning
		cd active_data
		num_U=$(cat unlabeled_${i} | wc -l)
		num_chunk=$(($num_U / $NGPUS + 1))
		split -l $num_chunk unlabeled_$i unlabeled_${i}_ -da 1
		split -l $num_chunk oracle_$i oracle_${i}_ -da 1
		cd ..	

		for j in $( seq 0 $((NGPUS - 1)) )
		do
			echo "CUDA_VISIBLE_DEVICES=$j python3 active.py score -a $ACTIVE_FUNC \
				-i ${U}${i}_$j -ref ${ORACLE}${i}_$j \
				-ckpt checkpoints/$((i+1))/checkpoint_best_ppl.pth \
				--max_batch_size 0 \
				--tokens_per_batch 16384 > test_active.out_${i}_${j}" >> parallel_active.sh
		done
		parallel -j $NGPUS < parallel_active.sh
		rm parallel_active.sh
		mv test_active.out_${i}_? active_data/
		cat active_data/test_active.out_${i}_? >> active_data/test_active.out_$i
		python3 active.py modify -U $U$i -L $L$i.$SRC,$L$i.$TGT --oracle $ORACLE$i -tb $TOK_BUDGET \
			-OU $U$((i+1)) -OL $L$((i+1)).$SRC,$L$((i+1)).$TGT \
			-OO $ORACLE$((i+1)) -AO $ACTIVE_OUT$i 
		cd active_data
		rm test_active.out*
		rm *_${i}_?
		cd ..

		# Train network on new labeled dataset
		export NGPUS=8
		rm -rf data_bin
		python3 dataset.py --store --SRC_RAW_TRAIN_PATH $L$((i+1)).$SRC \
			--TGT_RAW_TRAIN_PATH $L$((i+1)).$TGT 
		if [ $i -eq 0 ]; then
			python3 -m torch.distributed.launch --nproc_per_node=$NGPUS train.py \
				--raw_src $L$((i+1)).$SRC \
				--raw_tgt $L$((i+1)).$TGT \
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

		python3 $FAIRSEQ_PATH/score.py --sys checkpoints/$((i+1))/generate.sys --ref checkpoints/$((i+1))/generate.ref > checkpoints/$((i+1))/bleu.txt
		
	
	done
}

main
