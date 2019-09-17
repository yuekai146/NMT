ACTIVE=./active_data
ACTIVE_OUT=$ACTIVE/test_active.out_
tmp=$ACTIVE/tmp
prep=$ACTIVE/train
U=$ACTIVE/unlabeled_
L=$ACTIVE/labeled_
ORACLE=$ACTIVE/oracle_
ACTIVE_FUNC=margin
N_ROUNDS=11
NGPUS=8
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
		python3 active.py modify -U $U$i -L $L$i.$SRC,$L$i.$TGT --oracle $ORACLE$i -tb 279315 \
			-OU $U$((i+1)) -OL $L$((i+1)).$SRC,$L$((i+1)).$TGT \
			-OO $ORACLE$((i+1)) -AO $ACTIVE_OUT$i 
		cd active_data
		rm test_active.out*
		rm *_${i}_?
		cd ..
	
	done
}

main
