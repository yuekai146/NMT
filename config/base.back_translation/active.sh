# back translation not finished

SRC=de
TGT=en


SCRIPTS=/data/mosesdecoder/scripts
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
FAIRSEQ_PATH=/data/fairseq
BPEROOT=/data/fastBPE

INIT_SRC=../../data/de-en/wmt17_de_en/init/labeled_1.$SRC
INIT_TGT=../../data/de-en/wmt17_de_en/init/labeled_1.$TGT
INIT_UNLABELED=../../data/de-en/wmt17_de_en/init/unlabeled_1
INIT_ORACLE=../../data/de-en/wmt17_de_en/init/oracle_1
TEST_INPUT=../../data/de-en/wmt17_de_en/test.de
TEST_REF=../../data/de-en/wmt17_de_en/test.en

function supervised () {
	# Initialize labeled and unlabeled dataset
	ACTIVE=./active_data
	ACTIVE_OUT=$ACTIVE/test_active.out_
	U=$ACTIVE/unlabeled_
	L=$ACTIVE/labeled_
	ORACLE=$ACTIVE/oracle_
	ACTIVE_FUNC=lc
	N_ROUNDS=11
	START_ROUND=0
	NGPUS=8
	TOK_BUDGET=1600000
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


function Init_Active_BT () {
	local SRC=$1
	local TGT=$2

	ACTIVE=active_data
	ACTIVE_SRC2TGT=$ACTIVE/$SRC-$TGT
	ACTIVE_TGT2SRC=$ACTIVE/$TGT-$SRC

	mkdir -p $ACTIVE $ACTIVE_SRC2TGT $ACTIVE_TGT2SRC data_bin
	cp $INIT_SRC $ACTIVE_SRC2TGT/labeled_1.$SRC
	cp $INIT_TGT $ACTIVE_SRC2TGT/labeled_1.$TGT
	cp $INIT_UNLABELED $ACTIVE_SRC2TGT/unlabeled_1
	cp $INIT_ORACLE $ACTIVE_SRC2TGT/oracle_1
}


function Train_Model () {
	local round=$1
	local SRC=$2
	local TGT=$3
	
	export NGPUS=8
	rm -rf data_bin/$SRC-$TGT
	SRC_RAW_TRAIN_PATH=active_data/$SRC-$TGT/labeled_${round}.$SRC
	TGT_RAW_TRAIN_PATH=active_data/$SRC-$TGT/labeled_${round}.$TGT
	SRC_RAW_VALID_PATH=../../data/de-en/wmt17_de_en/valid.$SRC
	TGT_RAW_VALID_PATH=../../data/de-en/wmt17_de_en/valid.$TGT
	python3 dataset.py --store \
		--SRC_RAW_TRAIN_PATH $SRC_RAW_TRAIN_PATH \
		--TGT_RAW_TRAIN_PATH $TGT_RAW_TRAIN_PATH \
		--SRC_RAW_VALID_PATH $SRC_RAW_VALID_PATH \
		--TGT_RAW_VALID_PATH $TGT_RAW_VALID_PATH \
		--data_bin data_bin/$SRC-$TGT/
	if [ $round -eq 1 ]; then
		python3 -m torch.distributed.launch --nproc_per_node=$NGPUS train.py \
			--data_bin data_bin/$SRC-$TGT/ \
			--dump_path checkpoints/$SRC-$TGT/$round/
	else
		continue_path=checkpoints/$SRC-$TGT/$((round-1))/checkpoint_best_ppl.pth
		python3 -m torch.distributed.launch --nproc_per_node=$NGPUS train.py \
			--data_bin data_bin/$SRC-$TGT/ \
			--continue_path $continue_path \
			--dump_path checkpoints/$SRC-$TGT/$round/
	fi
	rm -rf checkpoints/$SRC-$TGT/$round/checkpoint_?.pth
	rm -rf checkpoints/$SRC-$TGT/$round/checkpoint_??.pth
	rm -rf checkpoints/$SRC-$TGT/$round/checkpoint_???.pth
}


function Test_Model () {
	local i=$1
	local SRC=$2
	local TGT=$3

	python3 translate.py -ckpt checkpoints/$SRC-$TGT/$i/checkpoint_best_ppl.pth \
		-text $TEST_SRC -ref_text $TEST_TGT \
		--max_batch_size 0 --tokens_per_batch 2000 -k 5 -max_len 200 > checkpoints/$SRC-$TGT/$i/total.out

	cat checkpoints/$SRC-$TGT/$i/total.out | grep ^H | cut -d " " -f3- > checkpoints/$SRC-$TGT/$i/sys.out
	cat checkpoints/$SRC-$TGT/$i/total.out | grep ^T | cut -d " " -f3- > checkpoints/$SRC-$TGT/$i/ref.out

	cat checkpoints/$SRC-$TGT/$i/sys.out | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > checkpoints/$SRC-$TGT/$i/generate.sys
	cat checkpoints/$SRC-$TGT/$i/ref.out | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > checkpoints/$SRC-$TGT/$i/generate.ref

	python3 $FAIRSEQ_PATH/score.py --sys checkpoints/$SRC-$TGT/$i/generate.sys \
		--ref checkpoints/$SRC-$TGT/$i/generate.ref > checkpoints/$SRC-$TGT/$i/bleu.txt

}


function Active_Learn () {
	local i=$1
	local SRC=$2
	local TGT=$3
	local ACTIVE_FUNC=$4
	local TOK_BUDGET=$5
	local REV=$6
	export NGPUS=8
	
	# Split unlabeled data into NGPUS chunks
	cd active_data/$SRC-$TGT
	num_U=$(cat unlabeled_$i | wc -l)
	num_chunk=$(($num_U / $NGPUS + 1))
	split -l $num_chunk unlabeled_$i unlabeled_${i}_ -da 1
	split -l $num_chunk oracle_$i oracle_${i}_ -da 1
	cd -	
	
	# Get active function score
	for j in $( seq 0 $((NGPUS - 1)) )
	do
		echo "CUDA_VISIBLE_DEVICES=$j python3 active.py score \
			-a $ACTIVE_FUNC \
			-i active_data/$SRC-$TGT/unlabeled_${i}_$j \
			-ref active_data/$SRC-$TGT/oracle_${i}_$j \
			-ckpt checkpoints/$SRC-$TGT/$i/checkpoint_best_ppl.pth \
			--max_batch_size 0 \
			--tokens_per_batch 16384 > test_active.$SRC-$TGT.out_${i}_${j}" >> parallel_active.sh
	done
	parallel -j $NGPUS < parallel_active.sh
	rm parallel_active.sh
	mv test_active.$SRC-$TGT.out_${i}_? active_data/$SRC-$TGT/
	cd active_data/$SRC-$TGT/
	cat test_active.$SRC-$TGT.out_${i}_? >> test_active.$SRC-$TGT.out_${i}
	cd -

	# Modify all data
	U=active_data/$SRC-$TGT/unlabeled_$i
	L=active_data/$SRC-$TGT/labeled_$i.$SRC,active_data/$SRC-$TGT/labeled_$i.$TGT
	oracle=active_data/$SRC-$TGT/oracle_$i

	if [ "$REV" == "True" ]; then
		OU=active_data/$TGT-$SRC/oracle_$i
		OL=active_data/$TGT-$SRC/labeled_$i.$SRC,active_data/$TGT-$SRC/labeled_$i.$TGT
		OO=active_data/$TGT-$SRC/unlabeled_$i
		AO=active_data/$SRC-$TGT/test_active.$SRC-$TGT.out_${i}
		rnq=active_data/$TGT-$SRC/rnq_$i.$TGT-$SRC.$SRC,active_data/$TGT-$SRC/rnq_$i.$TGT-$SRC.$TGT
		onq=active_data/$SRC-$TGT/rnq_$((i+1)).$SRC-$TGT.$SRC,active_data/$SRC-$TGT/rnq_$((i+1)).$SRC-$TGT.$TGT
	else
		OU=active_data/$TGT-$SRC/oracle_$i
		OL=active_data/$TGT-$SRC/labeled_$i.$SRC,active_data/$TGT-$SRC/labeled_$i.$TGT
		OO=active_data/$TGT-$SRC/unlabeled_$i
		AO=active_data/$SRC-$TGT/test_active.$SRC-$TGT.out_${i}
		rnq=active_data/$TGT-$SRC/rnq_$((i-1)).$TGT-$SRC.$SRC,active_data/$TGT-$SRC/rnq_$((i-1)).$TGT-$SRC.$TGT
		onq=active_data/$SRC-$TGT/rnq_$i.$SRC-$TGT.$SRC,active_data/$SRC-$TGT/rnq_$i.$SRC-$TGT.$TGT
	fi
	python3 active.py modify -U $U \
		-L $L \
		--oracle $oracle \
		-tb $TOK_BUDGET \
		-OU $OU \
		-OL $OL \
		-OO $OO \
		-AO $AO \
		-bt \
	       	-rnq $rnq \
		-onq $onq
	cd active_data/$SRC-$TGT
	#rm test_active.$SRC-$TGT.out*
	rm -rf *_${i}_?
	cd -
}


function BT () {
	# Initialize labeled and unlabeled dataset
	N_ROUNDS=3
	START_ROUND=1
	NGPUS=8
	TOK_BUDGET=533333
	ACTIVE_FUNC=random

	Init_Active_BT de en

	for i in $( seq $START_ROUND $N_ROUNDS )
	do	
		# Train source to target network on new labeled dataset
		Train_Model $i de en
		
		# Do active learning
		Active_Learn $i de en $ACTIVE_FUNC $TOK_BUDGET False
		
		# Train target to source network on new labeled dataset	
		Train_Model $i en de
		
		# Do active learning
		if [ $i -ne $N_ROUNDS ]; then
			Active_Learn $i en de $ACTIVE_FUNC $TOK_BUDGET True
		fi
		
	done
}

 #Active_Learn 1 de en random 533333 False
 Train_Model 1 en de
