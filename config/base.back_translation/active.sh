# back translation not finished

SRC=de
TGT=en


SCRIPTS=/data/mosesdecoder/scripts
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
FAIRSEQ_PATH=/data/fairseq
BPEROOT=/data/fastBPE

TEST_INPUT=../../data/de-en/wmt17_de_en/test.de
TEST_REF=../../data/de-en/wmt17_de_en/test.en


function Init_Active_BT () {
	local SRC=$1
	local TGT=$2

	INIT_SRC=../../data/de-en/wmt17_de_en/init/labeled_1.$SRC
	INIT_TGT=../../data/de-en/wmt17_de_en/init/labeled_1.$TGT
	INIT_UNLABELED=../../data/de-en/wmt17_de_en/init/unlabeled_1.$SRC
	INIT_ORACLE=../../data/de-en/wmt17_de_en/init/unlabeled_1.$TGT

	ACTIVE=active_data
	ACTIVE_SRC2TGT=$ACTIVE/$SRC-$TGT
	ACTIVE_TGT2SRC=$ACTIVE/$TGT-$SRC

	mkdir -p $ACTIVE $ACTIVE_SRC2TGT $ACTIVE_TGT2SRC data_bin
	cp $INIT_UNLABELED $ACTIVE_SRC2TGT/unlabeled_1
	cp $INIT_ORACLE $ACTIVE_SRC2TGT/oracle_1
	cp $INIT_SRC $ACTIVE_SRC2TGT/train_1.$SRC
	cp $INIT_TGT $ACTIVE_SRC2TGT/train_1.$TGT
	cp $INIT_SRC $ACTIVE_SRC2TGT/labeled_1.$SRC
	cp $INIT_TGT $ACTIVE_SRC2TGT/labeled_1.$TGT
}


function Train_Model () {
	local round=$1
	local SRC=$2
	local TGT=$3
	
	export NGPUS=8
	rm -rf data_bin/$SRC-$TGT
	SRC_RAW_TRAIN_PATH=active_data/$SRC-$TGT/train_${round}.$SRC
	TGT_RAW_TRAIN_PATH=active_data/$SRC-$TGT/train_${round}.$TGT
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

	TEST_SRC=../../data/de-en/wmt17_de_en/test.$SRC
	TEST_TGT=../../data/de-en/wmt17_de_en/test.$TGT

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
		OU=active_data/$TGT-$SRC/oracle_$((i+1))
		OL=active_data/$TGT-$SRC/labeled_$((i+1)).$SRC,active_data/$TGT-$SRC/labeled_$((i+1)).$TGT
		OO=active_data/$TGT-$SRC/unlabeled_$((i+1))
		AO=active_data/$SRC-$TGT/test_active.$SRC-$TGT.out_${i}
		rnq=active_data/$TGT-$SRC/rnq_$i.$TGT-$SRC.$SRC,active_data/$TGT-$SRC/rnq_$i.$TGT-$SRC.$TGT
		onq=active_data/$SRC-$TGT/rnq_$i.$SRC-$TGT.$SRC,active_data/$SRC-$TGT/rnq_$i.$SRC-$TGT.$TGT
		OT=active_data/$TGT-$SRC/train_$((i+1)).$SRC,active_data/$TGT-$SRC/train_$((i+1)).$TGT
	else
		OU=active_data/$TGT-$SRC/oracle_$i
		OL=active_data/$TGT-$SRC/labeled_$i.$SRC,active_data/$TGT-$SRC/labeled_$i.$TGT
		OO=active_data/$TGT-$SRC/unlabeled_$i
		AO=active_data/$SRC-$TGT/test_active.$SRC-$TGT.out_${i}
		rnq=active_data/$TGT-$SRC/rnq_$((i-1)).$TGT-$SRC.$SRC,active_data/$TGT-$SRC/rnq_$((i-1)).$TGT-$SRC.$TGT
		onq=active_data/$SRC-$TGT/rnq_$i.$SRC-$TGT.$SRC,active_data/$SRC-$TGT/rnq_$i.$SRC-$TGT.$TGT
		OT=active_data/$TGT-$SRC/train_$i.$SRC,active_data/$TGT-$SRC/train_$i.$TGT
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
		-onq $onq \
		-OT $OT
	cd active_data/$SRC-$TGT
	rm test_active.$SRC-$TGT.out*
	rm -rf *_${i}_?
	cd -
}


function BT () {
	# Initialize labeled and unlabeled dataset
	local ACTIVE_FUNC=${1:-random}
	local TOK_BUDGET=${2:-800000}
	local N_ROUNDS=${3:-12}
	local START_ROUND=${4:-1}
	local LAN1=${5:-en}
	local LAN2=${6:-de}
	NGPUS=8

	Init_Active_BT $LAN1 $LAN2

	for i in $( seq $START_ROUND $N_ROUNDS )
	do	
		# Train source to target network on new labeled dataset
		Train_Model $i $LAN1 $LAN2

		# Test source to target network
		Test_Model $i $LAN1 $LAN2
		
		# Do active learning
		Active_Learn $i $LAN1 $LAN2 $ACTIVE_FUNC $TOK_BUDGET False
		
		# Train target to source network on new labeled dataset	
		Train_Model $i $LAN2 $LAN1

		# Test target to source betwork
		Test_Model $i $LAN2 $LAN1
		
		# Do active learning
		if [ $i -ne $N_ROUNDS ]; then
			Active_Learn $i $LAN2 $LAN1 $ACTIVE_FUNC $TOK_BUDGET True
		fi
		
	done
}


function main () {
	for ACTIVE_FUNC in random longest shortest lc margin te tte; do
		BT $ACTIVE_FUNC 
	done
}
Active_Learn 1 en de random 800000 False
Train_Model 1 de en
#main
