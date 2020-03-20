# back translation not finished

SCRIPTS=/data/mosesdecoder/scripts
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
FAIRSEQ_PATH=/data/fairseq
BPEROOT=/data/fastBPE


function Init_Active_BT () {
	local SRC=$1
	local TGT=$2

	local INIT_SRC=/data/ru-en/init/labeled_1.$SRC
	local INIT_TGT=/data/ru-en/init/labeled_1.$TGT
	local INIT_UNLABELED=/data/ru-en/init/unlabeled_1.$SRC
	local INIT_ORACLE=/data/ru-en/init/unlabeled_1.$TGT

	local ACTIVE=active_data
	local ACTIVE_SRC2TGT=$ACTIVE/$SRC-$TGT
	local ACTIVE_TGT2SRC=$ACTIVE/$TGT-$SRC

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
	local SRC_RAW_TRAIN_PATH=active_data/$SRC-$TGT/train.$SRC
	local TGT_RAW_TRAIN_PATH=active_data/$SRC-$TGT/train.$TGT
	local SRC_RAW_VALID_PATH=/data/ru-en/valid.$SRC
	local TGT_RAW_VALID_PATH=/data/ru-en/valid.$TGT
	python3 dataset.py --store \
		--SRC_RAW_TRAIN_PATH $SRC_RAW_TRAIN_PATH \
		--TGT_RAW_TRAIN_PATH $TGT_RAW_TRAIN_PATH \
		--SRC_RAW_VALID_PATH $SRC_RAW_VALID_PATH \
		--TGT_RAW_VALID_PATH $TGT_RAW_VALID_PATH \
		--SRC_VOCAB_PATH /data/ru-en/vocab.$SRC \
		--TGT_VOCAB_PATH /data/ru-en/vocab.$TGT \
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
			--epoch_size 1 \
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

	local TEST_SRC=/data/ru-en/test.$SRC
	local TEST_TGT=/data/ru-en/test.$TGT

	python3 translate.py -ckpt checkpoints/$SRC-$TGT/$i/checkpoint_best_ppl.pth \
		-text $TEST_SRC -ref_text $TEST_TGT \
		--max_batch_size 0 --tokens_per_batch 2000 -k 5 -max_len 200 > checkpoints/$SRC-$TGT/$i/total.out

	cat checkpoints/$SRC-$TGT/$i/total.out | grep ^H | cut -d " " -f3- > checkpoints/$SRC-$TGT/$i/sys.out
	cat checkpoints/$SRC-$TGT/$i/total.out | grep ^T | cut -d " " -f3- > checkpoints/$SRC-$TGT/$i/ref.out

	cat checkpoints/$SRC-$TGT/$i/sys.out | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > checkpoints/$SRC-$TGT/$i/generate.sys
	cat checkpoints/$SRC-$TGT/$i/ref.out | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > checkpoints/$SRC-$TGT/$i/generate.ref

	python3 $FAIRSEQ_PATH/score.py --sys checkpoints/$SRC-$TGT/$i/generate.sys \
		--ref checkpoints/$SRC-$TGT/$i/generate.ref > checkpoints/$SRC-$TGT/$i/bleu.out

}


function Active_Learn () {
	local i=$1
	local out_i=$2
	local SRC=$3
	local TGT=$4
	local ACTIVE_FUNC=$5
	local TOK_BUDGET=$6
	export NGPUS=8
	
	# Split unlabeled data into NGPUS chunks
	cd active_data/$SRC-$TGT
	local num_U=$(cat unlabeled_$i | wc -l)
	local num_chunk=$(($num_U / $NGPUS + 1))
	split -l $num_chunk unlabeled_$i unlabeled_${i}_ -da 1
	split -l $num_chunk oracle_$i oracle_${i}_ -da 1
	cd -	
	
	# Get active function score
	local j
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
	cat test_active.$SRC-$TGT.out_${i} | grep ^S | cut -d ' ' -f3- > ../$TGT-$SRC/train.$SRC
	cat test_active.$SRC-$TGT.out_${i} | grep ^H | cut -d ' ' -f3- > ../$TGT-$SRC/train.$TGT
	cat labeled_1.$SRC >> ../$TGT-$SRC/train.$SRC
	cat labeled_1.$TGT >> ../$TGT-$SRC/train.$TGT
	rm test_active.$SRC-$TGT.out*
	rm -rf *_${i}_?
	cd -
}


function BT () {
	# Initialize labeled and unlabeled dataset
	local ACTIVE_FUNC=${1:-random}
	local TOK_BUDGET=${2:-600000}
	local N_ROUNDS=${3:-8}
	local START_ROUND=${4:-1}
	local LAN1=${5:-en}
	local LAN2=${6:-ru}
	NGPUS=8

	if [ $START_ROUND -eq 1 ]; then
		Init_Active_BT $LAN1 $LAN2
		Init_Active_BT $LAN2 $LAN1
		cp -r /data/ru-en/BT_init_ckpt checkpoints	
		START_ROUND=2
	fi

	local i
	for i in $( seq $START_ROUND $N_ROUNDS )
	do	
		# Do active learning
		Active_Learn $((i-1)) $i $LAN1 $LAN2 $ACTIVE_FUNC $TOK_BUDGET
		
		# Train target to source network on new labeled dataset	
		Train_Model $i $LAN2 $LAN1

		# Test target to source betwork
		Test_Model $i $LAN2 $LAN1

		# Do active learning
		Active_Learn $i $i $LAN2 $LAN1 $ACTIVE_FUNC $TOK_BUDGET

		# Train source to target network on new labeled dataset
		Train_Model $i $LAN1 $LAN2

		# Test source to target network
		Test_Model $i $LAN1 $LAN2
	done
}


function main () {

	for ACTIVE_FUNC in random; do
		BT $ACTIVE_FUNC
		mkdir -p result/$ACTIVE_FUNC
		mv active_data checkpoints result/$ACTIVE_FUNC/
		rm -rf data_bin
	done
}


main
