FAIRSEQ_PATH=/data/fairseq
BPEROOT=/data/fastBPE


function Init_Active_BT () {
	local SRC=$1
	local TGT=$2

	INIT_SRC=../../data/defr-en/init/labeled_1.$SRC
	INIT_TGT=../../data/defr-en/init/labeled_1.$TGT
	INIT_UNLABELED=../../data/defr-en/init/unlabeled_1.$SRC
	INIT_ORACLE=../../data/defr-en/init/unlabeled_1.$TGT

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
	SRC_RAW_VALID_PATH=../../data/defr-en/valid.$SRC
	TGT_RAW_VALID_PATH=../../data/defr-en/valid.$TGT
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
	local TEST_DIR=$4

	TEST_SRC=~/$TEST_DIR/test.$SRC
	TEST_TGT=~/$TEST_DIR/test.$TGT

	python3 translate.py -ckpt checkpoints/$SRC-$TGT/$i/checkpoint_best_ppl.pth \
		-text $TEST_SRC -ref_text $TEST_TGT \
		--max_batch_size 0 --tokens_per_batch 2000 -k 5 -max_len 200 > checkpoints/$SRC-$TGT/$i/total.out

	cat checkpoints/$SRC-$TGT/$i/total.out | grep ^H | cut -d " " -f3- > checkpoints/$SRC-$TGT/$i/sys.out
	cat checkpoints/$SRC-$TGT/$i/total.out | grep ^T | cut -d " " -f3- > checkpoints/$SRC-$TGT/$i/ref.out

	cat checkpoints/$SRC-$TGT/$i/sys.out | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > checkpoints/$SRC-$TGT/$i/generate.sys
	cat checkpoints/$SRC-$TGT/$i/ref.out | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > checkpoints/$SRC-$TGT/$i/generate.ref

	python3 $FAIRSEQ_PATH/score.py --sys checkpoints/$SRC-$TGT/$i/generate.sys \
		--ref checkpoints/$SRC-$TGT/$i/generate.ref > checkpoints/$SRC-$TGT/$i/bleu.out

	cd checkpoints/$SRC-$TGT/$i
	mkdir $TEST_DIR
	mv *.out generate.* $TEST_DIR
	cd -

}


function Active_Learn () {
	local i=$1
	local out_i=$2
	local SRC=$3
	local TGT=$4
	local ACTIVE_FUNC=$5
	local TOK_BUDGET=$6
	export NGPUS=8
	
	# Get active function score
	echo "Calculating active function scores"
	python3 active.py score \
		-a $ACTIVE_FUNC \
		-i active_data/$SRC-$TGT/unlabeled_${i} \
		-lb active_data/$SRC-$TGT/labeled_${i} \
		-ref active_data/$SRC-$TGT/oracle_${i} \
		-ckpt checkpoints/$SRC-$TGT/$i/checkpoint_best_ppl.pth \
		--max_batch_size 0 \
		--tokens_per_batch 16384 > text_active.$SRC-$TGT.out_${i}

	mv text_active.$SRC-$TGT.out_${i} active_data/$SRC-$TGT/
	
	# Split into oracle, pseudo and others
	echo "Choose sentences to label and translate"
	python3 modify.py get \
		-AO active_data/$SRC-$TGT/text_active.$SRC-$TGT.out_${i} \
		-tb $TOK_BUDGET \
		-bttb $((i*TOK_BUDGET + 9 * TOK_BUDGET)) \
		-o active_data/$SRC-$TGT/active_tmp \
		-on $NGPUS
	
	# Translate pseudo sentences
	echo "Translating sentences"
	for j in $( seq 0 $((NGPUS-1)) ); do
		echo "CUDA_VISIBLE_DEVICES=$j python3 modify.py translate \
			-i active_data/$SRC-$TGT/active_tmp_pseudo_$j \
			-o active_data/$SRC-$TGT/active_tmp_bt_$j \
			--ckpt checkpoints/$SRC-$TGT/$i/checkpoint_best_ppl.pth \
			--max_len 200 \
			--gen_a 1.3 \
			--gen_b 5 \
			--max_batch_size 0 \
			--tokens_per_batch 16384" >> parallel_active.sh
	done
	parallel -j $NGPUS < parallel_active.sh
	rm parallel_active.sh
	
	# Concatenate all output files	
	cd active_data/$SRC-$TGT
	cat active_tmp_oracle >> test_active.$SRC-$TGT.out_$i
	for j in $( seq 0 $((NGPUS-1)) ); do
		cat active_tmp_bt_$j >> test_active.$SRC-$TGT.out_$i
	done
	cat active_tmp_others >> test_active.$SRC-$TGT.out_$i
	rm active_tmp*
	rm text_active.$src-$TGT.out_$i
	cd -

	# Modify all data
	echo "Get new data for next round"
	U=active_data/$SRC-$TGT/unlabeled_$i
	L=active_data/$SRC-$TGT/labeled_$i.$SRC,active_data/$SRC-$TGT/labeled_$i.$TGT
	oracle=active_data/$SRC-$TGT/oracle_$i

	OU=active_data/$TGT-$SRC/oracle_$out_i
	OL=active_data/$TGT-$SRC/labeled_$out_i.$SRC,active_data/$TGT-$SRC/labeled_$out_i.$TGT
	OO=active_data/$TGT-$SRC/unlabeled_$out_i
	AO=active_data/$SRC-$TGT/test_active.$SRC-$TGT.out_${i}
	onq=active_data/$SRC-$TGT/rnq_$out_i.$SRC-$TGT.$SRC,active_data/$SRC-$TGT/rnq_$out_i.$SRC-$TGT.$TGT
	OT=active_data/$TGT-$SRC/train_$out_i.$SRC,active_data/$TGT-$SRC/train_$out_i.$TGT
	
	python3 active.py modify -U $U \
		-L $L \
		--oracle $oracle \
		-tb $TOK_BUDGET \
		-OU $OU \
		-OL $OL \
		-OO $OO \
		-AO $AO \
		-bt \
		-onq $onq \
		-OT $OT \
		-bttb $((i*TOK_BUDGET + 9 * TOK_BUDGET))
	cd active_data/$SRC-$TGT
	rm test_active.$SRC-$TGT.out*
	rm -rf *_${i}_?
	cd -
}


function BT () {
	# Initialize labeled and unlabeled dataset
	local ACTIVE_FUNC=${1:-random}
	local TOK_BUDGET=${2:-590000}
	local N_ROUNDS=${3:-11}
	local START_ROUND=${4:-1}
	local LAN1=${5:-en}
	local LAN2=${6:-de}
	NGPUS=8

	if [ $START_ROUND -eq 1 ]; then
		Init_Active_BT $LAN1 $LAN2
		Init_Active_BT $LAN2 $LAN1
		cp -r ../../data/defr-en/BT_init_ckpt checkpoints	
		START_ROUND=2
	fi


	for i in $( seq $START_ROUND $N_ROUNDS )
	do	
		# Do active learning
		Active_Learn $((i-1)) $i $LAN1 $LAN2 $ACTIVE_FUNC $TOK_BUDGET
		
		# Train target to source network on new labeled dataset	
		Train_Model $i $LAN2 $LAN1

		# Test target to source betwork
		Test_Model $i $LAN2 $LAN1 test_2014
		Test_Model $i $LAN2 $LAN1 test_2016

		# Do active learning
		Active_Learn $i $i $LAN2 $LAN1 $ACTIVE_FUNC $TOK_BUDGET

		# Train source to target network on new labeled dataset
		Train_Model $i $LAN1 $LAN2

		# Test source to target network
		Test_Model $i $LAN1 $LAN2 test_2014
		Test_Model $i $LAN1 $LAN2 test_2016
	done
}


function main () {
	for ACTIVE_FUNC in dden; do
		BT $ACTIVE_FUNC 590000 11 2 
		mkdir -p result/$ACTIVE_FUNC
		mv active_data checkpoints result/$ACTIVE_FUNC/
		rm -rf data_bin
	done
}


main
