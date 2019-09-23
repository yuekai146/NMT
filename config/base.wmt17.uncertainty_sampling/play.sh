ACTIVE=./active_data
ACTIVE_OUT=$ACTIVE/test_active.out_
U=$ACTIVE/unlabeled_
L=$ACTIVE/labeled_
ORACLE=$ACTIVE/oracle_
ACTIVE_FUNC=lc
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
	mkdir -p $ACTIVE
	cp $ORIG_SRC ${U}0
	cp $ORIG_TGT ${ORACLE}0
	touch ${L}0.$SRC ${L}0.$TGT
	i=0

	# Do active learning
	cd active_data
	num_U=$(cat unlabeled_0 | wc -l)
	num_chunk=$(($num_U / $NGPUS + 1))
	split -l $num_chunk unlabeled_$i unlabeled_${i}_ -da 1
	split -l $num_chunk oracle_$i oracle_${i}_ -da 1
	cd ..	

	for j in $( seq 0 $((NGPUS - 1)) )
	do
		echo "CUDA_VISIBLE_DEVICES=$j python3 active.py score -a $ACTIVE_FUNC \
			-i ${U}${i}_$j -ref ${ORACLE}${i}_$j \
			-ckpt ../base.wmt17/checkpoints/checkpoint_best_ppl.pth \
			--max_batch_size 0 \
			--tokens_per_batch 16384 > test_active.out_${i}_${j}" >> parallel_active.sh
	done
	parallel -j $NGPUS < parallel_active.sh
	rm parallel_active.sh
	mv test_active.out_${i}_? active_data/
	cat active_data/test_active.out_${i}_? > active_data/test_active.out_$i
	python3 active.py modify -U $U$i -L $L$i.$SRC,$L$i.$TGT --oracle $ORACLE$i -tb $TOK_BUDGET \
		-OU $U$((i+1)) -OL $L$((i+1)).$SRC,$L$((i+1)).$TGT \
		-OO $ORACLE$((i+1)) -AO $ACTIVE_OUT$i 
	cd active_data
	rm test_active.out*
	rm *_${i}_?
	cd ..
}

main
