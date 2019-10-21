function run_rand () {
	local active_func=$1
	local SRC=$2
	local TGT=$3

	rand_ckpt=/data/NMT/config/BT_comp.rand_20_percent/checkpoints
	FAIRSEQ_PATH=/data/fairseq
	data_dir=more_cmp

	python3 translate.py -ckpt $rand_ckpt/$SRC-$TGT/checkpoint_best_ppl.pth -text $data_dir/test.rand-${active_func}.$SRC -ref_text $data_dir/test.rand-${active_func}.$TGT --max_batch_size 0 --tokens_per_batch 2000 -max_len 200 --greedy > total.out

	cat total.out | grep ^H | cut -d " " -f2- > sys.out
	cat total.out | grep ^T | cut -d " " -f2- > ref.out

	cat sys.out | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > generate.sys
	cat ref.out | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > generate.ref

	python3 $FAIRSEQ_PATH/score.py --sys generate.sys --ref generate.ref > bleu.$SRC-$TGT.rand-${active_func}.rand.txt
	rm *.out
	rm generate.*
}


function run_active () {
	local active_func=$1
	local SRC=$2
	local TGT=$3

	ac_ckpt=/data/NMT/config/BT_comp.uncertainty_sampling/checkpoints/$active_func
	FAIRSEQ_PATH=/data/fairseq
	data_dir=more_cmp

	python3 translate.py -ckpt $ac_ckpt/$SRC-$TGT/checkpoint_best_ppl.pth -text $data_dir/test.rand-${active_func}.$SRC -ref_text $data_dir/test.rand-${active_func}.$TGT --max_batch_size 0 --tokens_per_batch 2000 -max_len 200 --greedy > total.out

	cat total.out | grep ^H | cut -d " " -f2- > sys.out
	cat total.out | grep ^T | cut -d " " -f2- > ref.out

	cat sys.out | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > generate.sys
	cat ref.out | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > generate.ref

	python3 $FAIRSEQ_PATH/score.py --sys generate.sys --ref generate.ref > bleu.$SRC-$TGT.rand-${active_func}.${active_func}.txt
	rm *.out
	rm generate.*
}


function main () {
	for active_func in lc;
	do
		run_rand $active_func en de
		run_active $active_func en de
		run_rand $active_func de en
		run_active $active_func de en
		mv bleu.*.txt more_cmp/
	done
}


main
