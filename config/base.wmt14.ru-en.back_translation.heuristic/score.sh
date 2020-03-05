FAIRSEQ_PATH=/data/fairseq

function test () {
	setup=$1
	method=$2
	i=$3

	mkdir -p result_2016/$method/de-en/$i
	#mkdir -p result_2016/$method/en-de/$i 
	python3 translate.py -ckpt result/dden/checkpoints/de-en/$i/checkpoint_best_ppl.pth -text ~/test_2016/test.de -ref_text ~/test_2016/test.en --max_batch_size 0 --tokens_per_batch 2000 -max_len 200 > total.out

	cat total.out | grep ^H | cut -d " " -f2- > sys.out
	cat total.out | grep ^T | cut -d " " -f2- > ref.out

	cat sys.out | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > generate.sys
	cat ref.out | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > generate.ref

	python3 $FAIRSEQ_PATH/score.py --sys generate.sys --ref generate.ref > bleu.out
	mv *.out generate.* result_2016/$method/de-en/$i
	
#	python3 translate.py -ckpt result/dden/checkpoints/en-de/$i/checkpoint_best_ppl.pth -text ~/test_2016/test.en -ref_text ~/test_2016/test.de --max_batch_size 0 --tokens_per_batch 2000 -max_len 200 > total.out
#
#	cat total.out | grep ^H | cut -d " " -f2- > sys.out
#	cat total.out | grep ^T | cut -d " " -f2- > ref.out
#
#	cat sys.out | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > generate.sys
#	cat ref.out | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > generate.ref
#
#	python3 $FAIRSEQ_PATH/score.py --sys generate.sys --ref generate.ref > bleu.out
#	mv *.out generate.* result_2016/$method/en-de/$i
}


function main() {
	 for i in $( seq 1 11 ); do
		 test base.back_translation dden $i
	 done
}

main
