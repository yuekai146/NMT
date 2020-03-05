FAIRSEQ_PATH=/data/fairseq

function test () {
	setup=$1
	method=$2
	i=$3

	mkdir -p result_2016/$method/de-en/$i
	mkdir -p result_2016/$method/en-de/$i 
	mkdir -p checkpoints
	source ~/.bashrc
	aws --endpoint-url=http://oss.hh-b.brainpp.cn s3 cp s3://zhaoyuekai/active_NMT/config/$setup/result/$method/checkpoints/de-en/$i/checkpoint_best_ppl.pth ./checkpoints/
	python3 translate.py -ckpt checkpoints/checkpoint_best_ppl.pth -text ~/test_2016/test.de -ref_text ~/test_2016/test.en --max_batch_size 0 --tokens_per_batch 2000 -max_len 200 > total.out

	cat total.out | grep ^H | cut -d " " -f2- > sys.out
	cat total.out | grep ^T | cut -d " " -f2- > ref.out

	cat sys.out | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > generate.sys
	cat ref.out | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > generate.ref

	python3 $FAIRSEQ_PATH/score.py --sys generate.sys --ref generate.ref > bleu.out
	mv *.out generate.* result_2016/$method/de-en/$i
	rm checkpoints/checkpoint_best_ppl.pth
	
	aws --endpoint-url=http://oss.hh-b.brainpp.cn s3 cp s3://zhaoyuekai/active_NMT/config/$setup/result/$method/checkpoints/en-de/$i/checkpoint_best_ppl.pth ./checkpoints/
	python3 translate.py -ckpt checkpoints/checkpoint_best_ppl.pth -text ~/test_2016/test.en -ref_text ~/test_2016/test.de --max_batch_size 0 --tokens_per_batch 2000 -max_len 200 > total.out

	cat total.out | grep ^H | cut -d " " -f2- > sys.out
	cat total.out | grep ^T | cut -d " " -f2- > ref.out

	cat sys.out | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > generate.sys
	cat ref.out | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > generate.ref

	python3 $FAIRSEQ_PATH/score.py --sys generate.sys --ref generate.ref > bleu.out
	mv *.out generate.* result_2016/$method/en-de/$i
	rm checkpoints/checkpoint_best_ppl.pth
}


function main() {
	for method in lc margin te tte; do
		 for i in $( seq 1 11 ); do
			 test base.back_translation $method $i
		 done
	 done
}

for i in $( seq 7 11 ); do
	test base.back_translation random $i
done

main
