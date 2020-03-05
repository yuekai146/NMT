TEST_INPUT=/data/NMT/data/de-en/iwslt14_de_en/test.de
TEST_REF=/data/NMT/data/de-en/iwslt14_de_en/test.en
FAIRSEQ_PATH=/data/fairseq
for method in longest shortest random;
do
	for i in $( seq 0 11 );
	do
		test_dir=../base.fp16.active_base/results/$method/$((i+1))
		cd $test_dir 
		rm -rf *.out generate.* bleu.txt
		cd -
		python3 translate.py -ckpt $test_dir/checkpoint_best_ppl.pth \
			-text $TEST_INPUT -ref_text $TEST_REF \
			--max_batch_size 0 --tokens_per_batch 1000 \
			-k 5 -max_len 200 > $test_dir/total.out
		
		cat $test_dir/total.out | grep ^H | cut -d " " -f2- > $test_dir/sys.out
		cat $test_dir/total.out | grep ^T | cut -d " " -f2- > $test_dir/ref.out

		cat $test_dir/sys.out | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > $test_dir/generate.sys
		cat $test_dir/ref.out | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > $test_dir/generate.ref

		python3 $FAIRSEQ_PATH/score.py --sys $test_dir/generate.sys --ref $test_dir/generate.ref > $test_dir/bleu.txt
	done
done
