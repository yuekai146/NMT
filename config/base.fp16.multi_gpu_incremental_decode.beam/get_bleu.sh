TEST_INPUT=/data/NMT/data/de-en/iwslt14_de_en/test.de
TEST_REF=/data/NMT/data/de-en/iwslt14_de_en/test.en
for method in lc margin tte te;
do
	for i in $( seq 0 11 );
	do
		python3 translate.py -ckpt ../base.fp16.incremental_decode.uncertainty_sampling.$method/$method/checkpoints/$((i+1))/checkpoint_best_ppl.pth \
			-text $TEST_INPUT -ref_text $TEST_REF \
			--max_batch_size 0 --tokens_per_batch 2000 \
			-k 5 -max_len 200 > ../base.fp16.incremental_decode.uncertainty_sampling.$method/$method/checkpoints/$((i+1))/total.out
		
		cat ../base.fp16.incremental_decode.uncertainty_sampling.$method/$method/checkpoints/$((i+1))/total.out | grep ^H | cut -d " " -f2- > ../base.fp16.incremental_decode.uncertainty_sampling.$method/$method/checkpoints/$((i+1))/sys.out
		cat ../base.fp16.incremental_decode.uncertainty_sampling.$method/$method/checkpoints/$((i+1))/total.out | grep ^T | cut -d " " -f2- > ../base.fp16.incremental_decode.uncertainty_sampling.$method/$method/checkpoints/$((i+1))/ref.out

		cat ../base.fp16.incremental_decode.uncertainty_sampling.$method/$method/checkpoints/$((i+1))/sys.out | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > ../base.fp16.incremental_decode.uncertainty_sampling.$method/$method/checkpoints/$((i+1))/generate.sys
		cat ../base.fp16.incremental_decode.uncertainty_sampling.$method/$method/checkpoints/$((i+1))/ref.out | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > ../base.fp16.incremental_decode.uncertainty_sampling.$method/$method/checkpoints/$((i+1))/generate.ref

		python3 $FAIRSEQ_PATH/score.py --sys ../base.fp16.incremental_decode.uncertainty_sampling.$method/$method/checkpoints/$((i+1))/generate.sys --ref ../base.fp16.incremental_decode.uncertainty_sampling.$method/$method/checkpoints/$((i+1))/generate.ref > ../base.fp16.incremental_decode.uncertainty_sampling.$method/$method/checkpoints/$((i+1))/bleu.txt
	done
done
