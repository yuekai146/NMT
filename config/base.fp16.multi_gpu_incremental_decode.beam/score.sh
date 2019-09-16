FAIRSEQ_PATH=/data/fairseq

python3 translate.py -ckpt ../base.fp16.incremental_decode/checkpoints/checkpoint_best_ppl.pth -text /data/NMT/data/de-en/iwslt14_de_en/test.de -ref_text /data/NMT/data/de-en/iwslt14_de_en/test.en --max_batch_size 0 --tokens_per_batch 2000 -k 5 -max_len 200 > total.out

cat total.out | grep ^H | cut -d " " -f2- > sys.out
cat total.out | grep ^T | cut -d " " -f2- > ref.out

cat sys.out | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > generate.sys
cat ref.out | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > generate.ref

python3 $FAIRSEQ_PATH/score.py --sys generate.sys --ref generate.ref
