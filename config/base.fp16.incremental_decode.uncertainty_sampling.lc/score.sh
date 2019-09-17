sys=$1
ref=$2
FAIRSEQ_PATH=/data/fairseq

cat $sys | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > generate.sys
cat $ref | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > generate.ref

python3 $FAIRSEQ_PATH/score.py --sys generate.sys --ref generate.ref
