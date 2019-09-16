#!/bin/bash
# Adapted from https://github.com/facebookresearch/MIXER/blob/master/prepareData.sh

#echo 'Cloning Moses github repository (for tokenization scripts)...'
#git clone https://github.com/moses-smt/mosesdecoder.git

#echo 'Cloning Subword NMT repository (for BPE pre-processing)...'
#git clone https://github.com/rsennrich/subword-nmt.git

SCRIPTS=~/mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl
BPEROOT=/home/zhanghaoran02/fastBPE
BPE_TOKENS=40000
OUTDIR=wmt14_en_de

URLS=(
    "http://statmt.org/wmt13/training-parallel-europarl-v7.tgz"
    "http://statmt.org/wmt13/training-parallel-commoncrawl.tgz"
    "http://data.statmt.org/wmt17/translation-task/training-parallel-nc-v12.tgz"
    "http://data.statmt.org/wmt17/translation-task/dev.tgz"
    "http://statmt.org/wmt14/test-full.tgz"
)
FILES=(
    "training-parallel-europarl-v7.tgz"
    "training-parallel-commoncrawl.tgz"
    "training-parallel-nc-v12.tgz"
    "dev.tgz"
    "test-full.tgz"
)
CORPORA=(
    "training/europarl-v7.de-en"
    "commoncrawl.de-en"
    "training/news-commentary-v12.de-en"
)

src=en
tgt=de
lang=en-de
prep=$OUTDIR
tmp=$prep/tmp
orig=orig
dev=dev/newstest2013

TRAIN=$tmp/train.de-en
BPE_CODE=$prep/code

URLS[2]="http://statmt.org/wmt14/training-parallel-nc-v9.tgz"
FILES[2]="training-parallel-nc-v9.tgz"
CORPORA[2]="training/news-commentary-v9.de-en"

if [ ! -d "$SCRIPTS" ]; then
    echo "Please set SCRIPTS variable correctly to point to Moses scripts."
    exit
fi

function download () {
	mkdir -p $orig $tmp $prep

	cd $orig

	for ((i=0;i<${#URLS[@]};++i)); do
	    file=${FILES[i]}
	#    if [ -f $file ]; then
	#	echo "$file already exists, skipping download"
	#    else
	#	url=${URLS[i]}
	#	wget "$url"
	#	if [ -f $file ]; then
	#	    echo "$url successfully downloaded."
	#	else
	#	    echo "$url not successfully downloaded."
	#	    exit -1
	#	fi
	#	if [ ${file: -4} == ".tgz" ]; then
		    tar zxvf $file
	#	elif [ ${file: -4} == ".tar" ]; then
	#	    tar xvf $file
	#	fi
	#    fi
	done
	#cd ..
}

function pre-processing () {
	echo "pre-processing train data..."
	for l in $src $tgt; do
	    rm $tmp/train.tags.$lang.tok.$l
	    for f in "${CORPORA[@]}"; do
		cat $orig/$f.$l | \
		    perl $NORM_PUNC $l | \
		    perl $REM_NON_PRINT_CHAR | \
		    perl $TOKENIZER -threads 8 -a -l $l >> $tmp/train.tags.$lang.tok.$l
	    done
	done

	echo "pre-processing test data..."
	for l in $src $tgt; do
	    if [ "$l" == "$src" ]; then
		t="src"
	    else
		t="ref"
	    fi
	    grep '<seg id' $orig/test-full/newstest2014-deen-$t.$l.sgm | \
		sed -e 's/<seg id="[0-9]*">\s*//g' | \
		sed -e 's/\s*<\/seg>\s*//g' | \
		sed -e "s/\’/\'/g" | \
	    perl $TOKENIZER -threads 8 -a -l $l > $tmp/test.$l
	    echo ""
	done

	echo "splitting train and valid..."
	for l in $src $tgt; do
	    awk '{if (NR%100 == 0)  print $0; }' $tmp/train.tags.$lang.tok.$l > $tmp/valid.$l
	    awk '{if (NR%100 != 0)  print $0; }' $tmp/train.tags.$lang.tok.$l > $tmp/train.$l
	done

	rm -f $TRAIN
	for l in $src $tgt; do
	    cat $tmp/train.$l >> $TRAIN
	done
}

function bpe () {
	echo "learn_bpe.py on ${TRAIN}..."
	$BPEROOT/fast learnbpe $BPE_TOKENS $tmp/train.$src $tmp/train.$tgt > $BPE_CODE

	echo "apply codes to train"
	for L in $src $tgt; do
	    $BPEROOT/fast applybpe $tmp/bpe.train.$L $tmp/train.$L $BPE_CODE
	done

	echo "get train vocabulary"
	for L in $src $tgt; do
	    $BPEROOT/fast getvocab $tmp/bpe.train.$L > $tmp/vocab.$L.$BPE_TOKENS
	done

	for L in $src $tgt; do
	    for f in valid.$L test.$L; do
		echo "apply_bpe.py to ${f}..."
		$BPEROOT/fast applybpe $tmp/bpe.$f $tmp/$f $BPE_CODE $tmp/vocab.$L.$BPE_TOKENS
	    done
	done

	perl $CLEAN -ratio 1.5 $tmp/bpe.train $src $tgt $prep/train 1 250
	perl $CLEAN -ratio 1.5 $tmp/bpe.valid $src $tgt $prep/valid 1 250

	for L in $src $tgt; do
	    cp $tmp/bpe.test.$L $prep/test.$L
	done
}

pre-processing
bpe
