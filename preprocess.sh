SCRIPTS=/data/mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl
LC=$SCRIPTS/tokenizer/lowercase.perl
BPEROOT=/data/fastBPE


function process_iwslt () {
	BPE_TOKENS=10000
	BASE_DIR=/mnt/data/zhaoyuekai/active_NMT/playground/de-en

	src=de
	tgt=en
	lang=de-en
	prep=$BASE_DIR/iwslt14_de_en
	tmp=$prep/tmp
	orig=$BASE_DIR/orig

	CORPORA=(
		"total"
	)

	TEST_CORPORA=(
		"test"
	)

	mkdir -p $orig $tmp $prep

	cd $orig

	echo "pre-processing train data..."

	for l in $src $tgt; do
	    rm $tmp/train.tags.$lang.tok.$l
	    for f in "${CORPORA[@]}"; do
		cat $orig/$f.$l | \
		    perl $NORM_PUNC $l | \
		    perl $REM_NON_PRINT_CHAR | \
		    perl $TOKENIZER -threads 8 -a -l $l | \
		    perl $LC >> $tmp/train.tags.$lang.tok.$l
	    done
	done

	echo "Tokenize test set"
	for l in $src $tgt; do
	    rm $tmp/test.tags.$lang.tok.$l
	    for f in "${TEST_CORPORA[@]}"; do
		cat $orig/$f.$l | \
		    perl $NORM_PUNC $l | \
		    perl $REM_NON_PRINT_CHAR | \
		    perl $TOKENIZER -threads 8 -a -l $l | \
		    perl $LC >> $tmp/test.$l
	    done
	done

	echo "splitting train and valid..."
	for l in $src $tgt; do
	    awk '{if (NR%1333 == 0)  print $0; }' $tmp/train.tags.$lang.tok.$l > $tmp/valid.$l
	    awk '{if (NR%1333 != 0)  print $0; }' $tmp/train.tags.$lang.tok.$l > $tmp/train.$l
	done

	TRAIN=$tmp/train
	BPE_CODE=$prep/code

	echo "learn_bpe.py on ${TRAIN}..."
	$BPEROOT/fast learnbpe $BPE_TOKENS $TRAIN.$src $TRAIN.$tgt > $BPE_CODE

	for L in $src $tgt; do
		f=train.$L
		echo "apply_bpe to ${f}..."
		$BPEROOT/fast applybpe $tmp/bpe.$f $tmp/$f $BPE_CODE
		$BPEROOT/fast getvocab $tmp/bpe.$f > vocab.$L
		$BPEROOT/fast applybpe $tmp/bpe.valid.$L $tmp/valid.$L $BPE_CODE vocab.$L
		$BPEROOT/fast applybpe $tmp/bpe.test.$L $tmp/test.$L $BPE_CODE vocab.$L
		cp $tmp/bpe.test.$L $prep/test.$L
	done


	perl $CLEAN -ratio 1.5 $tmp/bpe.train $src $tgt $prep/train 1 250
	perl $CLEAN -ratio 1.5 $tmp/bpe.valid $src $tgt $prep/valid 1 250
	mv $orig/vocab.$src $prep
	mv $orig/vocab.$tgt $prep
}


function process_wmt () {
	BPE_TOKENS=32000
	
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

	OUTDIR=wmt17_en_de
	BASE_DIR=/data/NMT/data/de-en/wmt17_de_en

	if [ ! -d "$SCRIPTS" ]; then
	    echo "Please set SCRIPTS variable correctly to point to Moses scripts."
	    exit
	fi

	src=en
	tgt=de
	lang=en-de
	prep=$BASE_DIR/$OUTDIR
	tmp=$prep/tmp
	orig=$BASE_DIR/orig
	dev=dev/newstest2013

	mkdir -p $orig $tmp $prep

	cd $orig

	for ((i=0;i<${#FILES[@]};++i)); do
	    file=${FILES[i]}
	    if [ ${file: -4} == ".tgz" ]; then
	        tar zxvf $file
       	    elif [ ${file: -4} == ".tar" ]; then
	        tar xvf $file
	    fi
	done
	cd ..

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

	TRAIN=$tmp/train.de-en
	BPE_CODE=$prep/code

	echo "learn_bpe.py on ${TRAIN}..."
	$BPEROOT/fast learnbpe $BPE_TOKENS $tmp/train.$src $tmp/train.$tgt > $BPE_CODE

	for L in $src $tgt; do
		f=train.$L
		echo "apply_bpe.py to ${f}..."
		$BPEROOT/fast applybpe $tmp/bpe.$f $tmp/$f $BPE_CODE
		$BPEROOT/fast getvocab $tmp/bpe.$f > vocab.$L
	done
	
	for L in $src $tgt; do
		for f in valid.$L test.$L; do
			echo "apply_bpe.py to ${f}..."
			$BPEROOT/fast applybpe $tmp/bpe.$f $tmp/$f $BPE_CODE vocab.$L
		done
	done

	perl $CLEAN -ratio 1.5 $tmp/bpe.train $src $tgt $prep/train 1 250
	perl $CLEAN -ratio 1.5 $tmp/bpe.valid $src $tgt $prep/valid 1 250

	for L in $src $tgt; do
	    cp $tmp/bpe.test.$L $prep/test.$L
	done
}


if [ "$1" == "--wmt" ]; then
	process_wmt
else
	process_iwslt
fi

