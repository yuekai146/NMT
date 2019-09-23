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

	OUTDIR=wmt17_de_en
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


function process_defr2en () {
	BPE_TOKENS=32000
	LEAST_WORD_FREQ=100
	
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

	OUTDIR=wmt17_de_en
	BASE_DIR=/data/NMT/data/defr-en/wmt17_de_en

	if [ ! -d "$SCRIPTS" ]; then
	    echo "Please set SCRIPTS variable correctly to point to Moses scripts."
	    exit
	fi

	src=de
	tgt=en
	lang=de-en
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

	FILES=(
	    "training-parallel-europarl-v7.tgz"
	    "training-parallel-commoncrawl.tgz"
	    "training-parallel-un.tgz"
	    "training-parallel-nc-v9.tgz"
	    "training-giga-fren.tar"
	    "test-full.tgz"
	)
	CORPORA=(
	    "training/europarl-v7.fr-en"
	    "commoncrawl.fr-en"
	    "un/undoc.2000.fr-en"
	    "training/news-commentary-v9.fr-en"
	    "giga-fren.release2.fixed"
	)

	if [ ! -d "$SCRIPTS" ]; then
	    echo "Please set SCRIPTS variable correctly to point to Moses scripts."
	    exit
	fi

	OUTDIR=wmt14_fr_en
	BASE_DIR=/data/NMT/data/defr-en/wmt14_fr_en

	src=fr
	tgt=en
	lang=fr-en
	prep=$BASE_DIR/$OUTDIR
	tmp=$prep/tmp
	orig=$BASE_DIR/orig

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

	gunzip giga-fren.release2.fixed.*.gz
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
	    grep '<seg id' $orig/test-full/newstest2014-fren-$t.$l.sgm | \
		sed -e 's/<seg id="[0-9]*">\s*//g' | \
		sed -e 's/\s*<\/seg>\s*//g' | \
		sed -e "s/\’/\'/g" | \
	    perl $TOKENIZER -threads 8 -a -l $l > $tmp/test.$l
	    echo ""
	done

	fr_base_dir=data/defr-en/wmt14_fr_en/wmt14_fr_en/tmp
	de_base_dir=data/defr-en/wmt17_de_en/wmt17_de_en/tmp

	$BPEROOT/fast getvocab $fr_base_dir/train.tags.fr-en.tok.en $fr_base_dir/train.tags.fr-en.tok.fr > $fr_base_dir/vocab.total	
	$BPEROOT/fast getvocab $de_base_dir/train.tags.de-en.tok.en $de_base_dir/train.tags.de-en.tok.de > $de_base_dir/vocab.total	

	python3 get_char_count.py -tv $fr_base_dir/vocab.total -lf $((LEAST_WORD_FREQ*10)) -os $fr_base_dir/train.tags.fr-en.tok.fr -ot $fr_base_dir/train.tags.fr-en.tok.en > $fr_base_dir/train.fr-en.filtered.tok.total
	cat $fr_base_dir/train.fr-en.filtered.tok.total | grep ^S | cut -d " " -f3- > $fr_base_dir/train.fr-en.filtered.tok.fr
	cat $fr_base_dir/train.fr-en.filtered.tok.total | grep ^T | cut -d " " -f3- > $fr_base_dir/train.fr-en.filtered.tok.en

	python3 get_char_count.py -tv $de_base_dir/vocab.total -lf $LEAST_WORD_FREQ -os $de_base_dir/train.tags.de-en.tok.de -ot $de_base_dir/train.tags.de-en.tok.en > $de_base_dir/train.de-en.filtered.tok.total
	cat $de_base_dir/train.de-en.filtered.tok.total | grep ^S | cut -d " " -f3- > $de_base_dir/train.de-en.filtered.tok.de
	cat $de_base_dir/train.de-en.filtered.tok.total | grep ^T | cut -d " " -f3- > $de_base_dir/train.de-en.filtered.tok.en

	rm -rf $fr_base_dir/train.fr-en.filtered.tok.total
	rm -rf $de_base_dir/train.de-en.filtered.tok.total

	base_dir=data/defr-en
	tmp=$base_dir/tmp
	src1=fr
	src2=de
	tgt=en
	mkdir -p $tmp

	paste -d ' ||| ' $fr_base_dir/train.fr-en.filtered.tok.fr /dev/null /dev/null /dev/null /dev/null $fr_base_dir/train.fr-en.filtered.tok.en > $tmp/corpus.fr-en.total
       	cat $tmp/corpus.fr-en.total | shuf | head -n 4000000 > $tmp/train.fr-en.total
	cat $tmp/train.fr-en.total | awk -F ' \|\|\| ' '{print $1}' > $tmp/train.fr-en.fr
	cat $tmp/train.fr-en.total | awk -F ' \|\|\| ' '{print $2}' > $tmp/train.fr-en.en
	cat $de_base_dir/train.de-en.filtered.tok.de  >> $tmp/train.de-en.de
	cat $de_base_dir/train.de-en.filtered.tok.en  >> $tmp/train.de-en.en

	cat $tmp/train.fr-en.fr >> $tmp/bpe.src
	cat $tmp/train.de-en.de >> $tmp/bpe.src
	cat $tmp/train.fr-en.en >> $tmp/bpe.tgt
	cat $tmp/train.de-en.en >> $tmp/bpe.tgt

	$BPEROOT/fast learnbpe $BPE_TOKENS $tmp/bpe.src $tmp/bpe.tgt > $tmp/code 
	
	for L in $src1 $src2; do
		f=train.$L-$tgt.$L
		echo "apply_bpe to ${f}..."
		$BPEROOT/fast applybpe $tmp/bpe.$f $tmp/$f $tmp/code
		$BPEROOT/fast getvocab $tmp/bpe.$f > $tmp/vocab.$L-$tgt.$L
		
		f=train.$L-$tgt.$tgt
		$BPEROOT/fast applybpe $tmp/bpe.$f $tmp/$f $tmp/code
		$BPEROOT/fast getvocab $tmp/bpe.$f > $tmp/vocab.$L-$tgt.$tgt
	done
	
	for L in $src $tgt; do
		for f in valid.$L test.$L; do
			echo "apply_bpe.py to ${f}..."
			$BPEROOT/fast applybpe $tmp/bpe.$f $tmp/$f $BPE_CODE vocab.$L
		done
	done

	perl $CLEAN -ratio 1.5 $tmp/bpe.train.$src1-$tgt $src1 $tgt $base_dir/train.$src1-$tgt 1 250
	perl $CLEAN -ratio 1.5 $tmp/bpe.train.$src2-$tgt $src2 $tgt $base_dir/train.$src2-$tgt 1 250
	$BPEROOT/fast applybpe $base_dir/test.fr-en.$src1 $fr_base_dir/test.$src1 $tmp/code $tmp/vocab.fr-en.$src1 
	$BPEROOT/fast applybpe $base_dir/test.fr-en.$tgt $fr_base_dir/test.$tgt $tmp/code $tmp/vocab.fr-en.$tgt
	
	$BPEROOT/fast applybpe $base_dir/test.de-en.$src2 $de_base_dir/test.$src2 $tmp/code $tmp/vocab.de-en.$src2
	$BPEROOT/fast applybpe $base_dir/test.de-en.$tgt $de_base_dir/test.$tgt $tmp/code $tmp/vocab.de-en.$tgt
	cp $base_dir/test.fr-en.fr $base_dir/valid.fr-en.fr
	cp $base_dir/test.fr-en.en $base_dir/valid.fr-en.en
	cp $base_dir/test.de-en.de $base_dir/valid.de-en.de
	cp $base_dir/test.de-en.en $base_dir/valid.de-en.en

	cat $base_dir/train.fr-en.fr >> $base_dir/train.src.total
	cat $base_dir/train.fr-en.en >> $base_dir/train.tgt.total
	cat $base_dir/train.de-en.de >> $base_dir/train.src.total
	cat $base_dir/train.de-en.en >> $base_dir/train.tgt.total
	$BPEROOT/fast getvocab $base_dir/train.src.total $base_dir/train.tgt.total > $base_dir/vocab.total
	mv $tmp/code $base_dir/code
	rm -rf $base_dir/train.*.total
}


if [ "$1" == "--wmt" ]; then
	process_wmt
elif [ "$1" == "--defr2en" ]; then
	process_defr2en
else
	process_iwslt
fi

