SCRIPTS=~/mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl
LC=$SCRIPTS/tokenizer/lowercase.perl
BPEROOT=~/fastBPE
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
