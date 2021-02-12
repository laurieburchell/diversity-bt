#!/bin/bash
set -eo pipefail
# script to preprocess data for diversity experiment

SRC="ko"
TRG="en"
OUTPUT_FOLDER="processed-${SRC}${TRG}-data"
INPUT_FOLDER="/home/laurie/projects/diversity/korean-baseline/data/korean-parallel-corpora/korean-english-news-v1"
SRC_TRAIN="$INPUT_FOLDER/korean-english-park.train.ko"
SRC_DEV="$INPUT_FOLDER/korean-english-park.dev.ko"
SRC_TEST="$INPUT_FOLDER/korean-english-park.test.ko"
TRG_TRAIN="$INPUT_FOLDER/korean-english-park.train.en"
TRG_DEV="$INPUT_FOLDER/korean-english-park.dev.en"
TRG_TEST="$INPUT_FOLDER/korean-english-park.test.en"
MOSES_DECODER="/home/laurie/apps/mosesdecoder"
SUBWORD_NMT="/home/laurie/apps/subword-nmt/subword_nmt"
BPE_OPS=2000  # this is quite low because low-resource
BPE_THRESHOLD=10  # minimum number of times we see character sequence in train text before we merge to one unit. applied independently even with joint vocab.
MARIAN_VOCAB="/home/laurie/apps/marian-dev/build/marian-vocab"

echo "running preprocessing script at current location"
# make output folder
mkdir -p $OUTPUT_FOLDER
mkdir -p $OUTPUT_FOLDER/model
mkdir -p $OUTPUT_FOLDER/data
echo "switching to output folder"
cd $OUTPUT_FOLDER
echo "copying input files"
cp $SRC_TRAIN data/train.raw.$SRC
cp $SRC_DEV data/dev.raw.$SRC
cp $SRC_TEST data/test.raw.$SRC
cp $TRG_TRAIN data/train.raw.$TRG
cp $TRG_DEV data/dev.raw.$TRG
cp $TRG_TEST data/test.raw.$TRG

# normalise punctuation and tokenise
for lang in $SRC $TRG
do 
    for prefix in train dev test
    do
        echo "tokenising $prefix.raw.$lang"
        cat data/$prefix.raw.$lang  \
        | $MOSES_DECODER/scripts/tokenizer/normalize-punctuation.perl -l $lang \
        | $MOSES_DECODER/scripts/tokenizer/tokenizer.perl -a -l $lang > data/$prefix.tok.$lang
    done    
done

# clean empty and long sentences, and sentences with high source-target ratio (training corpus only)
echo "clean long sentences"
$MOSES_DECODER/scripts/training/clean-corpus-n.perl -ratio 9 data/train.tok $SRC $TRG data/train.tok.clean 1 200

# switch out newly clean training corpus (for the sake of my nice loops)
for lang in $TRG $SRC
do
    mv data/train.tok.$lang data/train.tok.unclean.$lang
    mv data/train.tok.clean.$lang data/train.tok.$lang
done

# train truecaser
echo "training and applying truecaser"
for lang in $SRC $TRG
do 
    $MOSES_DECODER/scripts/recaser/train-truecaser.perl -corpus data/train.tok.$lang -model model/tc.$lang
    # apply truecaser
    for prefix in train dev test
    do
        $MOSES_DECODER/scripts/recaser/truecase.perl -model model/tc.$lang < data/$prefix.tok.$lang > data/$prefix.tc.$lang
    done
done

# train BPE models
echo "training BPE models"
for lang in $SRC $TRG
do
    $SUBWORD_NMT/learn_joint_bpe_and_vocab.py -i data/train.tc.$lang --write-vocabulary model/vocab.$lang -s $BPE_OPS -o model/$lang.bpe
done

# apply BPE
echo "applying BPE"
for lang in $SRC $TRG
do
    for prefix in train dev test
    do
	#$SUBWORD_NMT/apply_bpe.py -c model/$SRC$TRG.bpe < data/$prefix.tc.$lang > data/$prefix.bpe.$lang
        $SUBWORD_NMT/apply_bpe.py -c model/$lang.bpe --vocabulary model/vocab.$lang --vocabulary-threshold 10 < data/$prefix.tc.$lang > data/$prefix.bpe.$lang
    done
done

# generate marian vocab
echo "generating marian vocab"
for lang in $SRC $TRG
do
    cat data/train.bpe.$lang | $MARIAN_VOCAB > model/vocab.$lang.yml
done
