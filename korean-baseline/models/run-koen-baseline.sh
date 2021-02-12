#!/bin/bash
set -eo pipefail
# runs korean baseline model. usage is `bash run-koen-baseline.sh MODEL-OUTPUT-FOLDER`

TODAY=`date '+%Y%m%d-%H%M'`
MARIAN="/home/laurie/apps/marian-dev/build/marian"
DATA="/mnt/startiger0/laurie/diversity/korean-baseline/data/processed-koen-data/data"
SRC="ko"
TRG="en"
TRAIN="train.bpe"
DEV="dev.bpe"
VOCAB_SRC="/mnt/startiger0/laurie/diversity/korean-baseline/data/processed-koen-data/model/vocab.$SRC.yml"
VOCAB_TRG="/mnt/startiger0/laurie/diversity/korean-baseline/data/processed-koen-data/model/vocab.$TRG.yml"
MODELS_LOC="/mnt/startiger0/laurie/diversity/korean-baseline/models"
MODEL_DIR="$MODELS_LOC/$1"
MODEL="koen-baseline.npz"
GPUS="0 1 2 3 4 5 6 7"
WORKSPACE="10000"

# change to correct directory
mkdir -p $MODEL_DIR
cd $MODEL_DIR
echo "changed dir to $MODEL_DIR"

$MARIAN \
    --model $MODEL --type transformer \
    --train-sets $DATA/$TRAIN.$SRC $DATA/$TRAIN.$TRG \
    --max-length 80 \
    --max-length-crop \
    --vocabs $VOCAB_SRC $VOCAB_TRG \
    -w $WORKSPACE --maxi-batch 40 \
    --mini-batch 80 \
    --early-stopping 10 \
    --valid-freq 5000 --save-freq 5000 --disp-freq 500 \
    --valid-metrics cross-entropy perplexity bleu-detok \
    --valid-sets $DATA/$DEV.$SRC $DATA/$DEV.$TRG \
    --valid-translation-output $MODEL_DIR/valid.bpe.en.output \
    --quiet-translation \
    --valid-mini-batch 40 \
    --beam-size 5 --normalize 0.6 \
    --enc-depth 1 --dec-depth 1 \
    --transformer-heads 2 \
    --transformer-postprocess-emb d \
    --transformer-postprocess dan \
    --transformer-dropout 0.3 --label-smoothing 0.2 \
    --dropout-src 0.3 --dropout-trg 0.3 \
    --learn-rate 0.0005 --lr-warmup 16000 --lr-decay-inv-sqrt 16000 --lr-report \
    --optimizer-params 0.9 0.98 1e-09 --clip-norm 5 \
    --tied-embeddings \
    --devices $GPUS --sync-sgd --seed 2626 \
    --exponential-smoothing \
    --log $MODEL_DIR/train-$TODAY.log --valid-log $MODEL_DIR/valid-$TODAY.log  
