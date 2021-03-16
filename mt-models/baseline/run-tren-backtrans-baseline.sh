#!/bin/bash
set -eo pipefail
# runs tr-en baseline backtranslation model. usage is `bash run-tren-baseline.sh MODEL-OUTPUT-FOLDER`

TODAY=`date '+%Y%m%d-%H%M'`
MARIAN="/home/laurie/apps/marian-dev/build/marian"
INPUT_FOLDER="/mnt/startiger0/laurie/diversity/turkish/parallel_data/data"
DATA="$INPUT_FOLDER"
SRC="tr"
TRG="en"
TRAIN="train.sp"
DEV="dev.sp"
VOCAB_SRC="$INPUT_FOLDER/vocab.$SRC$TRG.yml"
VOCAB_TRG="$INPUT_FOLDER/vocab.$SRC$TRG.yml"
MODELS_LOC="/mnt/startiger0/laurie/diversity/mt-models/baseline/models"
MODEL_DIR="$MODELS_LOC/$1"
MODEL="bt-baseline.npz"
GPUS="0 1 2 3"
WORKSPACE="9500"

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
    -w $WORKSPACE \
    --mini-batch-fit \
    --early-stopping 10 \
    --valid-freq 5000 --save-freq 5000 --disp-freq 500 \
    --valid-metrics bleu cross-entropy perplexity \
    --valid-sets $DATA/$DEV.$SRC $DATA/$DEV.$TRG \
    --valid-translation-output $MODEL_DIR/valid.bpe.en.output \
    --quiet-translation \
    --beam-size 5 --normalize 0.6 \
    --enc-depth 5 --dec-depth 5 \
    --transformer-heads 4 \
    --transformer-dim-ffn 2048 \
    --transformer-postprocess-emb d \
    --transformer-postprocess dan \
    --transformer-dropout 0.3 --label-smoothing 0.3 \
    --dropout-src 0.2 --dropout-trg 0.2 \
    --learn-rate 0.0003 --lr-warmup 16000 --lr-decay-inv-sqrt 16000 --lr-report \
    --optimizer-params 0.9 0.98 1e-09 --clip-norm 5 \
    --tied-embeddings \
    --devices $GPUS --sync-sgd --seed 2626 \
    --exponential-smoothing \
    --log $MODEL_DIR/train-$TODAY.log --valid-log $MODEL_DIR/valid-$TODAY.log \
    --keep-best --overwrite
