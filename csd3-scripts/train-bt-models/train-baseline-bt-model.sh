#!/bin/bash

# train baseline bt model to generate n-best beam bt and sampled bt

set -eo pipefail

TODAY=`date '+%F_%H%M'`
MARIAN="/home/cs-burc1/apps/marian-dev/build/marian"
GPUS="0 1 2 3"
WORKSPACE=14000
SRC="tr"
TRG="en"
MODEL_DIR="/home/cs-burc1/projects/diversity-bt/turkish/mt-models"
MODEL="baseline-bt-$SRC$TRG"
DATA_DIR="/home/cs-burc1/projects/diversity-bt/turkish/parallel-data"
TRAIN_SRC="train.tr.pp.filtered"
TRAIN_TRG="train.en.pp.filtered"
DEV_SRC="dev.tr.pp"
DEV_TRG="dev.en.pp"
VOCAB="$DATA_DIR/vocab.tren.spm"
SCRATCH="/local/scratch"

# copy data
echo "copying data from $DATA_DIR to $SCRATCH"
rsync -avzhL --progress $DATA_DIR/$TRAIN_SRC $DATA_DIR/$TRAIN_TRG $DATA_DIR/$DEV_SRC $DATA_DIR/$DEV_TRG $SCRATCH/
echo "data copied"

# make model dir
echo "model dir is at $MODEL_DIR/$MODEL"
mkdir -p $MODEL_DIR/$MODEL

$MARIAN \
        --devices $GPUS \
        --type transformer \
        --model $MODEL_DIR/$MODEL/model.npz \
        --train-sets $SCRATCH/$TRAIN_SRC $SCRATCH/$TRAIN_TRG \
        --valid-sets $SCRATCH/$DEV_SRC $SCRATCH/$DEV_TRG \
        --vocabs $VOCAB $VOCAB \
        --valid-metrics ce-mean-words bleu perplexity \
        --valid-mini-batch 64 \
        --early-stopping 5 \
        -w $WORKSPACE \
        --mini-batch-fit \
        --maxi-batch 1000 \
	--max-length 150 \
	--max-length-crop \
        --valid-freq 5000 \
        --save-freq 5000 \
        --disp-freq 500 \
        --log $MODEL_DIR/$MODEL/train-${TODAY}.log \
        --valid-log $MODEL_DIR/$MODEL/valid-${TODAY}.log \
        --overwrite \
        --keep-best \
        --quiet-translation \
        --enc-depth 6 \
        --dec-depth 6 \
        --transformer-heads 8 \
        --dim-emb 512 \
        --transformer-dim-ffn 2048 \
        --transformer-ffn-activation swish \
        --lr-warmup 16000 \
        --learn-rate 0.0003 \
        --lr-decay-inv-sqrt 16000 \
        --lr-report \
        --label-smoothing 0.1 \
        --dropout-src 0.1 \
        --dropout-trg 0.1 \
        --transformer-dropout 0.1 \
        --transformer-dropout-attention 0.1 \
        --transformer-dropout-ffn 0.1 \
        --sync-sgd \
        --optimizer-params 0.9 0.98 1e-09 \
        --clip-norm 0 \
        --beam-size 6 \
        --normalize 0.6 \
        --exponential-smoothing \
	--tied-embeddings-all
