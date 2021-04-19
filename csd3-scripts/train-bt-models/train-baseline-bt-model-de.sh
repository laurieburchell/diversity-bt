#!/bin/bash

# train baseline bt model to generate n-best beam bt and sampled bt

set -eo pipefail

TODAY=`date '+%F_%H%M'`
MARIAN="/home/cs-burc1/apps/marian-dev/build/marian"
GPUS="0 1 2 3"
WORKSPACE=13000
SRC="de"
TRG="en"
MODEL_DIR="/rds/user/cs-burc1/hpc-work/datasets/german/mt-models"
MODEL="baseline-bt-deen-2021-04-06_1055"
DATA_DIR="/rds/user/cs-burc1/hpc-work/datasets/german/parallel"
TRAIN_SRC="clean.$SRC.gz"
TRAIN_TRG="clean.$TRG.gz"
DEV_SRC="wmt19-test.$SRC.clean"
DEV_TRG="wmt19-test.$TRG.clean"
VOCAB="$DATA_DIR/spm.deen.spm"

# make model dir
echo "model dir is at $MODEL_DIR/$MODEL"
mkdir -p $MODEL_DIR/$MODEL

$MARIAN \
        --devices $GPUS \
        --type transformer \
        --model $MODEL_DIR/$MODEL/model.npz \
        --train-sets $DATA_DIR/$TRAIN_SRC $DATA_DIR/$TRAIN_TRG \
        --valid-sets $DATA_DIR/$DEV_SRC $DATA_DIR/$DEV_TRG \
        --vocabs $VOCAB $VOCAB \
	--valid-translation-output valid.output.txt \
        --valid-metrics ce-mean-words bleu perplexity \
        --valid-mini-batch 64 \
        --early-stopping 5 \
        -w $WORKSPACE \
        --mini-batch-fit \
	--mini-batch 1000 \
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
        --transformer-dropout 0.1 \
        --sync-sgd \
        --optimizer-params 0.9 0.98 1e-09 \
        --clip-norm 0 \
        --beam-size 12 \
        --normalize 1 \
        --exponential-smoothing \
	--tied-embeddings-all
