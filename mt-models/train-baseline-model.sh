#!/bin/bash

TODAY=`date '+%F_%H%m'`
MARIAN="~/apps/marian-dev/build/marian"
GPUS="0 1 2 3"
WORKSPACE=7500
MODEL_DIR=""
MODEL=""
SRC="tr"
TRG="en"
TRAIN_SRC=""
TRAIN_TRG=""
DEV_SRC=""
DEV_TRG=""
VOCAB=""

$MARIAN \
        --devices $GPUS \
        --type transformer \
        --model $MODEL_DIR/$MODEL \
        --train-sets $TRAIN_SRC $TRAIN_TRG \
        --valid_sets $DEV_SRC $DEV_TRG \
        --vocabs $VOCAB $VOCAB \
        --valid-metrics ce-mean-words bleu perplexity \
        --valid-mini-batch 64 \
        --early-stopping 5 \
        -w $WORKSPACE \
        --mini-batch-fit \
        --maxi-batch 1000 \
        --valid-freq 5000 \
        --save-freq 5000 \
        --disp-freq 500 \
        --log $MODEL_DIR/train${TODAY}.log \
        --valid-log $MODEL_DIR/valid${TODAY}.log \
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
