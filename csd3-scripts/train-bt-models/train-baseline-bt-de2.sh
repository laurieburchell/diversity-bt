#!/bin/bash

# train baseline bt model to generate n-best beam bt and sampled bt
# usage: script MODEL_NUMBER

set -eo pipefail

# check for one arg
if [ "$#" -ne 1 ]; then
	echo "need one int arg"
	exit 1
fi

# set other vars
TODAY=`date '+%F_%H%M'`
MARIAN="/home/cs-burc1/apps/marian-dev/build/marian"
GPUS="0 1 2 3"
WORKSPACE=12000
SRC="de"
TRG="en"
MODEL_DIR="/rds/user/cs-burc1/hpc-work/datasets/german/mt-models"
MODEL="baseline-bt-deen-$1"
DATA_DIR="/rds/user/cs-burc1/hpc-work/datasets/german/parallel/qual75"
TRAIN_SRC="train.$SRC"
TRAIN_TRG="train.$TRG"
DEV_SRC="wmt19.$SRC"
DEV_TRG="wmt19.$TRG"
VOCAB="vocab.deen.spm"
SEED=2626$1

# make model dir
echo "model dir is at $MODEL_DIR/$MODEL"
mkdir -p $MODEL_DIR/$MODEL

$MARIAN \
	--task transformer-big \
        --devices $GPUS \
        --model $MODEL_DIR/$MODEL/model.npz \
	--seed $SEED \
        --train-sets $DATA_DIR/$TRAIN_SRC $DATA_DIR/$TRAIN_TRG \
        --valid-sets $DATA_DIR/$DEV_SRC $DATA_DIR/$DEV_TRG \
        --vocabs $DATA_DIR/$VOCAB $DATA_DIR/$VOCAB \
	--valid-translation-output $MODEL_DIR/$MODEL/valid.output.txt \
        --valid-metrics ce-mean-words bleu-detok perplexity \
        --valid-mini-batch 8 \
	--valid-max-length 300 \
	--learn-rate 0.0003 \
	--optimizer-delay 2 \
        -w $WORKSPACE \
        --valid-freq 3000 \
        --disp-freq 100 \
        --log $MODEL_DIR/$MODEL/train-${TODAY}.log \
        --valid-log $MODEL_DIR/$MODEL/valid-${TODAY}.log \
        --overwrite \
        --keep-best 
