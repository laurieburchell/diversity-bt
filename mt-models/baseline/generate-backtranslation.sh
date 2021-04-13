#/bin/bash
set -eo pipefail

# script to test performance of trained model

MARIAN_DECODER="/home/laurie/apps/marian-dev/build/marian-decoder"
SRC="/mnt/startiger0/laurie/diversity/turkish/mono_data/news.tr.pp"
MODEL_FOLDER="/mnt/startiger0/laurie/diversity/mt-models/baseline/models/bt-baseline-160321"
MODEL="bt-baseline.npz"
GPUS="0 1 2 3"
VOCAB="/mnt/startiger0/laurie/diversity/turkish/parallel_data/data/filtered_data/vocab.tren.spm"
OUTPUT_FOLDER="/mnt/startiger0/laurie/diversity/turkish/backtranslation-data/baseline"
OUTPUT="news.baseline-bt.en"

$MARIAN_DECODER -m $MODEL_FOLDER/$MODEL \
    -i $SRC -o $OUTPUT_FOLDER/$OUTPUT \
    -d $GPUS -v $VOCAB $VOCAB \
    -w 7500 -b 6 --normalize 0.6 \
    --mini-batch 64 --maxi-batch-sort src \
    --maxi-batch 100 --n-best

