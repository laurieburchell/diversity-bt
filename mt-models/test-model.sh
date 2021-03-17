#/bin/bash
set -eo pipefail

# script to test performance of trained model

MARIAN_DECODER="/home/laurie/apps/marian-dev/build/marian-decoder"
SRC="/mnt/startiger0/laurie/diversity/turkish/parallel_data/data/filtered_data/test.tr.pp"
TRG="/mnt/startiger0/laurie/diversity/turkish/parallel_data/data/filtered_data/test.en.pp"
MODEL_FOLDER="/mnt/startiger0/laurie/diversity/mt-models/baseline/models/bt-baseline-160321"
MODEL="bt-baseline.npz"
GPUS="0"
VOCAB="/mnt/startiger0/laurie/diversity/turkish/parallel_data/data/filtered_data/vocab.tren.spm"
OUTPUT="test-output.en"

$MARIAN_DECODER -m $MODEL_FOLDER/$MODEL -i $SRC -o $MODEL_FOLDER/$OUTPUT -d $GPUS -v $VOCAB $VOCAB

cat $MODEL_FOLDER/$OUTPUT | sacrebleu $TRG | tee $MODEL_FOLDER/test-bleu.txt
