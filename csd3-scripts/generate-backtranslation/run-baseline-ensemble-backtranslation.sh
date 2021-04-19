#!/bin/bash
set -eo pipefail

TODAY=`date '+%F_%H%M'`
INPUT="/rds/user/cs-burc1/hpc-work/datasets/turkish/mono-data/news.tr.pp"
OUTPUT_LOC="/rds/user/cs-burc1/hpc-work/datasets/turkish/backtranslation/baseline"
OUTPUT="baseline-bt-$TODAY.en"
MODEL1="/home/cs-burc1/projects/diversity-bt/turkish/mt-models/baseline-bt-tren-2021-03-27_0403/model.npz"
MODEL2="/home/cs-burc1/projects/diversity-bt/turkish/mt-models/baseline-bt-tren-2021-03-27_1503/model.npz"
MODEL3="/home/cs-burc1/projects/diversity-bt/turkish/mt-models/baseline-bt-tren-2021-03-28_0003/model.npz"
MODEL4="/home/cs-burc1/projects/diversity-bt/turkish/mt-models/baseline-bt-tren-2021-03-28_0803/model.npz"
VOCAB="/rds/user/cs-burc1/hpc-work/datasets/turkish/parallel-data/vocab.tren.spm"
MARIAN_DECODER="/home/cs-burc1/apps/marian-dev/build/marian-decoder"
SCRATCH_DISK="/local"
GPUS="0 1 2 3"


# check args are OK
if [ -f "${INPUT}" ]; then
    echo "found input text to translate ${INPUT}"
else
    echo "cannot find input text to translate. quitting."
    exit 1
fi

if [ -z "${OUTPUT_LOC}" ]; then
    echo "missing output location"
    exit 1
else
    mkdir -p ${OUTPUT_LOC}
    echo "output is at ${OUTPUT_LOC}"
fi

if [ -f "${VOCAB}" ]; then
    echo "found vocab ${VOCAB}"
else
    echo "cannot find vocab. quitting."
    exit 1
fi

echo "translating input"
$MARIAN_DECODER -m $MODEL1 $MODEL2 $MODEL3 $MODEL4  --devices $GPUS -v $VOCAB $VOCAB --beam-size 6 --n-best --input $INPUT  --output $OUTPUT_LOC/$OUTPUT --max-length 150 --max-length-crop --mini-batch 8 --maxi-batch 100 --maxi-batch-sort src --workspace 15000 --quiet-translation
