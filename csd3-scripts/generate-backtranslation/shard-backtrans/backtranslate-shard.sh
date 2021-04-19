#!/bin/bash

# script to translate INPUT with MODEL and VOCAB, saving as OUTPUT

# locations on disk
INPUT=$1
OUTPUT=$2
MODEL=$3
VOCAB=$4

MARIAN_DECODER="/home/cs-burc1/apps/marian-dev/build/marian-decoder"
SCRATCH_DISK="/local"
GPUS="0"

echo "copying shard of translation data to $SCRATCH_DISK/input${SLURM_ARRAY_TASK_ID}.txt"
rsync -avzh --progress $INPUT $SCRATCH_DISK/input${SLURM_ARRAY_TASK_ID}.txt

echo "translating input with given model"
$MARIAN_DECODER -m $MODEL --devices $GPUS -v $VOCAB $VOCAB --beam-size 1 --n-best --input $SCRATCH_DISK/input${SLURM_ARRAY_TASK_ID}.txt  --output $SCRATCH_DISK/output${SLURM_ARRAY_TASK_ID}.txt --max-length 150 --max-length-crop --mini-batch 8 --maxi-batch 100 --maxi-batch-sort src --no-spm-decode --workspace 15000 --quiet-translation --output-sampling=true

echo "copying output back to $OUTPUT"
rsync -avzh $SCRATCH_DISK/output${SLURM_ARRAY_TASK_ID}.txt $OUTPUT
rm -f $SCRATCH_DISK/input${SLURM_ARRAY_TASK_ID}.txt
rm -f $SCRATCH_DISK/output${SLURM_ARRAY_TASK_ID}.txt
echo "done"
exit 0
