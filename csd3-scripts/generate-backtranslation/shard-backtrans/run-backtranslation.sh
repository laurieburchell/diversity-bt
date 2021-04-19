#!/bin/bash
set -eo pipefail

# usage: run-backtranslation.sh
# takes input file of text to translate, splits into batches, translates, then combines into final translation
INPUT="/home/cs-burc1/projects/diversity-bt/turkish/mono-data/news.tr.pp"
OUTPUT_LOC="/home/cs-burc1/projects/diversity-bt/turkish/backtranslation/sampling"
MODEL="/home/cs-burc1/projects/diversity-bt/turkish/mt-models/baseline/bt-baseline-160321/bt-baseline.npz.best-bleu.npz"
VOCAB="/home/cs-burc1/projects/diversity-bt/turkish/parallel-data/vocab.tren.spm"
SLURM_SCRIPT="/home/cs-burc1/projects/diversity-bt/csd3-scripts/generate-backtranslation/backtranslation.slurm"

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

if [ -f "${MODEL}" ]; then
    echo "found marian model ${MODEL}"
else
    echo "cannot find marian model. quitting."
    exit 1
fi

if [ -f "${VOCAB}" ]; then
    echo "found vocab ${VOCAB}"
else
    echo "cannot find vocab. quitting."
    exit 1
fi

if [ -f "${SLURM_SCRIPT}" ]; then
    echo "found slurm script ${SLURM_SCRIPT}"
else
    echo "cannot find slurm script. quitting."
    exit 1
fi

# split file into chunks of 30,000 lines
echo "generating split files in ${OUTPUT_LOC}/split-input"
mkdir -p ${OUTPUT_LOC}/split-input
split -d -l 30000 ${INPUT} ${OUTPUT_LOC}/split-input/input

# generate list of files to send to translate
echo "making list of shards to send to translate"
ls -d -1 ${OUTPUT_LOC}/split-input/* > ${OUTPUT_LOC}/split-file-list.txt

# send each file to be translated by calling bash script
echo "submitting slurm script with batched files"
NUM_FILES=`cat ${OUTPUT_LOC}/split-file-list.txt | wc -l`
sbatch --array=1-${NUM_FILES}%20 $SLURM_SCRIPT ${OUTPUT_LOC}/split-file-list.txt $OUTPUT_LOC $MODEL $VOCAB

echo "done!"

