#!/bin/bash

FILE_LIST=$1
NUM_FILES=`cat ${FILE_LIST} | wc -l`
MAX_PARALLEL_JOBS=20

sbatch --array=1-${NUM_FILES}%${MAX_PARALLEL_JOBS} submit-select-code-bt.slurm $FILE_LIST

