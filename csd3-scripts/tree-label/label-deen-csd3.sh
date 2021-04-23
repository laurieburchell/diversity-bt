#!/bin/bash
set -eo pipefail

date=$(date '+%d%m%Y-%H%M%S')
DATA_FOLDER="/home/cs-burc1/projects/diversity-bt/german/parallel/qual75/split-data/split-data"
inputNo=$1

# pad number
while [[ ${#inputNo} -lt 2 ]] ; do
    inputNo="0${inputNo}"
done

## spm encode German data
#/home/cs-burc1/apps/marian-dev/build/spm_encode \
#	--model=/home/cs-burc1/projects/diversity-bt/german/parallel/qual75/split-data/wmt20-de.model \
#	--input=$DATA_FOLDER/train.de$1 \
#	--output=$DATA_FOLDER/train.de$1.spm

~/.conda/envs/tree/bin/python /home/cs-burc1/projects/diversity-bt/tree-labeller/run.py \
	--data_folder $DATA_FOLDER \
        --model_folder /home/cs-burc1/projects/diversity-bt/tree-labeller/data \
        --model_name wmt14.model \
	--source_corpus train.de.spm${inputNo} \
	--target_corpus train.en${inputNo} \
	--source_vocab wmt14.de.sp.vocab \
	--target_trees train.en.cfg${inputNo} \
	--target_tree_vocab wmt14.treelstm.vocab \
	--output_name train.en.tree-labels${inputNo} \
	--export_code --make_target
