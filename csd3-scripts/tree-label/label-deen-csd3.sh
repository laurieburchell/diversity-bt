#!/bin/bash
set -eo pipefail

date=$(date '+%d%m%Y-%H%M%S')
DATA_FOLDER="/home/cs-burc1/projects/diversity-bt/german/parallel/qual75/split-data"

# spm encode German data
/home/cs-burc1/apps/marian-dev/build/spm_encode \
	--model=/home/cs-burc1/projects/diversity-bt/german/parallel/qual75/vocab.deen.spm \
	--input=$DATA_FOLDER/train.de$1 \
	--output=$DATA_FOLDER/train.de$1.spm

~/.conda/envs/tree/bin/python /home/cs-burc1/projects/diversity-bt/tree-labeller/run.py \
	--data_folder $DATA_FOLDER \
        --model_folder /home/cs-burc1/projects/diversity-bt/tree-labeller/data \
        --model_name wmt14.model \
	--source_corpus train.de$1.spm \
	--target_corpus train.en$1 \
	--source_vocab wmt14.de.sp.vocab \
	--target_trees train.en$1.cfg \
	--target_tree_vocab wmt14.treelstm.vocab \
	--output_name tree-output/train.en$1.tree-labels \
	--export_code --make_target
