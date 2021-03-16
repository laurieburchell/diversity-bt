#!/bin/bash

# test run duplicating Shu et al. with my implementation
# we need to pretrain the model without any bottleneck
date=$(date '+%d%m%Y-%H%M')

python3 ../run.py --data_folder /home/laurie/projects/diversity/turkish/parallel_data/data \
        --model_folder /home/laurie/projects/diversity/turkish/tree2code \
        --model_name tree2code_tren_${date}_pretrain_ \
	--source_corpus train.tr.sp \
	--target_corpus train.en.sp \
	--source_vocab train.tr.sp.vocab \
	--target_trees train.en.parsed \
	--target_tree_vocab treelstm.vocab \
	--opt_limit_tree_depth 2 \
	--opt_limit_datapoints 800000 \
	--opt_codebits 0 \
	--train
