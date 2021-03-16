#!/bin/bash

# test run duplicating Shu et al. with my implementation
# we need to pretrain the model without any bottleneck
date=$(date '+%d%m%Y-%H%M%S')

python3 ../run.py --data_folder /home/laurie/projects/diversity/turkish/parallel_data/data \
        --model_folder /home/laurie/projects/diversity/turkish/tree2code \
        --model_name tree2code_tren_15032021-131337_limit_datapoints-800000_limit_tree_depth-2 \
	--source_corpus dev.sp.tr \
	--target_corpus dev.sp.en \
	--source_vocab train.tr.sp.vocab \
	--target_trees dev.en.parsed \
	--target_tree_vocab treelstm.vocab \
	--export_code --make_target
