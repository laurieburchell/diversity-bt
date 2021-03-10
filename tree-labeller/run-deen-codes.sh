#!/bin/bash

# test run duplicating Shu et al. with my implementation
date=$(date '+%d-%m-%Y-%H:%M:%S')

python3 run.py --model_name tree2code_wmt14_${date}_ \
	--source_corpus wmt14.de.sp.filtered \
	--target_corpus wmt14.en.sp.filtered \
	--source_vocab wmt14.de.sp.vocab \
	--target_trees wmt14.en.cfg.filtered \
	--target_tree_vocab wmt14.treelstm.vocab \
	--opt_limit_tree_depth 2 \
	--opt_limit_datapoints 100000 \
	--opt_codebits 8 \
	--load_pretrain tree2code_wmt14_pretraincodebits-0_limit_datapoints-100000_limit_tree_depth-2 \
	--train
