#!/bin/bash

# test run, now labelling data with trained model
date=$(date '+%d-%m-%Y-%H:%M:%S')

python3 run.py --model_name tree2code_wmt14_11-03-2021-14:16:56_ \
	--source_corpus wmt14_deen_test.de.sp \
	--target_corpus wmt-deen-test.en \
	--source_vocab wmt14.de.sp.vocab \
	--target_trees wmt14_deen_test.en.cfg.oneline \
	--target_tree_vocab wmt14.treelstm.vocab \
	--opt_limit_tree_depth 2 \
	--opt_limit_datapoints 800000 \
	--opt_codebits 8 \
        --export_code \
        --make_target \
        --device 0 \
        --output_name test-run2

