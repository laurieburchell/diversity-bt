#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 10:42:50 2021

@author: laurie

Reimplementation of the syntax tree code labeller from Shu et al. (2019)
https://www.aclweb.org/anthology/P19-1177/

Generates an encoding of a target sentence based on its constituency parse.
At train time, the source side of the corpus is also an input in order to 
reduce the space of possible encodings.
"""

import argparse
import pathlib
import torch
from trainer import MTTrainer, SimpleScheduler
from tree_autoencoder import TreeAutoEncoder
from tree_data_loader import BilingualTreeDataLoader
from utils import OPTS



# paths to data - hardcoded test files for now. 
# need to check they are good filepaths
data_folder = pathlib.Path.cwd().joinpath('data')
src_sents = data_folder.joinpath('wmt14_deen_test.de.sp')
src_vocab = data_folder.joinpath('wmt14.de.sp.vocab')
trg_tree = data_folder.joinpath('wmt14_deen_test.en.cfg.oneline')
tree_lstm_vocab = data_folder.joinpath('wmt14.treelstm.vocab')  # pickled

# constants
n_valid_per_epoch = 4
part_index = 0 # this is multi-GPU horovod stuff that I'm leaving out for now
part_num = 1
gpu_num = 1
batch_size = 128 * gpu_num

##############################################################################

# command-line arguments
# TODO: add more args
parser = argparse.ArgumentParser(
    description="Generate syntactic codes for target corpus")
parser.add_argument("--source_corpus", default=src_sents, type=str,
                    help="filepath of the source-side training corpus")
parser.add_argument("--source_vocab", default=src_vocab, type=str,
                    help="filepath of the source-side vocabulary")
parser.add_argument("--target_trees", default=trg_tree, type=str,
                    help="filepath of the target-side parsed trees")
parser.add_argument("--target_tree_vocab", default=tree_lstm_vocab, type=str,
                    help="filepath of the target-side tree vocab")
parser.add_argument("--opt_limit_tree_depth", type=int, default=0,
                    help="limit the depth of the parse tree to consider")
parser.add_argument("--opt_limit_datapoints", type=int, default=-1,
                    help="limit the number of input datapoints (per GPU)")
parser.add_argument("--opt_hidden_size", type=int, default=256,
                    help="Dimension of hidden layer in tree autoencoder")
parser.add_argument("--opt_without_source", action="store_true",
                    help="Do not have input source sentence")
parser.add_argument("--opt_codebits", type=int, default=8,
                help="Number of bits for each discrete code")
parser.add_argument("--train", action="store_true", default=True,
                    help="Train the model")

OPTS.parse(parser)



# Define dataset
dataset = BilingualTreeDataLoader(
    src_path=OPTS.source_corpus,
    cfg_path=OPTS.target_trees,
    src_vocab_path=OPTS.source_vocab,
    treelstm_vocab_path=OPTS.target_tree_vocab,
    cache_path=None,
    batch_size=batch_size,
    part_index=part_index,
    part_num=part_num,
    max_tokens=60,
    limit_datapoints=OPTS.limit_datapoints,
    limit_tree_depth=OPTS.limit_tree_depth
)

# Load the tree autoencoder onto GPU
autoencoder = TreeAutoEncoder(dataset, 
                              hidden_size=OPTS.hidden_size, 
                              code_bits=OPTS.codebits, 
                              without_source=OPTS.without_source)
if torch.cuda.is_available():
    autoencoder.cuda()
    
print(autoencoder)
    
# train the model
if OPTS.train:
    # Training code
    scheduler = SimpleScheduler(30)
    weight_decay = 1e-5 if OPTS.weightdecay else 0
    optimizer = torch.optim.Adagrad(autoencoder.parameters(), lr=0.05)
    trainer = MTTrainer(autoencoder, dataset, optimizer, 
                        scheduler=scheduler)
    OPTS.trainer = trainer
    trainer.configure(
        save_path=OPTS.model_path,
        n_valid_per_epoch=n_valid_per_epoch,
        criteria="loss",
    )
    if OPTS.load_pretrain:
        import re
        pretrain_path = re.sub(r"_codebits-\d", "", OPTS.model_path)
        pretrain_path = pretrain_path.replace("_load_pretrain", "")
        # if is_root_node():  # horovod multi-gpu stuff
        #     print("loading pretrained model in ", pretrain_path)
        autoencoder.load_pretrain(pretrain_path)
    else:
        scheduler = SimpleScheduler(10)
    if OPTS.resume:
        trainer.load()
    trainer.run()


