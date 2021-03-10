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
import os
import pathlib
import sys
import torch
from trainer import MTTrainer, SimpleScheduler
from tree_autoencoder import TreeAutoEncoder
from tree_data_loader import BilingualTreeDataLoader
from utils import OPTS
from vocab import Vocab



# paths to data - hardcoded test files for now. 
# need to check they are good filepaths
data_folder = pathlib.Path.cwd().joinpath('data')
model_folder = pathlib.Path.cwd().joinpath('models')
#src_sents = data_folder.joinpath('wmt14_deen_test.de.sp')
#tgt_sents = data_folder.joinpath('wmt14_deen_test.en')  # TODO: add in sp
#src_vocab = data_folder.joinpath('wmt14.de.sp.vocab')
#trg_tree = data_folder.joinpath('wmt14_deen_test.en.cfg.oneline')
#tree_lstm_vocab = data_folder.joinpath('wmt14.treelstm.vocab')  # pickled
#model_path = model_folder.joinpath('tree-codes-wmt14-test.pt')


# constants
n_valid_per_epoch = 4
part_index = 0 # this is multi-GPU horovod stuff that I'm leaving out
part_num = 1
gpu_num = 1
batch_size = 128 * gpu_num

##############################################################################

# command-line arguments
# TODO: add more args
parser = argparse.ArgumentParser(
    description="Generate syntactic codes for target corpus")

parser.add_argument("--data_folder", type=str, default=data_folder,
                    help="Dir containing source and target data. Default: $(pwd)/data")
parser.add_argument("--model_folder", type=str, default=model_folder,
                    help="Dir to save models. Default: $(pwd)/models")
parser.add_argument("--model_name", type=str, 
                    help="Name for model files")

parser.add_argument("--source_corpus", type=str,
                    help="filepath of the source-side training corpus")
parser.add_argument("--target_corpus", type=str,
                    help="filepath of the target-side training corpus")
parser.add_argument("--source_vocab", type=str,
                    help="filepath of the source-side vocabulary")
parser.add_argument("--target_trees", type=str,
                    help="filepath of the target-side parsed trees")
parser.add_argument("--target_tree_vocab", type=str,
                    help="filepath of the target-side tree vocab")

parser.add_argument("--opt_limit_tree_depth", type=int, default=0,
                    help="limit the depth of the parse tree to consider. \
                        Default: 0")
parser.add_argument("--opt_limit_datapoints", type=int, default=-1,
                    help="limit the number of input datapoints (per GPU). \
                        Default: -1")
parser.add_argument("--opt_hidden_size", type=int, default=256,
                    help="Dimension of hidden layer in tree autoencoder \
                        Default: 256")
parser.add_argument("--opt_without_source", action="store_true",
                    help="Do not have input source sentence")
parser.add_argument("--opt_codebits", type=int, default=8,
                help="Number of bits for each discrete code. Default: 8")

parser.add_argument("--train", action="store_true",
                    help="Train the model")
parser.add_argument("--export_code", action="store_true", 
                    help="Export codes from trained model")
parser.add_argument("--make_target", action="store_true",
                    help="Merge codes with target sentences in training set")
parser.add_argument("--all", action="store_true",
                    help="Train model, export codes, and merge with target")

OPTS.parse(parser)

# turn ops into Paths and check they exist
OPTS.source_corpus = data_folder.joinpath(OPTS.source_corpus)
OPTS.target_corpus = data_folder.joinpath(OPTS.target_corpus)
OPTS.source_vocab = data_folder.joinpath(OPTS.source_vocab)
OPTS.target_trees = data_folder.joinpath(OPTS.target_trees)
OPTS.target_tree_vocab = data_folder.joinpath(OPTS.target_tree_vocab)

assert data_folder.exists()
assert OPTS.source_corpus.exists()
assert OPTS.target_corpus.exists()
assert OPTS.source_vocab.exists()
assert OPTS.target_trees.exists()
assert OPTS.target_tree_vocab.exists()

# create folder for the models to live in
model_folder.mkdir(parents=True, exist_ok=True)
model_path = model_folder.joinpath(OPTS.model_name)

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

    
# train the model
if OPTS.train or OPTS.all:
    print("training model...")
    # Training code
    scheduler = SimpleScheduler(30)
    weight_decay = 1e-5 if OPTS.weightdecay else 0
    optimizer = torch.optim.Adagrad(autoencoder.parameters(), lr=0.05)
    trainer = MTTrainer(autoencoder, dataset, optimizer, 
                        scheduler=scheduler)
    OPTS.trainer = trainer
    trainer.configure(
        save_path=model_path,
        n_valid_per_epoch=n_valid_per_epoch,
        criteria="loss",
    )
    # TODO: sort out whatever this pretrain thing is
    if OPTS.load_pretrain:
        import re
        pretrain_path = re.sub(r"_codebits-\d", "", model_path)
        pretrain_path = pretrain_path.replace("_load_pretrain", "")
        print("loading pretrained model in ", pretrain_path)
        autoencoder.load_pretrain(pretrain_path)
    else:
        scheduler = SimpleScheduler(10)
    if OPTS.resume:
        trainer.load()
    trainer.run()
    print("training finished.")

# add codes to data with model
if OPTS.export_code or OPTS.all:
    print("exporting codes...")
    assert model_path.exists()
    autoencoder.load(model_path)
    out_path = model_path.with_suffix('.codes')

    autoencoder.train(False)
    if torch.cuda.is_available():
        autoencoder.cuda()
    c = 0
    c1 = 0
    with open(out_path, "w") as outf:
        print("code path", out_path)
        for batch in dataset.yield_all_batches(batch_size=512):
            src_lines, cfg_lines, src_batch, enc_tree, dec_tree = batch
            out = autoencoder(src_batch.cuda(), enc_tree, dec_tree, return_code=True)
            codes = out["codes"]
            for i in range(len(src_lines)):
                src = src_lines[i]
                cfg = cfg_lines[i]
                code = str(codes[i].int().cpu().numpy())
                outf.write("{}\t{}\t{}\n".format(src, cfg, code))
            outf.flush()
            c += len(src_lines)
            if c - c1 > 10000:
                sys.stdout.write(".")
                sys.stdout.flush()
                c1 = c
        sys.stdout.write("\n")
    print("codes exported.")


if OPTS.make_target or OPTS.all:
    export_path = model_path.with_suffix('.codes')
    out_path = model_path.with_suffix('.tgt')
    print("out path", out_path)
    export_map = {}
    for line in open(export_path):
        if len(line.strip().split("\t")) < 3:
            continue
        src, cfg, code = line.strip().rsplit("\t", maxsplit=2)
        code_str = " ".join(["<c{}>".format(int(c) + 1) for c in code.split()])
        export_map["{}\t{}".format(src, cfg)] = code_str
    with open(out_path, "w") as outf:
        src_path= OPTS.source_corpus
        tgt_path = OPTS.target_corpus
        cfg_path = OPTS.target_trees
        for src, tgt, cfg in zip(open(src_path), open(tgt_path), open(cfg_path)):
            key = "{}\t{}".format(src.strip(), cfg.strip())
            if key in export_map:
                outf.write("{} <eoc> {}\n".format(export_map[key], tgt.strip()))
            else:
                outf.write("\n")




