#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 15:32:49 2021

@author: laurie

Generates vocabulary files for tree-LSTM syntax labeller

Input: SentencePiece-encoded text file
Output: pickled vocabulary file
"""
import vocab
import argparse
import pathlib

# take in arguments
parser = argparse.ArgumentParser()
parser.add_argument("input_file", type=str,
                    help="text file for generating vocab")
parser.add_argument("-o", "--output_file", type=str, 
                    help="output vocab file. Default: <input_file>.vocab")

args = parser.parse_args()

# assign file names and check they are valid
input_file = pathlib.Path(args.input_file)
if args.output_file:
    output_file = pathlib.Path(args.output_file)
else:
    output_file = pathlib.Path(args.input_file + '.vocab')
    
assert input_file.exists()

# build vocab and export
v = vocab.Vocab()
print('building vocab...')
v.build(input_file)
print(f'saving vocab to {output_file}')
v.save(output_file)
print('done')
