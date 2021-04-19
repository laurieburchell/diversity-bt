import sacrebleu
import spacy
import argparse
import subprocess
from tqdm import tqdm
import numpy as np
import random
import pickle


def self_BLEU(triple):
    """Calculate three-way BLEU between sentences in triple"""
    bleu0 = sacrebleu.corpus_bleu(
        triple[0], [[triple[1]], [triple[2]]], lowercase=True).score
    bleu1 = sacrebleu.corpus_bleu(
        triple[1], [[triple[0]], [triple[2]]], lowercase=True).score
    bleu2 = sacrebleu.corpus_bleu(
        triple[2], [[triple[0]], [triple[1]]], lowercase=True).score
    return bleu0, bleu1, bleu2

def calc_self_BLEUS(data):
    bleus = []
    for line in tqdm(data):
        b = self_BLEU(line)
        bleus.append(b)
    return bleus

def triples_to_pos(data):
    """converts triples of sentences to their PoS tag representation"""
    nlp = spacy.load('en_core_web_trf')
    pos_data = []
    for triple in tqdm(data):
        pos_triple = []
        for sent in triple:
            doc = nlp(sent)
            pos = [token.pos_ for token in doc]
            pos = ' '.join(pos)
            pos_triple.append(pos)
        pos_data.append(pos_triple)
    return pos_data

########################################################################

random.seed(2626)

parser = argparse.ArgumentParser(description="Calculate diversity metrics")
parser.add_argument('input_file', help="input file of sentence triples",
                    type=str)
parser.add_argument('sample_size', help="number of triples to sample",
                    type=int)
parser.add_argument('--output', help="output intermediate files as pickles",
                    action="store_true")

args = parser.parse_args()

# get length of input file
line_count = subprocess.run(
    ['wc', '-l', args.input_file], capture_output=True)
file_length = int(str(line_count.stdout).split(' ')[0].strip("b'"))

# make set of line number to sample
number_of_triples = file_length//3
lines_to_sample = np.array(
    random.sample(list(range(number_of_triples)), args.sample_size))*3
sample_set = set(lines_to_sample)  # fast lookup
sample_set.update(
    lines_to_sample+1, lines_to_sample+2)  # to get other triple members

print("loading data")
data = []
triple = []

with open(args.input_file, 'r') as f:
    for c, line in tqdm(enumerate(f)):
        if c not in sample_set:
            continue
        if c % 3 == 0:
            if triple:
                data.append(triple)
            triple = []
        triple.append(line.strip())
    data.append(triple)

print("calculating self-BLEU")
bleus = calc_self_BLEUS(data)

print("generating PoS tags")
pos_data = triples_to_pos(data)

print("calculating PoS self-BLEU")
pos_bleus = calc_self_BLEUS(pos_data)

print(f"mean self-BLEU is {np.mean(np.mean(bleus, axis=1)):.2f}")
print(f"mean std dev of self-BLEU is {np.mean(np.std(bleus, axis=1)):.2f}")
print(f"mean self-BLEU for PoS is {np.mean(np.mean(pos_bleus, axis=1)):.2f}")
print(f"mean std dev of self-BLEU for PoS is {np.mean(np.std(pos_bleus, axis=1)):.2f}")

if args.output:
    with open(args.input_file + '.lines', 'wb') as f:
        pickle.dump(sample_set, f, pickle.HIGHEST_PROTOCOL)
    with open(args.input_file + '.bleus', 'wb') as f:
        pickle.dump(bleus, f, pickle.HIGHEST_PROTOCOL)
    with open(args.input_file + '.pos_bleus', 'wb') as f:
        pickle.dump(pos_bleus, f, pickle.HIGHEST_PROTOCOL)      
    
