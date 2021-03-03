#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 11:37:17 2021

@author: laurie

from https://github.com/zomux/nmtlab/blob/master/nmtlab/utils/vocab.py
"""

import torchtext.vocab
import pickle
from collections import Counter, defaultdict

DEFAULT_SPECIAL_TOKENS = ["<null>", "<s>", "</s>", "UNK"]


class Vocab(torchtext.vocab.Vocab):

    def __init__(self, path=None, unk_token="UNK", picklable=False):
        self._unk_token = unk_token
        self.itos = []
        if picklable:
            self.stoi = {}
        else:
            self.stoi = defaultdict(lambda: 3)
        if path:
            self.load(path)

    def size(self):
        """
        Convenience function to generate vocabulary size

        Returns
        -------
        vocab_size : int
            Length of vocabulary

        """
        return len(self.itos)

    def initialize(self, special_tokens=None):
        if special_tokens is None:
            special_tokens = DEFAULT_SPECIAL_TOKENS
        self.itos = special_tokens
        self._build_vocab_map()

    def build(self, txt_path, limit=None, special_tokens=None,
              char_level=False, field=None, delim="\t"):
        """
        Builds a vocabulary based on whitespace-separated words,
        updating self.stoi

        Parameters
        ----------
        txt_path : file
            Path to text file for building vocabulary.
        limit : int, optional
            If not None, vocab length is set to limit. The default is None.
        special_tokens : list of str, optional
            Special tokens required by vocabulary. The default is None.
        char_level : bool, optional
            If True, builds vocab on char level. The default is False.
        field : int, optional
            Extract field at given index. The default is None.
        delim : str, optional
            Use given delimiter to separate fields. The default is "\t".

        Returns
        -------
        None.

        """
        vocab_counter = Counter()
        for line in open(txt_path):
            line = line.strip()
            if field is not None:
                line = line.split(delim)[field]
            if char_level:
                words = [w.encode("utf-8") for w in line.decode("utf-8")]
            else:
                words = line.split(" ")
            vocab_counter.update(words)
        if special_tokens is None:
            special_tokens = DEFAULT_SPECIAL_TOKENS
        if limit is not None:
            final_items = vocab_counter.most_common()[
                :limit - len(special_tokens)]
        else:
            final_items = vocab_counter.most_common()
        final_items.sort(key=lambda x: (-x[1], x[0]))
        final_words = [x[0] for x in final_items]
        self.itos = special_tokens + final_words
        self._build_vocab_map()

    def set_vocab(self, unique_tokens, special_tokens=True):
        if special_tokens:
            self.itos = DEFAULT_SPECIAL_TOKENS + unique_tokens
        else:
            self.itos = unique_tokens
        self._build_vocab_map()

    def add(self, token):
        if token not in self.stoi:
            self.itos.append(token)
            self.stoi[token] = self.itos.index(token)

    def save(self, path):
        pickle.dump(self.itos, open(path, "wb"))

    def load(self, path):
        """
        Loads vocabulary file, unpickling if necessary, then
        updates the vocabulary map.

        Parameters
        ----------
        path : filepath
            Path to vocabulary file.

        Returns
        -------
        None.

        """
        try:
            with open(path, "rb") as f:
                self.itos = pickle.load(f, encoding='utf-8')
        except pickle.UnpicklingError:
            self.itos = open(path).read().strip().split("\n")
        self._build_vocab_map()

    def _build_vocab_map(self):
        self.stoi.update({tok: i for i, tok in enumerate(self.itos)})

    def encode(self, tokens):
        return list(map(self.encode_token, tokens))

    def encode_token(self, token):
        if token in self.stoi:
            return self.stoi[token]
        else:
            return self.stoi[self._unk_token]

    def decode(self, indexes):
        return list(map(self.decode_token, indexes))

    def decode_token(self, index):
        return self.itos[index] if index < len(self.itos) else self._unk_token

    def contains(self, token):
        return token in self.stoi

    def get_list(self):
        return self.itos
