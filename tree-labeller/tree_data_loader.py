#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 11:12:50 2021

@author: laurie

Loads CFG trees as dgl graphs.
"""

from abc import ABCMeta, abstractmethod
import dgl
import networkx as nx
from nltk.tree import Tree
import numpy as np
import _pickle as pickle
import re
import torch
from torch.nn.utils.rnn import pad_sequence
from vocab import Vocab


class Dataset(object):
    """Base class of nmtlab dataset.
    
    From https://github.com/zomux/nmtlab/blob/c047a530418f2bb2f4981071d92364af849aede5/nmtlab/dataset/base.py
    """
    __metaclass__ = ABCMeta

    def __init__(self, train_data=None, valid_data=None, batch_size=32, batch_type="sentence"):
        self._train_data = train_data
        self._valid_data = valid_data
        self._batch_size = batch_size
        self._batch_type = batch_type

    @abstractmethod
    def set_gpu_scope(self, scope_index, n_scopes):
        """Training a specific part of data for multigpu environment.
        """

    @abstractmethod
    def n_train_samples(self):
        """Return the number of training samples.
        """

    def n_train_batch(self):
        return int(self.n_train_samples() / self._batch_size)

    @abstractmethod
    def train_set(self):
        """
        Return an iterator of the training set.
        """

    @abstractmethod
    def valid_set(self):
        """
        Return an iterator of the validation set.
        """

    def raw_train_data(self):
        if hasattr(self, "_train_data"):
            return self._train_data
        else:
            return None

    def raw_valid_data(self):
        if hasattr(self, "_valid_data"):
            return self._valid_data
        else:
            return None

    def batch_size(self):
        return self._batch_size

    def batch_type(self):
        return self._batch_type

    def set_batch_size(self, batch_size):
        """Change the batch size of the dataset.
        """
        self._batch_size = batch_size


class TreeDataGenerator(object):
    """Creates a generator for tree data"""

    def __init__(self, cfg_path, treelstm_vocab_path, part_index=0,
                 part_num=1, cache_path=None, limit_datapoints=0,
                 limit_tree_depth=0):
        """

        Parameters
        ----------
        cfg_path : filepath
            Location of parsed tree
        treelstm_vocab_path : filepath
            Location of treeLSTM vocab
        part_index : int, optional
            In case of cached trees, the current part. The default is 0.
        part_num : int, optional
            In case of cached trees, the total number of parts. The default is 1.
        cache_path : filepath, optional
            Path to cached trees. The default is None.
        limit_datapoints : int, optional
            Limit the number of parsed lines to read to given limit. 
            The default is None.
        limit_tree_depth : int, optional
            Tree depth limit. The default is 0.

        Returns
        -------
        None.

        """
        if cache_path is not None:
            self._cache_path = f"{cache_path}.{part_index}in{part_num}"
        else:
            self._cache_path = None
        self._cfg_path = cfg_path
        self._cfg_lines = None
        self._part_index = part_index
        self._part_num = part_num
        self._limit_datapoints = limit_datapoints
        self._limit_tree_depth = limit_tree_depth
        self._vocab = Vocab(treelstm_vocab_path, picklable=True)
        self._trees = []

    def load(self):
        """
        Loads specified CFG parse into paired trees, then update self._trees

        Returns
        -------
        None.

        """
        # no desire to use OPTS
        # if not OPTS.smalldata and not OPTS.tinydata and self._cache_path is not None and os.path.exists(self._cache_path):
        #     print("loading cached trees part {} ...".format(self._part_index))
        #     self._trees = pickle.load(open(self._cache_path, "rb"))
        #     return
        self._cfg_lines = open(self._cfg_path).readlines()
        partition_size = int(len(self._cfg_lines) / self._part_num)
        self._cfg_lines = self._cfg_lines[self._part_index * partition_size:
                                          (self._part_index + 1) * partition_size]
        if self._limit_datapoints > 0:
            self._cfg_lines = self._cfg_lines[:self._limit_datapoints]
        print("building trees part {} ...".format(self._part_index))
        self._trees = self._build_batch_trees()  # returned paired trees
        if False and self._cache_path is not None:
            print("caching trees part {}...".format(self._part_index))
            pickle.dump(self._trees, open(self._cache_path, "wb"))

    def _parse_cfg_line(self, cfg_line):
        t = cfg_line.strip()
        # Replace leaves of the form (!), (,), with (! !), (, ,)
        t = re.sub(r"\((.)\)", r"(\1 \1)", t)
        # Replace leaves of the form (tag word root) with (tag word)
        t = re.sub(r"\(([^\s()]+) ([^\s()]+) [^\s()]+\)", r"(\1 \2)", t)
        try:
            tree = Tree.fromstring(t)
        except ValueError:
            tree = None
        return tree

    def _build_batch_trees(self):
        trees = []
        for line in self._cfg_lines:
            paired_tree = self.build_trees(line)
            trees.append(paired_tree)
        return trees

    def build_trees(self, cfg_line, device=1):
        parse = self._parse_cfg_line(cfg_line)
        if parse is None or not parse.leaves():
            return None
        enc_g = nx.DiGraph()  # directed networkx graphs
        dec_g = nx.DiGraph()
        failed = False

        def _rec_build(id_enc, id_dec, node, depth=0):
            if len(node) > 10:
                return
            if self._limit_tree_depth > 0 and depth >= self._limit_tree_depth:
                return
            # Skip all terminals
            all_terminals = True
            for child in node:
                if not isinstance(child[0], str) and not isinstance(
                        child[0], bytes):
                    all_terminals = False
                    break
            if all_terminals:
                return
            for j, child in enumerate(node):
                cid_enc = enc_g.number_of_nodes()
                cid_dec = dec_g.number_of_nodes()

                # Avoid leaf nodes
                tagid_enc = self._vocab.encode_token(
                    "{}_1".format(child.label()))
                tagid_dec = self._vocab.encode_token(
                    "{}_{}".format(node.label(), j+1))

                enc_g.add_node(cid_enc, x=tagid_enc, mask=0)
                dec_g.add_node(
                    cid_dec, x=tagid_dec, y=tagid_enc,
                    pos=j, mask=0, depth=depth+1)
                enc_g.add_edge(cid_enc, id_enc)
                dec_g.add_edge(id_dec, cid_dec)
                if not isinstance(child[0], str) and not isinstance(
                        child[0], bytes):
                    _rec_build(cid_enc, cid_dec, child, depth=depth + 1)

        if parse.label() == "ROOT" and len(parse) == 1:
            # Skip the root node
            parse = parse[0]
        root_tagid = self._vocab.encode_token("{}_1".format(parse.label()))
        enc_g.add_node(0, x=root_tagid, mask=1)
        dec_g.add_node(0, x=self._vocab.encode_token("ROOT_1"),
                       y=root_tagid, pos=0, mask=1, depth=0)
        _rec_build(0, 0, parse)
        if failed:
            return None
        enc_graph = dgl.from_networkx(enc_g, node_attrs=['x', 'mask'])
        dec_graph = dgl.from_networkx(
            dec_g, node_attrs=['x', 'y', 'pos', 'mask', 'depth'])
        return enc_graph.to(f"cuda:{device}"), dec_graph.to(f"cuda:{device}")

    def trees(self):
        return self._trees


class BilingualTreeDataLoader(Dataset):

    def __init__(self, src_path, cfg_path, src_vocab_path, 
                 treelstm_vocab_path, cache_path=None,
                 batch_size=64, max_tokens=80,
                 part_index=0, part_num=1,
                 load_data=True,
                 truncate=None,
                 limit_datapoints=0,
                 limit_tree_depth=0):
        self._max_tokens = max_tokens
        self._src_path = src_path
        self._src_vocab_path = src_vocab_path
        self._cfg_path = cfg_path
        self._treelstm_vocab_path = treelstm_vocab_path
        self._src_vocab = Vocab(self._src_vocab_path)
        self._label_vocab = Vocab(self._treelstm_vocab_path)
        self._cache_path = cache_path
        self._truncate = truncate
        self._part_index = part_index
        self._part_num = part_num
        self._limit_datapoints = limit_datapoints
        self._limit_tree_depth = limit_tree_depth
        self._rand = np.random.RandomState(3)
        if load_data:
            train_data, valid_data = self._load_data()
        self._n_train_samples = len(train_data)
        super(BilingualTreeDataLoader, self).__init__(
            train_data=train_data, 
            valid_data=valid_data, 
            batch_size=batch_size)

    def _load_data(self):
        src_vocab = self._src_vocab
        src_lines = open(self._src_path).readlines()
        partition_size = int(len(src_lines) / self._part_num)
        src_lines = src_lines[self._part_index *
                              partition_size: (self._part_index + 1) * partition_size]
        treegen = TreeDataGenerator(self._cfg_path, self._treelstm_vocab_path,
                                    cache_path=self._cache_path,
                                    part_index=self._part_index, part_num=self._part_num,
                                    limit_datapoints=self._limit_datapoints,
                                    limit_tree_depth=self._limit_tree_depth)
        treegen.load()
        trees = treegen.trees()
        if self._limit_datapoints > 0:
            src_lines = src_lines[:self._limit_datapoints]
        data_pairs = []
        assert len(src_lines) == len(trees)
        for src, paired_tree in zip(src_lines, trees):
            if paired_tree is None:
                continue
            enc_tree, dec_tree = paired_tree
            src_ids = src_vocab.encode(src.strip().split())
            if len(src_ids) > self._max_tokens:
                continue
            data_pairs.append((src_ids, enc_tree, dec_tree))
        if self._truncate is not None:
            data_pairs = data_pairs[:self._truncate]
        if len(data_pairs) < len(src_lines):
            missing_num = len(src_lines) - len(data_pairs)
            extra_indexes = np.random.RandomState(3).choice(
                np.arange(len(data_pairs)), missing_num)
            extra_data = [data_pairs[i] for i in extra_indexes.tolist()]
            data_pairs.extend(extra_data)
        np.random.RandomState(3).shuffle(data_pairs)
        valid_data = data_pairs[:1000]
        train_data = data_pairs[1000:]
        return train_data, valid_data

    def set_gpu_scope(self, scope_index, n_scopes):
        self._batch_size = int(self._batch_size / n_scopes)

    def n_train_samples(self):
        return len(self._train_data)

    def train_set(self):
        self._rand.shuffle(self._train_data)
        return self._train_iterator()

    def _train_iterator(self):
        for i in range(self.n_train_batch()):
            samples = self._train_data[i *
                                       self._batch_size: (i + 1) * self._batch_size]
            yield self.batch(samples)

    def batch(self, samples):
        src_samples = [x[0] for x in samples]
        enc_trees = [x[1] for x in samples]
        dec_trees = [x[2] for x in samples]
        src_batch = pad_sequence([torch.tensor(x)
                                  for x in src_samples], batch_first=True)
        enc_tree_batch = dgl.batch(enc_trees)
        dec_tree_batch = dgl.batch(dec_trees)
        return src_batch, enc_tree_batch, dec_tree_batch

    def valid_set(self):
        return self._valid_iterator()

    def _valid_iterator(self):
        n_batches = int(len(self._valid_data) / self._batch_size)
        for i in range(n_batches):
            samples = self._valid_data[i *
                                       self._batch_size: (i + 1) * self._batch_size]
            yield self.batch(samples)

    def src_vocab(self):
        return self._src_vocab

    def label_vocab(self):
        return self._label_vocab

    def yield_all_batches(self, batch_size=128):
        src_vocab = self._src_vocab
        src_lines = open(self._src_path).readlines()
        cfg_lines = open(self._cfg_path).readlines()
        assert len(src_lines) == len(cfg_lines)
        print("start to batch {} samples".format(len(src_lines)))
        treegen = TreeDataGenerator(self._cfg_path, self._treelstm_vocab_path,
                                    part_index=0, part_num=1,
                                    limit_tree_depth=self._limit_tree_depth)
        batch_samples = []
        batch_src_lines = []
        for src_line, cfg_line in zip(src_lines, cfg_lines):
            src_line = src_line.strip()
            cfg_line = cfg_line.strip()
            paired_tree = treegen.build_trees(cfg_line)
            if paired_tree is None:
                continue
            enc_tree, dec_tree = paired_tree
            src_ids = src_vocab.encode(src_line.split())
            if len(src_ids) > self._max_tokens:
                continue
            batch_samples.append((src_ids, enc_tree, dec_tree))
            batch_src_lines.append((src_line, cfg_line))
            if len(batch_samples) >= batch_size:
                src_batch, enc_tree_batch, dec_tree_batch = self.batch(
                    batch_samples)
                src_line_batch = [x[0] for x in batch_src_lines]
                cfg_line_batch = [x[1] for x in batch_src_lines]
                yield src_line_batch, cfg_line_batch, src_batch, enc_tree_batch, dec_tree_batch
                batch_src_lines.clear()
                batch_samples.clear()
        if len(batch_samples):
            src_batch, enc_tree_batch, dec_tree_batch = self.batch(
                batch_samples)
            src_line_batch = [x[0] for x in batch_src_lines]
            cfg_line_batch = [x[1] for x in batch_src_lines]
            yield src_line_batch, cfg_line_batch, src_batch, enc_tree_batch, dec_tree_batch
