#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 15:58:37 2021

@author: laurie

utility classes 
"""

from argparse import ArgumentParser
import networkx as nx
import matplotlib.pyplot as plt
import os
import sys
import torch
import torch.distributed as dist
import importlib
import numpy as np
from collections import Counter


class MapDict(dict):
    """
    Example:
    m = Map({'first_name': 'Eduardo'}, last_name='Pool', age=24, sports=['Soccer'])
    """

    def __init__(self, *args, **kwargs):
        super(MapDict, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.items():
                self[k] = v

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(MapDict, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(MapDict, self).__delitem__(key)
        del self.__dict__[key]

    def update(self, m):
        for k, v in m.items():
            self[k] = v

    def copy(self):
        return MapDict(self)


class GlobalOptions(MapDict):
    """A option class that generates model tags."""

    def parse(self, arg_parser):
        """Parse options with the arguments.
        
        Args:
            arg_parser (ArgumentParser): An instance of argparse.ArgumentParser.
        """
        args = arg_parser.parse_args()
        if "debug" in dir(args) and args.debug:
            self["debug"] = True
        model_tags = []
        test_tags = []
        # Gather all option tags
        for key in [k for k in dir(args)]:
            if key.startswith("opt_"):
                self[key.replace("opt_", "")] = getattr(args, key)
                if getattr(args, key) != arg_parser.get_default(key):
                    if type(getattr(args, key)) in [str, int, float]:
                        tok = "{}-{}".format(key.replace("opt_",
                                                         ""), getattr(args, key))
                    else:
                        tok = key.replace("opt_", "")
                    if not tok.startswith("T"):
                        model_tags.append(tok)
                    else:
                        test_tags.append(tok)
            else:
                self[key] = getattr(args, key)
        self["model_tag"] = "_".join(model_tags)
        self["result_tag"] = "_".join(model_tags + test_tags)
        # Create shortcuts for model path and result path
        if hasattr(args, "model_name") and getattr(args, "model_name") is not None:
            self["model_name"] = args.model_name + self["model_tag"]
        if hasattr(args, "result_name") and getattr(args, "result_name") is not None:
            self["result_name"] = args.result_name + self["result_tag"]
        if hasattr(args, "model_path") and getattr(args, "model_path") is not None:
            # assert "." in args.model_path
            # pieces = args.model_path.rsplit(".", 1)
            # self["model_path"] = "{}_{}.{}".format(
            #     pieces[0], self.model_tag, pieces[1])
            self["model_path"] = args.model_path
        if hasattr(args, "result_path") and getattr(args, "result_path") is not None:
            # assert "." in args.result_path
            # pieces = args.result_path.rsplit(".", 1)
            # self["result_path"] = "{}_{}.{}".format(
            #     pieces[0], self.result_tag, pieces[1])
            self["result_path"] = args.result_path
        # no horovod please
        # try:
        #     import horovod.torch as hvd
        #     hvd.init()
        #     if hvd.rank() == 0:
        #         print("[OPTS] Model tag:", self.model_tag)
        # except:
        #     print("[OPTS] Model tag:", self.model_tag)


def plot_tree(g):
    # this plot requires pygraphviz package
    pos = nx.nx_agraph.graphviz_layout(g, prog='dot')
    nx.draw(g, pos, with_labels=False, node_size=10,
            node_color=[[.5, .5, .5]], arrowsize=4)
    plt.show()


if "OPTS" not in globals():
    OPTS = GlobalOptions()


def bleu_stats(hypothesis, reference):
    yield len(hypothesis)
    yield len(reference)
    for n in range(1, 5):
        s_ngrams = Counter([tuple(hypothesis[i:i + n])
                            for i in range(len(hypothesis) + 1 - n)])
        r_ngrams = Counter([tuple(reference[i:i + n])
                            for i in range(len(reference) + 1 - n)])
        yield sum((s_ngrams & r_ngrams).values())
        yield max(len(hypothesis) + 1 - n, 0)


def smoothed_bleu(hypothesis, reference):
    stats = list(bleu_stats(hypothesis, reference))
    c, r = stats[:2]
    if c == 0:
        return 0
    log_bleu_prec = sum([np.log((1 + float(x)) / (1 + y))
                         for x, y in zip(stats[2::2], stats[3::2])]) / 4.
    return np.exp(min(0, 1 - float(r) / c) + log_bleu_prec) * 100


def execution_env():
    if not torch.cuda.is_available():
        return ""
    nsml_installed = importlib.util.find_spec("nsml") is not None
    hvd_installed = importlib.util.find_spec("horovod") is not None
    if nsml_installed:
        return "nsml"
    elif hvd_installed:
        raise SystemError("horovod is deprecated!")
        return "horovod"
    else:
        return ""


def world_size():
    env = execution_env()
    if env == "nsml":
        from nsml import GPU_NUM, PARALLEL_WORLD
        n_world = max(len(PARALLEL_WORLD), 1)
        return int(GPU_NUM) * n_world
    elif env == "horovod":
        import horovod.torch as hvd
        torch.init()
        return hvd.size()
    elif torch.cuda.is_available():
        return torch.cuda.device_count()
    else:
        return 1


def distributed_init(local_rank, local_size=1, master="127.0.0.1", port="12355"):
    if dist.is_nccl_available():
        backend = "nccl"
    else:
        backend = "gloo"
    if local_rank == 0:
        print("[DISTRIBUTED] using {} backend".format(backend))
        sys.stdout.flush()
    init_method = None
    node_rank = 0
    node_size = 1
    if execution_env() == "nsml":
        from nsml import PARALLEL_WORLD, PARALLEL_PORTS, MY_RANK
        if len(PARALLEL_WORLD) > 1:
            master = PARALLEL_WORLD[0]
            port = PARALLEL_PORTS[0]
            init_method = "tcp://{}:{}".format(master, port)
            node_rank = MY_RANK
            node_size = len(PARALLEL_WORLD)
    # print("[nmtlab] Backend {} is used for Data Distributed Parallel".format(backend))
    os.environ['MASTER_ADDR'] = master
    os.environ['MASTER_PORT'] = str(port)
    rank = node_rank * local_size + local_rank
    world_sz = node_size * local_size
    dist.init_process_group(
        backend, rank=rank, world_size=world_sz, init_method=init_method)
    OPTS.dist_local_rank = local_rank
    OPTS.dist_local_size = local_size


def global_rank():
    return node_rank() * local_size() + local_rank()


def global_size():
    return local_size() * node_size()


def local_rank():
    if "dist_local_rank" in OPTS and OPTS.dist_local_rank is not None:
        return OPTS.dist_local_rank
    else:
        return 0


def local_size():
    if "dist_local_size" in OPTS and OPTS.dist_local_size is not None:
        return OPTS.dist_local_size
    elif torch.cuda.is_available():
        return torch.cuda.device_count()
    else:
        return 1


def node_rank():
    node_rank = 0
    if execution_env() == "nsml":
        from nsml import PARALLEL_WORLD, MY_RANK
        if len(PARALLEL_WORLD) > 1:
            node_rank = MY_RANK
    return node_rank


def node_size():
    node_size = 1
    if execution_env() == "nsml":
        from nsml import PARALLEL_WORLD
        if len(PARALLEL_WORLD) > 1:
            node_size = len(PARALLEL_WORLD)
    return node_size


def distributed_cleanup():
    dist.destroy_process_group()
