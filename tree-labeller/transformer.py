#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 16:27:03 2021

@author: laurie

from https://github.com/zomux/nmtlab/blob/c047a530418f2bb2f4981071d92364af849aede5/nmtlab/modules/transformer_modules.py#L105
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

RESCALE_COEF = 1 / math.sqrt(2)


class TransformerEmbedding(nn.Embedding):
    """
    Rescale the embeddings.
    TODO: share the weight with pre-softmax linear transformation
    """

    def __init__(self, num_embeddings, embedding_dim, dropout_ratio=0.1):
        super(TransformerEmbedding, self).__init__(
            num_embeddings, embedding_dim)
        self.pos_layer = PositionalEmbedding(embedding_dim)
        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, x, start=None, positional_encoding=True):
        """
        Compute the embeddings with positional encoderi
        Args:
            x - input sequence ~ (batch, len)
            start - the begining position (option)
            positional_encoding - whether using positional encoding
        """
        embed = super(TransformerEmbedding, self).forward(x)
        embed = embed * math.sqrt(self.embedding_dim)
        if positional_encoding:
            if embed.dim() == 2:
                # Collapse one dimension of positional embedding
                pos_embed = self.pos_layer(embed.unsqueeze(1), start=start)
                pos_embed = pos_embed.squeeze(1)
            else:
                pos_embed = self.pos_layer(embed, start=start)
            embed += pos_embed
        return self.dropout(embed)
    

class PositionalEmbedding(nn.Module):
    """
    This function is stolen from The Annotated Transformer (same as openNMT implementation).
    http://nlp.seas.harvard.edu/2018/04/03/attention.html#embeddings-and-softmax
    """

    def __init__(self, size, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, size)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp((torch.arange(0, size, 2).float() *
                              -(math.log(10000.0) / size)).float())
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x, start=None):
        """
        Return 3d tensor with shape (1, len, size).
        """
        if start is None:
            start = 0
        if type(x) == int:
            length = x
        else:
            length = x.shape[1]
        return torch.autograd.Variable(
            self.pe[:, start:start + length], requires_grad=False)


def residual_connect(x, y, rescale=False):
    out = x + y
    if rescale:
        out *= RESCALE_COEF
    return out


def gelu(x):
    """
    ﻿Hendrycks, D., & Gimpel, K. (2016) Bridging Nonlinearities and Stochastic Regularizers with Gaussian Error Linear Units.
    """
    return 0.5 * x * (
        1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


