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


class TransformerEncoderLayer(nn.Module):

    def __init__(self, size, ff_size=None, n_att_head=8, dropout_ratio=0.1, relative_pos=False):
        super(TransformerEncoderLayer, self).__init__()
        if ff_size is None:
            ff_size = size * 4
        self.dropout = nn.Dropout(dropout_ratio)
        self.attention = MultiHeadAttention(
            size, n_att_head, dropout_ratio=dropout_ratio, relative_pos=relative_pos)
        self.ff_layer = TransformerFeedForward(
            size, ff_size, dropout_ratio=dropout_ratio)
        self.layer_norm1 = nn.LayerNorm(size)
        self.layer_norm2 = nn.LayerNorm(size)

    def forward(self, x, src_mask=None):
        # Attention layer
        y1 = self.layer_norm1(x)
        y1, _ = self.attention(y1, y1, y1, mask=src_mask)
        y1 = self.dropout(y1)
        y1 = residual_connect(y1, x)
        # Feed-forward layer
        y2 = self.layer_norm2(y1)
        y2 = self.ff_layer(y2)
        y2 = self.dropout(y2)
        y2 = residual_connect(y2, y1)
        return y2


class TransformerFeedForward(nn.Module):
    """The common feed-forward layer."""

    def __init__(self, size, hidden_size, dropout_ratio=0.1, activation="relu"):
        super(TransformerFeedForward, self).__init__()
        self.w_1 = nn.Linear(size, hidden_size)
        self.w_2 = nn.Linear(hidden_size, size)
        self.dropout = nn.Dropout(dropout_ratio)
        if activation == "relu":
            self._activate = F.relu
        elif activation == "gelu":
            self._activate = gelu
        else:
            raise NotImplementedError

    def forward(self, x):
        return self.w_2(self.dropout(self._activate(self.w_1(x))))


class MultiHeadAttention(nn.Module):
    """The implementation of multi-head attention.
    
    Following the original description in the transformer paper.
    """

    _RELATIVE_POS_CLIP = 2

    def __init__(self, out_size, num_head=8, hidden_size=None, additive=False, dropout_ratio=0, relative_pos=False):
        super(MultiHeadAttention, self).__init__()
        if hidden_size is None:
            hidden_size = out_size
        self._num_head = num_head
        self._hidden_size = hidden_size
        self._out_size = out_size
        self._additive = additive
        if relative_pos:
            self.relative_posmatrix = nn.Embedding(
                self._RELATIVE_POS_CLIP * 2 + 1, hidden_size)
        else:
            self.relative_posmatrix = None
        self._attention = KeyValAttention(
            scaling=True, dropout_ratio=dropout_ratio, )
        if additive:
            # Taken from RNMT+ paper
            raise NotImplementedError
        else:
            self.linear_Q = nn.Linear(out_size, hidden_size)
            self.linear_K = nn.Linear(out_size, hidden_size)
            self.linear_V = nn.Linear(out_size, hidden_size)
        self.linear_O = nn.Linear(hidden_size, out_size)


class KeyValAttention(nn.Module):

    def __init__(self, scaling=False, dropout_ratio=0):
        """Initialize a key-value attention class.
        Args:
            scaling - Whether normalize the attention weights by sqrt(size)
            dropout_ratio - The probability of dropout on the logits
        """
        super(KeyValAttention, self).__init__()
        self._scaling = scaling
        self._dropout = nn.Dropout(
            dropout_ratio) if dropout_ratio > 0 else None

    def forward_2d(self, query, keys, values, mask=None, additional_logits=None):
        """Compute attention for 2-dimensional queries (batch x hidden).
        """
        context_vector, weights = self.forward_3d(
            query.unsqueeze(-2), keys, values, mask, additional_logits)
        return context_vector.squeeze(-2), weights.squeeze(-2)

    def forward_3d(self, query, keys, values, mask=None, additional_logits=None):
        """Compute attention for 3-dimensional input (batch x step x hidden).
        """
        logits = torch.matmul(query, keys.transpose(-2, -1))
        if additional_logits is not None:
            logits += additional_logits
        if self._scaling:
            logits /= math.sqrt(query.shape[-1])
        if mask is not None:
            if self._dropout is not None:
                mask = self._dropout(mask)
            if mask.dim() < logits.dim():
                mask = mask.unsqueeze(-2)
            logits = logits.masked_fill(mask == 0, -1e3)
        elif self._dropout is not None:
            # Using dropout but no mask
            mask = self._dropout(logits.new_ones(logits.shape))
            logits = logits.masked_fill(mask == 0, -1e3)
        weights = F.softmax(logits, dim=-1)
        context_vector = torch.matmul(weights, values)
        return context_vector, weights

    def forward(self, query, keys, values, mask=None, additional_logits=None):
        """Compute the context vector with key value attention.
        
        Returns:
            context vector and attention weights.
        """
        if query.dim() == keys.dim() - 1:
            return self.forward_2d(query, keys, values, mask, additional_logits=additional_logits)
        else:
            return self.forward_3d(query, keys, values, mask, additional_logits=additional_logits)


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
