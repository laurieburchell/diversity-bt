#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 14:42:09 2021

@author: laurie

Tree autoencoder architecture
https://github.com/zomux/tree2code/blob/master/lib_treeautoencoder.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.init

from tree_data_loader import BilingualTreeDataLoader
from tree_lstm import TreeLSTMCell, DecoderTreeLSTMCell
from semantic_hashing import SemanticHashing
from transformer import TransformerEmbedding  # TransformerEncoderLayer


class TreeAutoEncoder(nn.Module):

    def __init__(self, dataset, hidden_size=256,
                 code_bits=5, without_source=False, dropout_ratio=0.1):
        super(TreeAutoEncoder, self).__init__()
        assert isinstance(dataset, BilingualTreeDataLoader)
        self.hidden_size = hidden_size
        self._vocab_size = dataset.src_vocab().size()
        self._label_size = dataset.label_vocab().size()
        self._code_bits = code_bits
        self._without_source = without_source
        ff_size = self.hidden_size * 4

        # Encoder
        self.src_embed_layer = TransformerEmbedding(
            self._vocab_size, self.hidden_size, dropout_ratio=dropout_ratio)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_size, 
                                           nhead=8,
                                           dim_feedforward=ff_size)
        self.encoder_norm = nn.LayerNorm(self.hidden_size)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, 
                                                    num_layers=3,
                                                    norm=self.encoder_norm)

        self.label_embed_layer = nn.Embedding(
            self._label_size, self.hidden_size)
        self.enc_cell = TreeLSTMCell(hidden_size, hidden_size)
        self.dec_cell = DecoderTreeLSTMCell(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_ratio)
        self.logit_nn = nn.Linear(self.hidden_size, self._label_size)
        if code_bits > 0:
            self.semhash = SemanticHashing(hidden_size, bits=code_bits)
        else:
            self.semhash = None
        self.initialize_parameters()
        

    def initialize_parameters(self):
        """Initialize the parameters in the model."""
        # Initialize weights
        for param in self.parameters():
            shape = param.shape
            if len(shape) > 1:
                nn.init.xavier_uniform_(param)
        for module in self.modules():
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    def encode_source(self, src_seq, src_mask, meanpool=False):
        """
        

        Parameters
        ----------
        src_seq : list
            Encoded source sentence.
        src_mask : TYPE
            DESCRIPTION.
        meanpool : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        encoder_outputs : dict
            DESCRIPTION.

        """
        src_seq = src_seq.long()
        x = self.src_embed_layer(src_seq)
        # have to transpose mask due to matrix nonsense in MultiheadAttention
        encoder_states = self.transformer_encoder(x, 
                                                  src_key_padding_mask=src_mask.transpose(0,1))
        
        if meanpool:
            src_mask = src_mask.float()
            encoder_states = encoder_states * src_mask.unsqueeze(-1)
            encoder_states = encoder_states.sum(
                1) / (src_mask.sum(1).unsqueeze(-1) + 10e-8)

        encoder_outputs = {
            "encoder_states": encoder_states,
            "src_mask": src_mask
        }
        return encoder_outputs

        
    

    def forward(self, src, enc_tree, dec_tree, return_code=False, **kwargs):
        # Source encoding
        # TODO: get mask working
        src_mask = torch.eq(src, 0)
        encoder_outputs = self.encode_source(src, src_mask, meanpool=True)
        encoder_states = encoder_outputs["encoder_states"]
        
        
        # Tree encoding
        enc_x = enc_tree.ndata["x"].cuda()
        x_embeds = self.label_embed_layer(enc_x)
        enc_tree.ndata['iou'] = self.enc_cell.W_iou(self.dropout(x_embeds))
        enc_tree.ndata['h'] = torch.zeros(
            (enc_tree.number_of_nodes(), self.hidden_size)).cuda()
        enc_tree.ndata['c'] = torch.zeros(
            (enc_tree.number_of_nodes(), self.hidden_size)).cuda()
        enc_tree.ndata['mask'] = enc_tree.ndata['mask'].float().cuda()
        dgl.prop_nodes_topo(enc_tree,
                            self.enc_cell.message_func,
                            self.enc_cell.reduce_func,
                            apply_node_func=self.enc_cell.apply_node_func)
        # Obtain root representation
        root_mask = enc_tree.ndata["mask"].float().cuda()
        # root_idx = torch.arange(root_mask.shape[0])[root_mask > 0].cuda()
        root_h = self.dropout(enc_tree.ndata.pop("h")) * \
            root_mask.unsqueeze(-1)
        orig_h = root_h.clone()[root_mask > 0]
        partial_h = orig_h
        if self._without_source:
            partial_h += encoder_states

        # Discretization
        if self._code_bits > 0:
            if return_code:
                codes = self.semhash(partial_h, return_code=True)
                ret = {"codes": codes}
                return ret
            else:
                partial_h = self.semhash(partial_h)
            if not self._without_source:
                partial_h += encoder_states

        root_h[root_mask > 0] = partial_h
        # Tree decoding
        dec_x = dec_tree.ndata["x"].cuda()
        dec_embeds = self.label_embed_layer(dec_x)
        dec_tree.ndata['iou'] = self.dec_cell.W_iou(self.dropout(dec_embeds))
        dec_tree.ndata['h'] = root_h
        dec_tree.ndata['c'] = torch.zeros(
            (enc_tree.number_of_nodes(), self.hidden_size)).cuda()
        dec_tree.ndata['mask'] = dec_tree.ndata['mask'].float().cuda()
        dgl.prop_nodes_topo(dec_tree,
                            self.dec_cell.message_func,
                            self.dec_cell.reduce_func,
                            apply_node_func=self.dec_cell.apply_node_func)
        # Compute logits
        all_h = self.dropout(dec_tree.ndata.pop("h"))
        logits = self.logit_nn(all_h)
        logp = F.log_softmax(logits, 1)
        # Compute loss
        y_labels = dec_tree.ndata["y"].cuda()
        monitor = {}
        loss = F.nll_loss(logp, y_labels, reduction="mean")
        acc = (logits.argmax(1) == y_labels).float().mean()
        monitor["loss"] = loss
        monitor["label_accuracy"] = acc
        return monitor

    def load_pretrain(self, pretrain_path):
        first_param = next(self.parameters())
        device_str = str(first_param.device)
        pre_state_dict = torch.load(pretrain_path, map_location=device_str)[
            "model_state"]
        keys = list(pre_state_dict.keys())
        for key in keys:
            if "semhash" in key:
                pre_state_dict.pop(key)
        state_dict = self.state_dict()
        state_dict.update(pre_state_dict)
        self.load_state_dict(state_dict)

    def load(self, model_path):
        first_param = next(self.parameters())
        device_str = str(first_param.device)
        state_dict = torch.load(model_path, map_location=device_str)[
            "model_state"]
        self.load_state_dict(state_dict)
