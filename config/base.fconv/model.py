# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
from common import config

from utils import Grad_Multiply

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder_Decoder(nn.Module):

    def __init__(self, encoder, decoder, src_emb, tgt_emb, generator):
        # A standard encoder decoder model
        super(Encoder_Decoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_emb = src_emb
        self.tgt_emb = tgt_emb
        self.generator = generator

    
    def forward(self, src, src_mask, tgt, tgt_mask):
        return self.generator(
                self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)
                )

    
    def encode(self, src, src_mask):
        return self.encoder(self.src_emb(src), src_mask)

    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_emb(tgt), memory, src_mask, tgt_mask)


class Generator(nn.Module):

    def __init__(self, d_model, n_vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, n_vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


def build_model():
    encoder = FConvEncoder(
            num_embeddings=config.encoder_num_embeddings,
            emb_dim=config.emb_dim,
            max_positions=config.src_max_positions,
            convolutions=config.encoder_convolutions,
            dropout=config.encoder_dropout
            )
    decoder = FConvDecoder(
            num_embeddings=config.decoder_num_embeddings,
            emb_dim=config.emb_dim,
            out_emb_dim=config.decoder_out_emb_dim,
            max_positions=config.trg_max_positions,
            convolutions=config.decoder_convolutions,
            attention=config.decoder_attention,
            dropout=config.decoder_dropout,
            )
    net = FConvModel(encoder, decoder)
    return net


class FConvModel(nn.Module):
    """
    A fully convolutional model, i.e. a convolutional encoder and a
    convolutional decoder, as described in `"Convolutional Sequence to Sequence
    Learning" (Gehring et al., 2017) <https://arxiv.org/abs/1705.03122>`_.

    Args:
        encoder (FConvEncoder): the encoder
        decoder (FConvDecoder): the decoder

    The Convolutional model provides the following named architectures and
    command-line arguments:

    .. argparse::
        :ref: fairseq.models.fconv_parser
        :prog:
    """

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src_tokens, src_mask, prev_output_tokens, trg_mask):
        encoder_out_x, encoder_out_y = self.encoder(src_tokens, src_mask)
        pred, attn_score = self.decoder(prev_output_tokens, encoder_out_x, encoder_out_y, src_mask, trg_mask)
        return pred, attn_score

    def max_decoder_positions(self):
        return self.decoder.max_positions()

    def get_normalized_probs(self, pred):
        return self.decoder.get_normalized_probs(pred)


class FConvEncoder(nn.Module):
    """
    Convolutional encoder consisting of `len(convolutions)` layers.

    Args:
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_dim (int, optional): embedding dimension
        embed_dict (str, optional): filename from which to load pre-trained
            embeddings
        max_positions (int, optional): maximum supported input sequence length
        convolutions (list, optional): the convolutional layer structure. Each
            list item `i` corresponds to convolutional layer `i`. Layers are
            given as ``(out_channels, kernel_width, [residual])``. Residual
            connections are added between layers when ``residual=1`` (which is
            the default behavior).
        dropout (float, optional): dropout to be applied before each conv layer
        normalization_constant (float, optional): multiplies the result of the
            residual block by sqrt(value)
        left_pad (bool, optional): whether the input is left-padded. Default:
            ``True``
    """

    def __init__(
            self, num_embeddings, emb_dim=512, max_positions=1024,
            convolutions=((512, 3),) * 20, dropout=0.1, left_pad=True,
    ):
        super().__init__()
        self.dropout = dropout
        self.left_pad = left_pad
        self.num_attention_layers = sum(config.decoder_attention)

        self.num_embeddings = num_embeddings
        self.pad_idx = config.pad_idx
        self.embed_tokens = Embedding(num_embeddings, emb_dim, self.pad_idx)
        
        self.embed_positions = PositionalEmbedding(
            max_positions,
            emb_dim,
            self.pad_idx,
            left_pad=self.left_pad,
        )

        convolutions = extend_conv_spec(convolutions)
        in_channels = convolutions[0][0]
        self.fc1 = Linear(emb_dim, in_channels, dropout=dropout)
        self.projections = nn.ModuleList()
        self.convolutions = nn.ModuleList()
        self.residuals = []

        layer_in_channels = [in_channels]
        for i, (out_channels, kernel_size, residual) in enumerate(convolutions):
            if residual == 0:
                residual_dim = out_channels
            else:
                residual_dim = layer_in_channels[-residual]
            self.projections.append(Linear(residual_dim, out_channels)
                                    if residual_dim != out_channels else None)
            assert kernel_size % 2 == 1
            padding = kernel_size // 2
            self.convolutions.append(
                Conv1D(in_channels, out_channels * 2, kernel_size,
                        dropout=dropout, padding=padding)
            )
            self.residuals.append(residual)
            in_channels = out_channels
            layer_in_channels.append(out_channels)
        self.fc2 = Linear(in_channels, emb_dim)
    
    
    def forward(self, src_tokens, src_mask):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (FloatTensor): lengths of each source sentence of shape
                `(batch, src_len)`

        Returns:
            dict:
                - **encoder_out** (tuple): a tuple with two elements, where the
                  first element is the last encoder layer's output and the
                  second element is the same quantity summed with the input
                  embedding (used for attention). The shape of both tensors is
                  `(batch, src_len, emb_dim)`.
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
        """
        # embed tokens and positions
        src_pos = torch.cumsum(src_mask, dim=1) * src_mask
        src_pos = src_pos.long()
        x = self.embed_tokens(src_tokens) + self.embed_positions(src_pos)
        x = F.dropout(x, p=self.dropout, training=self.training)
        input_embedding = x

        # project to size of convolution
        x = self.fc1(x)


        # B x T x C -> B x C x T
        x = x.transpose(1, 2)

        residuals = [x]
        # temporal convolutions
        for proj, conv, res_layer in zip(self.projections, self.convolutions, self.residuals):
            if res_layer > 0:
                residual = residuals[-res_layer]
                residual = residual if proj is None else proj(residual)
            else:
                residual = None
            x = x * src_mask.unsqueeze(1)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x)
            x = F.glu(x, dim=1)

            if residual is not None:
                x = (x + residual) * math.sqrt(0.5)
            residuals.append(x)

        # B x C x T -> B x T x C
        x = x.transpose(1, 2)

        # project back to size of embedding
        x = self.fc2(x)

        x = x * src_mask.unsqueeze(-1)

        # scale gradients (this only affects backward, not forward)
        x = Grad_Multiply.apply(x, 1.0 / (2.0 * self.num_attention_layers))

        # add output to input embedding for attention
        y = (x + input_embedding) * math.sqrt(0.5)

        return x, y

    
class AttentionLayer(nn.Module):
    
    def __init__(self, conv_channels, emb_dim):
        super().__init__()
        # projects from output of convolution to embedding dimension
        self.in_projection = Linear(conv_channels, emb_dim)
        # projects from embedding dimension to convolution size
        self.out_projection = Linear(emb_dim, conv_channels)

        self.bmm = torch.bmm

    
    def forward(self, x, target_embedding, encoder_out_x, encoder_out_y, src_mask):
        #param: x: batch_size, trg_len, x_channels
        #param: target_embedding: batch_size, trg_len, emb_dim
        #param: encoder_out_x: batch_size, src_len, emb_dim
        #param: encoder_out_y: batch_size, src_len, emb_dim
        #param: src_mask: batch_size, src_len
        residual = x

        # attention
        x = (self.in_projection(x) + target_embedding) * math.sqrt(0.5)
        x = self.bmm(x, encoder_out_x.transpose(1, 2))

        # don't attend over padding
        x = x.float().masked_fill(
            (1-src_mask).unsqueeze(1).byte(),
            float('-inf')
        ).type_as(x)  # FP16 support: cast to float and back

        # softmax over last dim
        sz = x.size()
        x = F.softmax(x.view(sz[0] * sz[1], sz[2]), dim=1)
        x = x.view(sz)
        attn_scores = x

        x = self.bmm(x, encoder_out_y)

        # scale attention output (respecting potentially different lengths)
        s = src_mask.type_as(x).sum(dim=1, keepdim=True)  # exclude padding
        s = s.unsqueeze(-1)
        x = x * (s * s.rsqrt())

        # project back
        x = (self.out_projection(x) + residual) * math.sqrt(0.5)
        return x, attn_scores


class FConvDecoder(nn.Module):
    """Convolutional decoder"""

    def __init__(
            self, num_embeddings, emb_dim=512, out_emb_dim=256,
            max_positions=1024, convolutions=((512, 3),) * 20, attention=True,
            dropout=0.1, share_embed=False, positional_embeddings=True,
            adaptive_softmax_cutoff=None, adaptive_softmax_dropout=0,
            left_pad=False,
    ):
        super().__init__()
        self.dropout = dropout
        self.left_pad = left_pad
        self.need_attn = True

        convolutions = extend_conv_spec(convolutions)
        in_channels = convolutions[0][0]
        if isinstance(attention, bool):
            # expand True into [True, True, ...] and do the same with False
            attention = [attention] * len(convolutions)
        if not isinstance(attention, list) or len(attention) != len(convolutions):
            raise ValueError('Attention is expected to be a list of booleans of '
                             'length equal to the number of layers.')

        pad_idx = config.pad_idx
        self.embed_tokens = Embedding(num_embeddings, emb_dim, pad_idx)
        self.embed_positions = PositionalEmbedding(
            max_positions,
            emb_dim,
            pad_idx,
            left_pad=self.left_pad,
        ) if positional_embeddings else None

        self.fc1 = Linear(emb_dim, in_channels, dropout=dropout)
        self.projections = nn.ModuleList()
        self.convolutions = nn.ModuleList()
        self.attention = nn.ModuleList()
        self.residuals = []

        layer_in_channels = [in_channels]
        for i, (out_channels, kernel_size, residual) in enumerate(convolutions):
            if residual == 0:
                residual_dim = out_channels
            else:
                residual_dim = layer_in_channels[-residual]
            self.projections.append(Linear(residual_dim, out_channels)
                                    if residual_dim != out_channels else None)
            self.convolutions.append(
                Conv1D(in_channels, out_channels * 2, kernel_size,
                                 padding=(kernel_size - 1), dropout=dropout)
            )
            self.attention.append(AttentionLayer(out_channels, emb_dim)
                                  if attention[i] else None)
            self.residuals.append(residual)
            in_channels = out_channels
            layer_in_channels.append(out_channels)

        self.fc2 = Linear(in_channels, out_emb_dim)
        if share_embed:
            assert out_emb_dim == emb_dim, \
                "Shared embed weights implies same dimensions " \
                " out_emb_dim={} vs emb_dim={}".format(out_emb_dim, emb_dim)
            self.fc3 = nn.Linear(out_emb_dim, num_embeddings)
            self.fc3.weight = self.embed_tokens.weight
        else:
            self.fc3 = Linear(out_emb_dim, num_embeddings, dropout=dropout)
    
    
    def forward(self, prev_output_tokens, encoder_out_x, encoder_out_y, src_mask, trg_mask=None):
        if self.embed_positions is not None:
            if trg_mask is None:
                trg_mask = 1.0 - prev_output_tokens.eq(config.pad_idx).float()
            trg_pos = torch.cumsum(trg_mask, dim=1) * trg_mask
            trg_pos = trg_pos.long()
            pos_embed = self.embed_positions(trg_pos)
        else:
            pos_embed = 0

        x = self.embed_tokens(prev_output_tokens)

        # embed tokens and combine with positional embeddings
        x += pos_embed
        x = F.dropout(x, p=self.dropout, training=self.training)
        target_embedding = x
        trg_len = x.shape[1]

        # project to size of convolution
        x = self.fc1(x)

        # B x T x C -> B x C x T
        x = x.transpose(1, 2)

        # temporal convolutions
        avg_attn_scores = None
        num_attn_layers = len(self.attention)
        residuals = [x]
        for proj, conv, attention, res_layer in zip(self.projections, self.convolutions, self.attention,
                                                    self.residuals):
            if res_layer > 0:
                residual = residuals[-res_layer]
                residual = residual if proj is None else proj(residual)
            else:
                residual = None

            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x)
            x = x[:, :, :trg_len]
            x = F.glu(x, dim=1)

            # attention
            if attention is not None:
                x = x.transpose(1, 2)

                x, attn_scores = attention(
                        x, target_embedding, encoder_out_x, encoder_out_y, src_mask
                        )

                if not self.training and self.need_attn:
                    attn_scores = attn_scores / num_attn_layers
                    if avg_attn_scores is None:
                        avg_attn_scores = attn_scores
                    else:
                        avg_attn_scores.add_(attn_scores)

                x = x.transpose(1, 2)

            # residual
            if residual is not None:
                x = (x + residual) * math.sqrt(0.5)
            residuals.append(x)

        # B x C x T -> B x T x C
        x = x.transpose(1, 2)

        # project back to size of vocabulary if not using adaptive softmax
        if self.fc2 is not None and self.fc3 is not None:
            x = self.fc2(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.fc3(x)

        return x, avg_attn_scores

    def max_positions(self):
        return self.embed_positions.num_embeddings if self.embed_positions is not None else float('inf')

    def get_normalized_probs(self, pred):
        return torch.log(F.softmax(pred, dim=-1))


def extend_conv_spec(convolutions):
    """
    Extends convolutional spec that is a list of tuples of 2 or 3 parameters
    (kernel size, dim size and optionally how many layers behind to look for residual)
    to default the residual propagation param if it is not specified
    """
    extended = []
    for spec in convolutions:
        if len(spec) == 3:
            extended.append(spec)
        elif len(spec) == 2:
            extended.append(spec + (1,))
        else:
            raise Exception('invalid number of parameters in convolution spec ' + str(spec) + '. expected 2 or 3')
    return tuple(extended)


def Embedding(num_embeddings, emb_dim, pad_idx):
    m = nn.Embedding(num_embeddings, emb_dim, padding_idx=pad_idx)
    nn.init.normal_(m.weight, 0, 0.1)
    nn.init.constant_(m.weight[pad_idx], 0)
    return m


def PositionalEmbedding(max_positions, emb_dim, pad_idx, left_pad):
    m = nn.Embedding(max_positions, emb_dim, padding_idx=pad_idx)
    nn.init.normal_(m.weight, 0, 0.1)
    nn.init.constant_(m.weight[pad_idx], 0)
    return m


def Linear(in_features, out_features, dropout=0):
    """Weight-normalized Linear layer (input: N x T x C)"""
    m = nn.Linear(in_features, out_features)
    nn.init.normal_(m.weight, mean=0, std=math.sqrt((1 - dropout) / in_features))
    nn.init.constant_(m.bias, 0)
    return nn.utils.weight_norm(m)


def Conv1D(in_channels, out_channels, kernel_size, dropout, **kwargs):
    m = nn.Conv1d(in_channels, out_channels, kernel_size, **kwargs)
    std = math.sqrt((4 * (1.0 - dropout)) / (m.kernel_size[0] * in_channels))
    nn.init.normal_(m.weight, mean=0, std=std)
    nn.init.constant_(m.bias, 0)
    return nn.utils.weight_norm(m)


def test():
    net = build_model()
    if config.cuda:
        net = net.cuda()
    _, epoch_itr = dataset.get()

    itr = epoch_itr.next_epoch_itr()
    itr = iterators.GroupedIterator(itr, config.update_freq)
    for i_batch, batch in enumerate(itr):
        batch = batch[0]
        if config.cuda:
            batch = to_cuda(batch)
        net_inputs = get_network_inputs(batch)
        break
    for i in range(20):
        net(**net_inputs)


if __name__ == "__main__":
    test()
