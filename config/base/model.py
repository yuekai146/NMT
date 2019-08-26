import copy
import math
import numpy as np

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

from common import config


class Encoder_Decoder(nn.Module):

    def __init__(self, encoder, decoder, src_emb, tgt_emb, generator):
        # A standard encoder decoder model
        super(Encoder_Decoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_emb = src_emb
        self.tgt_emb = tgt_emb
        self.generator = generator

        if config.share_encoder_decoder_embed:
            emb_params = []
            for k, v in self.tgt_emb.named_parameters():
                emb_params.append(v)
            assert len(emb_params) == 1
            self.generator.proj.weight = emb_params[0]

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


def clone(module, N):
    # Create N identical modules
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clone(layer, N)
        self.norm = Layer_Norm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        x = self.norm(x)
        return x


class Layer_Norm(nn.Module):

    def __init__(self, size, eps=1e-6):
        super(Layer_Norm, self).__init__()
        self.param_std = nn.Parameter(torch.ones(size))
        self.param_mean = nn.Parameter(torch.zeros(size))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        normalized_x = self.param_std * ( ( x - mean ) / ( std + self.eps ) ) + self.param_mean
        return normalized_x


class Sublayer_Connection(nn.Module):

    def __init__(self, size, dropout):
        super(Sublayer_Connection, self).__init__()
        self.norm = Layer_Norm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        # norm -> sublayer -> dropout -> residual add
        return x + self.dropout(sublayer(self.norm(x)))


class Encoder_Layer(nn.Module):

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(Encoder_Layer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayers = clone(Sublayer_Connection(size, dropout), 2)
        self.size = size

    def forward(self, x, x_mask):
        # Self attention sublayer
        x = self.sublayers[0](x, lambda x:self.self_attn(x, x, x, x_mask))
        x = self.sublayers[1](x, self.feed_forward)
        return x


class Decoder(nn.Module):

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clone(layer, N)
        self.norm = Layer_Norm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class Decoder_Layer(nn.Module):

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(Decoder_Layer, self).__init__()
        self.src_attn = src_attn
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.dropout = nn.Dropout(dropout)
        self.sublayers = clone(Sublayer_Connection(size, dropout), 3)
        self.size = size

    def forward(self, x, memory, src_mask, tgt_mask):
        x = self.sublayers[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayers[1](x, lambda x: self.src_attn(x, memory, memory, src_mask))
        x = self.sublayers[2](x, self.feed_forward)
        return x


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    attn_score = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    
    if mask is not None:
        attn_score = attn_score.masked_fill(mask == 0, -1e10)
    attn_score = F.softmax(attn_score, dim=-1)

    if dropout is not None:
        attn_score = dropout(attn_score)

    return torch.matmul(attn_score, value), attn_score


class Multi_Head_Attention(nn.Module):

    def __init__(self, d_model, h, dropout=0.1):
        super(Multi_Head_Attention, self).__init__()
        assert d_model % h == 0
        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h
        self.linears = clone(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        n_batches = query.size(0)
        
        query, key, value = \
                [l(x).view(n_batches, -1, self.h, self.d_k).transpose(1, 2) for l, x in zip(self.linears, (query, key, value))]
        output, self.attn = attention(query, key, value, mask, self.dropout)
        output = output.transpose(1, 2).contiguous().view(n_batches, -1, self.d_model)

        return self.linears[-1](output)


class Positionwise_Feed_Forward(nn.Module):

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(Positionwise_Feed_Forward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))


class Embeddings(nn.Module):

    def __init__(self, d_model, n_vocab):
        super(Embeddings, self).__init__()
        self.emb = nn.Embedding(n_vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.emb(x) * math.sqrt(self.d_model)


class Positional_Embeddings(nn.Module):

    def __init__(self, d_model, dropout, max_len=5000):
        super(Positional_Embeddings, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


class Label_Smoothing_Loss(nn.Module):

    def __init__(self, label_smoothing):
        super(Label_Smoothing_Loss, self).__init__()
        self.label_smoothing = label_smoothing

    def forward(self, logits, target, target_mask=None, reduce=True):
        # logits: bsz, length, n_vocab
        # target: bsz, length
        # mask: bsz, length
        target = target.unsqueeze(-1)
        nll_loss = -logits.gather(dim=-1, index=target)
        smooth_loss = -logits.sum(dim=-1, keepdim=True)
        if target_mask is not None:
            non_pad_mask = target_mask.eq(1)
            nll_loss = nll_loss[non_pad_mask]
            smooth_loss = smooth_loss[non_pad_mask]
            n_tokens = target_mask.sum()
        else:
            nll_loss = nll_loss.squeeze(-1)
            smooth_loss = smooth_loss.squeeze(-1)
            n_tokens = target.numel()

        if reduce:
            nll_loss = nll_loss.sum()
            smooth_loss = smooth_loss.sum()
        eps_i = self.label_smoothing / logits.size(-1)
        loss = ( 1. - self.label_smoothing ) * nll_loss + eps_i * smooth_loss
        
        return loss / n_tokens, nll_loss / n_tokens


def dummpy_input():
    # Used for testing network forward and backward
    lengths = np.random.randint(low=1, high=config.MAX_LEN, size=(config.BATCH_SIZE))
    src = []
    for l in lengths:
        src_sent = np.random.randint(low=config.N_SPECIAL_TOKENS, high=config.src_n_vocab, size=(l)).tolist()
        src_sent += np.zeros(config.MAX_LEN - l).tolist()
        src.append(src_sent)

    lengths = np.random.randint(low=1, high=config.MAX_LEN, size=(config.BATCH_SIZE))
    tgt = []
    for l in lengths:
        tgt_sent = np.random.randint(low=config.N_SPECIAL_TOKENS, high=config.tgt_n_vocab, size=(l)).tolist()
        tgt_sent += np.zeros(config.MAX_LEN - l).tolist()
        tgt.append(tgt_sent)

    src, tgt = torch.from_numpy(np.array(src)).long(), torch.from_numpy(np.array(tgt)).long()
    src_mask, tgt_mask = (src != 0), (tgt != 0)
    batch = {"src":src, "tgt":tgt, "src_mask":src_mask.unsqueeze(-2), "tgt_mask":tgt_mask.unsqueeze(-2)}
    if config.use_cuda:
        from utils import to_cuda
        batch = to_cuda(batch)

    return batch


def get():
    c = copy.deepcopy
    attn = Multi_Head_Attention(config.d_model, config.num_heads)
    ff = Positionwise_Feed_Forward(config.d_model, config.d_ff, config.dropout)
    position = Positional_Embeddings(config.d_model, config.dropout)
    net = Encoder_Decoder(
            Encoder(Encoder_Layer(config.d_model, c(attn), c(ff), config.dropout), config.encoder_num_layers),
            Decoder(Decoder_Layer(config.d_model, c(attn), c(attn), c(ff), config.dropout), config.decoder_num_layers),
            nn.Sequential(Embeddings(config.d_model, config.src_n_vocab), c(position)),
            nn.Sequential(Embeddings(config.d_model, config.tgt_n_vocab), c(position)),
            Generator(config.d_model, config.tgt_n_vocab)
            )
    for p in net.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    criterion = Label_Smoothing_Loss(config.label_smoothing)

    if config.use_cuda:
        net = net.cuda()
        criterion = criterion.cuda()
    return net, criterion


if __name__ == "__main__":
    net, criterion = get()
    """
    batch = dummpy_input()
    logits = net(**batch)
    loss, nll_loss = criterion(logits, batch['tgt'], batch['tgt_mask'].squeeze())
    loss.backward()
    print(nll_loss.item())
    """
    numel = 0
    for p in net.parameters():
        print(p.size())
        numel += p.numel()
    print(numel)
