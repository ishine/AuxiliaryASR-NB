import math
import torch
from torch import nn
from typing import Optional, Any
from torch import Tensor
import torch.nn.functional as F
import torchaudio
import torchaudio.functional as audio_F

import random
random.seed(0)


def _get_activation_fn(activ):
    if activ == 'relu':
        return nn.ReLU()
    elif activ == 'lrelu':
        return nn.LeakyReLU(0.2)
    elif activ == 'swish':
        return lambda x: x*torch.sigmoid(x)
    else:
        raise RuntimeError('Unexpected activ type %s, expected [relu, lrelu, swish]' % activ)

class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)


class ConvNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear', param=None):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain, param=param))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal

class CausualConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=1, dilation=1, bias=True, w_init_gain='linear', param=None):
        super(CausualConv, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2) * 2
        else:
            self.padding = padding * 2
        self.conv = nn.Conv1d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride,
                              padding=self.padding,
                              dilation=dilation,
                              bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain, param=param))

    def forward(self, x):
        x = self.conv(x)
        x = x[:, :, :-self.padding]
        return x

class CausualBlock(nn.Module):
    def __init__(self, hidden_dim, n_conv=3, dropout_p=0.2, activ='lrelu'):
        super(CausualBlock, self).__init__()
        self.blocks = nn.ModuleList([
            self._get_conv(hidden_dim, dilation=3**i, activ=activ, dropout_p=dropout_p)
            for i in range(n_conv)])

    def forward(self, x):
        for block in self.blocks:
            res = x
            x = block(x)
            x += res
        return x

    def _get_conv(self, hidden_dim, dilation, activ='lrelu', dropout_p=0.2):
        layers = [
            CausualConv(hidden_dim, hidden_dim, kernel_size=3, padding=dilation, dilation=dilation),
            _get_activation_fn(activ),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(p=dropout_p),
            CausualConv(hidden_dim, hidden_dim, kernel_size=3, padding=1, dilation=1),
            _get_activation_fn(activ),
            nn.Dropout(p=dropout_p)
        ]
        return nn.Sequential(*layers)

class ConvBlock(nn.Module):
    def __init__(self, hidden_dim, n_conv=3, dropout_p=0.2, activ='relu'):
        super().__init__()
        self._n_groups = 8
        self.blocks = nn.ModuleList([
            self._get_conv(hidden_dim, dilation=3**i, activ=activ, dropout_p=dropout_p)
            for i in range(n_conv)])


    def forward(self, x):
        for block in self.blocks:
            res = x
            x = block(x)
            x += res
        return x

    def _get_conv(self, hidden_dim, dilation, activ='relu', dropout_p=0.2):
        layers = [
            ConvNorm(hidden_dim, hidden_dim, kernel_size=3, padding=dilation, dilation=dilation),
            _get_activation_fn(activ),
            nn.GroupNorm(num_groups=self._n_groups, num_channels=hidden_dim),
            nn.Dropout(p=dropout_p),
            ConvNorm(hidden_dim, hidden_dim, kernel_size=3, padding=1, dilation=1),
            _get_activation_fn(activ),
            nn.Dropout(p=dropout_p)
        ]
        return nn.Sequential(*layers)

class LocationLayer(nn.Module):
    def __init__(self, attention_n_filters, attention_kernel_size,
                 attention_dim):
        super(LocationLayer, self).__init__()
        padding = int((attention_kernel_size - 1) / 2)
        self.location_conv = ConvNorm(2, attention_n_filters,
                                      kernel_size=attention_kernel_size,
                                      padding=padding, bias=False, stride=1,
                                      dilation=1)
        self.location_dense = LinearNorm(attention_n_filters, attention_dim,
                                         bias=False, w_init_gain='tanh')

    def forward(self, attention_weights_cat):
        processed_attention = self.location_conv(attention_weights_cat)
        processed_attention = processed_attention.transpose(1, 2)
        processed_attention = self.location_dense(processed_attention)
        return processed_attention


class Attention(nn.Module):
    def __init__(self, attention_rnn_dim, embedding_dim, attention_dim,
                 attention_location_n_filters, attention_location_kernel_size):
        super(Attention, self).__init__()
        self.query_layer = LinearNorm(attention_rnn_dim, attention_dim,
                                      bias=False, w_init_gain='tanh')
        self.memory_layer = LinearNorm(embedding_dim, attention_dim, bias=False,
                                       w_init_gain='tanh')
        self.v = LinearNorm(attention_dim, 1, bias=False)
        self.location_layer = LocationLayer(attention_location_n_filters,
                                            attention_location_kernel_size,
                                            attention_dim)
        self.score_mask_value = -float("inf")

    def get_alignment_energies(self, query, processed_memory,
                               attention_weights_cat):
        """
        PARAMS
        ------
        query: decoder output (batch, n_mel_channels * n_frames_per_step)
        processed_memory: processed encoder outputs (B, T_in, attention_dim)
        attention_weights_cat: cumulative and prev. att weights (B, 2, max_time)
        RETURNS
        -------
        alignment (batch, max_time)
        """

        processed_query = self.query_layer(query.unsqueeze(1))
        processed_attention_weights = self.location_layer(attention_weights_cat)
        energies = self.v(torch.tanh(
            processed_query + processed_attention_weights + processed_memory))

        energies = energies.squeeze(-1)
        return energies

    def forward(self, attention_hidden_state, memory, processed_memory,
                attention_weights_cat, mask):
        """
        PARAMS
        ------
        attention_hidden_state: attention rnn last output
        memory: encoder outputs
        processed_memory: processed encoder outputs
        attention_weights_cat: previous and cummulative attention weights
        mask: binary mask for padded data
        """
        alignment = self.get_alignment_energies(
            attention_hidden_state, processed_memory, attention_weights_cat)

        if mask is not None:
            alignment.data.masked_fill_(mask, self.score_mask_value)

        attention_weights = F.softmax(alignment, dim=1)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)

        return attention_context, attention_weights


class ForwardAttentionV2(nn.Module):
    def __init__(self, attention_rnn_dim, embedding_dim, attention_dim,
                 attention_location_n_filters, attention_location_kernel_size):
        super(ForwardAttentionV2, self).__init__()
        self.query_layer = LinearNorm(attention_rnn_dim, attention_dim,
                                      bias=False, w_init_gain='tanh')
        self.memory_layer = LinearNorm(embedding_dim, attention_dim, bias=False,
                                       w_init_gain='tanh')
        self.v = LinearNorm(attention_dim, 1, bias=False)
        self.location_layer = LocationLayer(attention_location_n_filters,
                                            attention_location_kernel_size,
                                            attention_dim)
        self.score_mask_value = -float(1e20)

    def get_alignment_energies(self, query, processed_memory,
                               attention_weights_cat):
        """
        PARAMS
        ------
        query: decoder output (batch, n_mel_channels * n_frames_per_step)
        processed_memory: processed encoder outputs (B, T_in, attention_dim)
        attention_weights_cat:  prev. and cumulative att weights (B, 2, max_time)
        RETURNS
        -------
        alignment (batch, max_time)
        """

        processed_query = self.query_layer(query.unsqueeze(1))
        processed_attention_weights = self.location_layer(attention_weights_cat)
        energies = self.v(torch.tanh(
            processed_query + processed_attention_weights + processed_memory))

        energies = energies.squeeze(-1)
        return energies

    def forward(self, attention_hidden_state, memory, processed_memory,
                attention_weights_cat, mask, log_alpha):
        """
        PARAMS
        ------
        attention_hidden_state: attention rnn last output
        memory: encoder outputs
        processed_memory: processed encoder outputs
        attention_weights_cat: previous and cummulative attention weights
        mask: binary mask for padded data
        """
        log_energy = self.get_alignment_energies(
            attention_hidden_state, processed_memory, attention_weights_cat)

        #log_energy =

        if mask is not None:
            log_energy.data.masked_fill_(mask, self.score_mask_value)

        #attention_weights = F.softmax(alignment, dim=1)

        #content_score = log_energy.unsqueeze(1) #[B, MAX_TIME] -> [B, 1, MAX_TIME]
        #log_alpha = log_alpha.unsqueeze(2) #[B, MAX_TIME] -> [B, MAX_TIME, 1]

        #log_total_score = log_alpha + content_score

        #previous_attention_weights = attention_weights_cat[:,0,:]

        log_alpha_shift_padded = []
        max_time = log_energy.size(1)
        for sft in range(2):
            shifted = log_alpha[:,:max_time-sft]
            shift_padded = F.pad(shifted, (sft,0), 'constant', self.score_mask_value)
            log_alpha_shift_padded.append(shift_padded.unsqueeze(2))

        biased = torch.logsumexp(torch.cat(log_alpha_shift_padded,2), 2)

        log_alpha_new = biased +  log_energy

        attention_weights =  F.softmax(log_alpha_new, dim=1)

        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)

        return attention_context, attention_weights, log_alpha_new


class PhaseShuffle2d(nn.Module):
    def __init__(self, n=2):
        super(PhaseShuffle2d, self).__init__()
        self.n = n
        self.random = random.Random(1)

    def forward(self, x, move=None):
        # x.size = (B, C, M, L)
        if move is None:
            move = self.random.randint(-self.n, self.n)

        if move == 0:
            return x
        else:
            left = x[:, :, :, :move]
            right = x[:, :, :, move:]
            shuffled = torch.cat([right, left], dim=3)
        return shuffled

class PhaseShuffle1d(nn.Module):
    def __init__(self, n=2):
        super(PhaseShuffle1d, self).__init__()
        self.n = n
        self.random = random.Random(1)

    def forward(self, x, move=None):
        # x.size = (B, C, M, L)
        if move is None:
            move = self.random.randint(-self.n, self.n)

        if move == 0:
            return x
        else:
            left = x[:, :,  :move]
            right = x[:, :, move:]
            shuffled = torch.cat([right, left], dim=2)

        return shuffled

class MFCC(nn.Module):
    def __init__(self, n_mfcc=40, n_mels=80):
        super(MFCC, self).__init__()
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.norm = 'ortho'
        dct_mat = audio_F.create_dct(self.n_mfcc, self.n_mels, self.norm)
        self.register_buffer('dct_mat', dct_mat)

    def forward(self, mel_specgram):
        if len(mel_specgram.shape) == 2:
            mel_specgram = mel_specgram.unsqueeze(0)
            unsqueezed = True
        else:
            unsqueezed = False
        # (channel, n_mels, time).tranpose(...) dot (n_mels, n_mfcc)
        # -> (channel, time, n_mfcc).tranpose(...)
        mfcc = torch.matmul(mel_specgram.transpose(1, 2), self.dct_mat).transpose(1, 2)

        # unpack batch
        if unsqueezed:
            mfcc = mfcc.squeeze(0)
        return mfcc



import torch
from torch import nn

class ConformerPreBlock(nn.Module):
    def __init__(self, dim_model, ff_multiplier=4, conv_kernel_size=31, dropout=0.1):
        super(ConformerPreBlock, self).__init__()

        # Feed-forward module (first half)
        self.ff1 = nn.Sequential(
            nn.LayerNorm(dim_model),
            nn.Linear(dim_model, ff_multiplier * dim_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_multiplier * dim_model, dim_model),
            nn.Dropout(dropout)
        )

        # Depthwise Convolution Module for Pre-block with kernel_size=31
        self.pre_conv_module = nn.Sequential(
            nn.Conv1d(dim_model, dim_model, kernel_size=conv_kernel_size, padding=conv_kernel_size // 2, groups=dim_model),
            nn.BatchNorm1d(dim_model),
            nn.SiLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Feed-forward module (first half)
        x = x + 0.5 * self.ff1(x)  # Residual connection with scaling (0.5)

        # Depthwise Convolution
        x = x.permute(1, 2, 0)  # (batch, channel, time)
        conv_output = self.pre_conv_module(x)
        x = conv_output.permute(2, 0, 1)  # (time, batch, channel)
        
        return x


class ConformerMainBlock(nn.Module):
    def __init__(self, dim_model, num_heads, ff_multiplier=4, conv_kernel_size=15, num_layers=1, dropout=0.1):
        super(ConformerMainBlock, self).__init__()

        # Create n-1 layers for the main body conformer block
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'self_attention': nn.MultiheadAttention(embed_dim=dim_model, num_heads=num_heads, dropout=dropout),
                'attention_norm': nn.LayerNorm(dim_model),
                'conv_module': nn.Sequential(
                    nn.Conv1d(dim_model, 2 * dim_model, kernel_size=1),
                    nn.GLU(dim=1),
                    nn.Conv1d(dim_model, dim_model, kernel_size=conv_kernel_size, padding=conv_kernel_size // 2, groups=dim_model),
                    nn.BatchNorm1d(dim_model),
                    nn.SiLU(),
                    nn.Conv1d(dim_model, dim_model, kernel_size=1),
                    nn.Dropout(dropout)
                ),
                'conv_norm': nn.LayerNorm(dim_model),
                'ff2': nn.Sequential(
                    nn.LayerNorm(dim_model),
                    nn.Linear(dim_model, ff_multiplier * dim_model),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(ff_multiplier * dim_model, dim_model),
                    nn.Dropout(dropout)
                )
            }) for _ in range(num_layers)
        ])

        # Final LayerNorm
        self.final_norm = nn.LayerNorm(dim_model)

    def forward(self, x, mask=None):
        # Iterate over each layer in the main block
        for layer in self.layers:
            # Self-Attention module
            attn_output, _ = layer['self_attention'](x, x, x, key_padding_mask=mask)
            x = x + layer['attention_norm'](attn_output)

            # Convolution module
            x = x.permute(1, 2, 0)  # (batch, channel, time)
            conv_output = layer['conv_module'](x)
            conv_output = conv_output.permute(2, 0, 1)  # (time, batch, channel)
            x = x.permute(2, 0, 1)  # (time, batch, channel)
            x = x + layer['conv_norm'](conv_output)

            # Feed-forward module (second half)
            x = x + 0.5 * layer['ff2'](x)

        return self.final_norm(x)


class ConformerBlock(nn.Module):
    def __init__(self, dim_model, num_heads, ff_multiplier=4, pre_conv_kernel_size=31, body_conv_kernel_size=15, num_layers=6, dropout=0.1):
        super(ConformerBlock, self).__init__()

        # Pre-block (one layer)
        self.pre_block = ConformerPreBlock(dim_model, ff_multiplier, conv_kernel_size=pre_conv_kernel_size, dropout=dropout)

        # Main-body block (n-1 layers)
        self.main_block = ConformerMainBlock(dim_model, num_heads, ff_multiplier, conv_kernel_size=body_conv_kernel_size, num_layers=num_layers - 1, dropout=dropout)

    def forward(self, x, mask=None):
        # Pre-block processing (1 layer)
        x = self.pre_block(x)

        # Main body processing (n-1 layers)
        x = self.main_block(x, mask=mask)

        return x

