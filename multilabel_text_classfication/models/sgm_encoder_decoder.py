#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/4/29
# @Author  : fangbing

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from models.attention import luong_attention,luong_attention,bahdanau_attention,maxout,luong_gate_attention

'''
编码类
'''
class rnn_encoder(nn.Module):

    def __init__(self, config, bert,embedding=None):
        super(rnn_encoder, self).__init__()

        self.config = config
        self.hidden_size = config.hidden_size
        self.pre_model=None
        if bert is not None:
            self.pre_model = bert
            for param in self.pre_model.parameters():
                param.requires_grad = True
            self.lstm = nn.LSTM(1024, config.hidden_size, num_layers=config.sgm.enc_num_layers,bidirectional=config.sgm.bidirectional, batch_first=True)
        else:
            self.embedding = embedding if embedding is not None else nn.Embedding(config.src_vocab_size, config.sgm.emb_size)
        if config.cell == 'gru':
            #input_size=config.emb_size
            self.rnn = nn.GRU(input_size=config.hidden_size, hidden_size=config.hidden_size,
                              num_layers=config.sgm.enc_num_layers, dropout=config.sgm.dropout,
                              bidirectional=config.sgm.bidirectional)
        else:
            # input_size=config.emb_size
            self.rnn = nn.LSTM(input_size=config.hidden_size, hidden_size=config.hidden_size,
                               num_layers=config.sgm.enc_num_layers, dropout=config.sgm.dropout,
                               bidirectional=config.sgm.bidirectional)

    def forward(self, inputs, input_mask,token_type_ids,lengths):
        if self.pre_model is not None:

            encoder_out, text_cls = self.pre_model(inputs,attention_mask=input_mask,token_type_ids=token_type_ids)
            outputs,state=self.lstm(encoder_out)

        else:
            encoder_out = pack(self.embedding(inputs), lengths)
            outputs, state = self.rnn(encoder_out)
            # print('1---state[0] size:', state[0].size())
            outputs = unpack(outputs)[0]

        if self.config.sgm.bidirectional:
            # outputs: [max_src_len, batch_size, hidden_size]
            #将双向lstm的输出进行相加，使得输出从[max_src_len, batch_size, 2*hidden_size]-》[max_src_len, batch_size, hidden_size]
            outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
            # print('2----outputs size:',outputs.size())
            if self.config.sgm.cell == 'gru':
                state = state[:self.config.sgm.dec_num_layers]
            else:
                state = (state[0][::2], state[1][::2])

        return outputs, state

'''
解码类
'''

class rnn_decoder(nn.Module):

    def __init__(self, config,embedding=None, use_attention=True):
        super(rnn_decoder, self).__init__()

        self.config = config
        self.hidden_size = config.hidden_size

        self.embedding = embedding if embedding is not None else nn.Embedding(config.tgt_vocab_size, config.sgm.emb_size)
        #config.emb_size
        input_size = 2 * config.hidden_size if config.sgm.global_emb else config.hidden_size

        if config.cell == 'gru':
            self.rnn = StackedGRU(input_size=input_size, hidden_size=config.hidden_size,
                                  num_layers=config.sgm.dec_num_layers, dropout=config.sgm.dropout)
        else:
            self.rnn = StackedLSTM(input_size=input_size, hidden_size=config.hidden_size,
                                   num_layers=config.sgm.dec_num_layers, dropout=config.sgm.dropout)

        self.linear = nn.Linear(config.hidden_size, config.tgt_vocab_size)

        if not use_attention or config.sgm.attention == 'None':
            self.attention = None
        elif config.sgm.attention == 'bahdanau':
            self.attention = bahdanau_attention(config.hidden_size, input_size)
        elif config.sgm.attention == 'luong':
            self.attention = luong_attention(config.hidden_size, input_size, config.sgm.pool_size)
        elif config.sgm.attention == 'luong_gate':
            self.attention = luong_gate_attention(config.hidden_size, input_size)
        
        self.dropout = nn.Dropout(config.sgm.dropout)
        
        if config.sgm.global_emb:
            self.ge_proj1 = nn.Linear(config.sgm.emb_size, config.sgm.emb_size)
            self.ge_proj2 = nn.Linear(config.sgm.emb_size, config.sgm.emb_size)
            self.softmax = nn.Softmax(dim=1)

    def forward(self, input, state, output=None, mask=None):

        #将decoder的输入进行编码

        embs = self.embedding(input)
        #考虑所有包含在yt-1中的有效信号
        if self.config.sgm.global_emb:
            if output is None:
                output = embs.new_zeros(embs.size(0), self.config.tgt_vocab_size)
            probs = self.softmax(output / self.config.sgm.tau)
            emb_avg = torch.matmul(probs, self.embedding.weight)
            H = torch.sigmoid(self.ge_proj1(embs) + self.ge_proj2(emb_avg))
            emb_glb = H * embs + (1 - H) * emb_avg         
            embs = torch.cat((embs, emb_glb), dim=-1)

        output, state = self.rnn(embs, state)
        # print('decoder outputs.size:',output.size())
        if self.attention is not None:
            if self.config.sgm.attention == 'luong_gate':
                output, attn_weights = self.attention(output)
            else:
                output, attn_weights = self.attention(output, embs)
        else:
            attn_weights = None
        output = self.compute_score(output)

        #如果当前时刻输出的标签结果在前面t-1时刻中有出现过，则赋予一个极小值，否则则赋予零向量。至于这个极小值，官方源码中是赋予了-9999999999。
        if self.config.sgm.mask and mask:
            mask = torch.stack(mask, dim=1).long()
            output.scatter_(dim=1, index=mask, value=-1e7)

        return output, state, attn_weights

    def compute_score(self, hiddens):
        scores = self.linear(hiddens)
        return scores


class StackedLSTM(nn.Module):
    def __init__(self, num_layers, input_size, hidden_size, dropout):
        super(StackedLSTM, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for _ in range(num_layers):
            lstm = nn.LSTMCell(input_size, hidden_size)
            self.layers.append(lstm)
            input_size = hidden_size

    def forward(self, input, hidden):
        h_0, c_0 = hidden
        h_1, c_1 = [], []
        for i, layer in enumerate(self.layers):
            h_1_i, c_1_i = layer(input, (h_0[i], c_0[i]))
            input = h_1_i
            if i + 1 != self.num_layers:
                input = self.dropout(input)
            h_1 += [h_1_i]
            c_1 += [c_1_i]

        h_1 = torch.stack(h_1)
        c_1 = torch.stack(c_1)

        return input, (h_1, c_1)


class StackedGRU(nn.Module):
    def __init__(self, num_layers, input_size, hidden_size, dropout):
        super(StackedGRU, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for _ in range(num_layers):
            self.layers.append(nn.GRUCell(input_size, hidden_size))
            input_size = hidden_size

    def forward(self, input, hidden):
        h_0 = hidden
        h_1 = []
        for i, layer in enumerate(self.layers):
            h_1_i = layer(input, h_0[i])
            input = h_1_i
            if i + 1 != self.num_layers:
                input = self.dropout(input)
            h_1 += [h_1_i]

        h_1 = torch.stack(h_1)

        return input, h_1
