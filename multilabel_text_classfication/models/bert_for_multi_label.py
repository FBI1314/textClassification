#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/4/29
# @Author  : fangbing

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import *

class BertCNNForMultiLabel(nn.Module):

    def __init__(self, config):
        super(BertCNNForMultiLabel, self).__init__()

        self.bert = BertModel.from_pretrained(config.pretrian_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.shot_dim = nn.Linear(1024, config.hidden_size)

        self.convs = nn.ModuleList([nn.Conv2d(1, config.cnn.num_filters, (k, config.hidden_size)) for k in config.cnn.filter_sizes])
        self.dropout = nn.Dropout(config.dropout)
        self.fc_cnn = nn.Linear(config.cnn.num_filters * len(config.cnn.filter_sizes), config.num_labels)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x))
        # print('x1.size:',x.size())
        x=x.squeeze(3)
        # print('x2.size:', x.size())
        x = F.max_pool1d(x, x.size(2))
        # print('x3.size:', x.size())
        x= x.squeeze(2)
        # print('4.size:', x.size())
        return x

    def forward(self, input_ids,
                attention_mask=None, token_type_ids=None, head_mask=None):
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            head_mask=head_mask)
        encoder_out, text_cls = outputs
        # print('encoder_out.size:',encoder_out.size())
        encoder_out=self.shot_dim(encoder_out)
        # print('encoder_out2.size:', encoder_out.size())
        out = encoder_out.unsqueeze(1)
        # print(out.size())
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc_cnn(out)
        return out


class BertRCNNForMultiLabel(nn.Module):

    def __init__(self, config):
        super(BertRCNNForMultiLabel, self).__init__()


        self.bert = BertModel.from_pretrained(config.pretrian_path)
        for param in self.bert.parameters():
            param.requires_grad = True

        self.shot_dim = nn.Linear(1024, config.hidden_size)
        self.lstm = nn.LSTM(config.hidden_size,
                            config.rcnn.rnn_hidden,
                            config.rcnn.num_layers,
                            bidirectional=True,
                            batch_first=True,
                            dropout=config.dropout)
        self.maxpool = nn.MaxPool1d(config.rcnn.kernel_size)
        self.fc = nn.Linear(config.rcnn.rnn_hidden * 2 +
                            config.hidden_size, config.tgt_vocab_size)
    def forward(self, input_ids,
                attention_mask=None, token_type_ids=None, head_mask=None):
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            head_mask=head_mask)
        encoder_out, text_cls = outputs
        encoder_out = self.shot_dim(encoder_out)
        out, _ = self.lstm(encoder_out)
        out = torch.cat((encoder_out, out), 2)
        out = F.relu(out)
        out = out.permute(0, 2, 1)
        out = self.maxpool(out).squeeze()
        out = self.fc(out)
        return out


