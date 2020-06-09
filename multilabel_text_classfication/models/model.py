#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/4/29
# @Author  : fangbing

import os
import torch
from models.bert_for_multi_label import BertCNNForMultiLabel,BertRCNNForMultiLabel
from models.textcnn import TextCNN
from models.sgm import SGM

class Classifier:

    def __init__(self, config):
        self.config=config
        self.choose_model = config["classifier"]
        self.choose_pretrain = config["pretrain"]


    def __call__(self, num_labels):
        print(num_labels)
        if self.choose_pretrain == "Bert":
            if self.choose_model == "BertCNN":
                model = BertCNNForMultiLabel(self.config)
            elif self.choose_model == "BertRCNN":
                model = BertRCNNForMultiLabel(self.config)
            elif self.choose_model == "BertSGM" :
                model = SGM(self.config)
            elif self.choose_model=='BertSeq2Set':
                model=Seq2Set(self.config)

        if self.choose_pretrain=='' and self.choose_model == "SGM":
             model = SGM(self.config)

        return model