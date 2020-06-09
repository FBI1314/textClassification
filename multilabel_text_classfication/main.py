#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/4/29
# @Author  : fangbing

import codecs
import json
import torch
from utils.utils import seed_everything, init_logger, logger, AttrDict
from models.model import Classifier
from train.losses import BCEWithLogLoss
from train.metrics import AUC, AccuracyThresh, MultiLabelReport,F1Score
from config.basic_config import config
from predata.preprocess import load_bert_data
from train.trainer import Trainer
import numpy as np
from models.optims import Optim
from utils import dict_helper
import torch.nn as nn


def train(data,config):

    id2label=data['id2label']
    label_size=data['label_size']
    config['num_labels']=label_size

    model=Classifier(config)(num_labels=label_size)

    optimizer = Optim(config.optim, config.learning_rate, config.max_grad_norm,
                         lr_decay=config.learning_rate_decay, start_decay_at=config.start_decay_at)
    optimizer.set_parameters(model.parameters())

    if config.classifier=='BertCNN' or config.classifier=='BertRCNN' or config.classifier=='BertDPCNN' or config.classifier=='BertFC':
        trainer = Trainer(config,
            model=model,
            logger=logger,
            criterion=BCEWithLogLoss(),
            optimizer=optimizer,
            early_stopping=None,
            epoch_metrics=[AUC(average='micro', task_type='binary'),MultiLabelReport(id2label=id2label),F1Score(average='micro')])
    elif config.classifier=='BertSGM' or config.classifier=='SGM':
        criterion = nn.CrossEntropyLoss(ignore_index=dict_helper.PAD, reduction='none')
        if config.n_gpu!='':
            criterion.cuda()
        trainer = Trainer(config,
                          model=model,
                          logger=logger,
                          criterion=criterion,
                          optimizer=optimizer,
                          early_stopping=None,
                          epoch_metrics=[AUC(average='micro', task_type='binary'),
                                         F1Score(average='micro')])
    elif config.classifier=='BertSeq2Set':
         trainer = Trainer(config,
                          model=model,
                          logger=logger,
                          criterion=None,
                          optimizer=optimizer,
                          early_stopping=None,
                          epoch_metrics=[AUC(average='micro', task_type='binary'),F1Score(average='micro')])
                          
    trainer.train(data=data,seed=config.seed)




def test(config):
    print('start test!')
    data = load_bert_data(config.train_batch_size, 'test',config)
    data_loader=data['testloader']
    id2label=data['id2label']
    model = torch.load(config.model_save_path+str(config.classifier)+'_bestmodel.pth')
    model.cuda()
    outputs = []
    targets = []
    auc_metric=AUC(average='micro', task_type='binary')
    f1_metric=F1Score(average='micro')
    for batch in data_loader:
        input_ids,src_len, input_mask, segment_ids,src_token, label_ids,tgt_len,tgt_token = batch
        input_ids = input_ids.to(0)
        input_mask = input_mask.to(0)
        segment_ids = segment_ids.to(0)
        label_ids = label_ids.to(0)

        logits = model(input_ids, input_mask, segment_ids)
        outputs.append(logits.cpu().detach())
        targets.append(label_ids.cpu().detach())
    outputs = torch.cat(outputs, dim=0).cpu().detach()
    print(outputs)
    y_prob=outputs.sigmoid().data.cpu().numpy()

    y_pred=y_prob.tolist()
    f=open(config.model_save_path+str(config.classifier)+'_preresult.txt','a',encoding='utf-8')
    for pre in y_pred:
        prelabel = []
        for index,i in enumerate(pre):
            if i > 0.5:
                prelabel.append(id2label.get(index))
        if len(prelabel)==0:
            index=pre.index(max(pre))
            prelabel.append(id2label.get(index))
        f.write(' '.join(prelabel)+'\n')
    f.close()
    targets = torch.cat(targets, dim=0).cpu().detach()
    f1_metric(logits=outputs, target=targets)
    print('f1:', f1_metric.value())

    auc_metric(logits=outputs, target=targets)
    print('auc:', auc_metric.value())

def test_SGM(config):
    print('start test!')
    data = load_bert_data(config.train_batch_size, 'test',config)
    testloader=data['testloader']
    tgt_vocab = data['tgt_vocab']

    model = torch.load(config.model_save_path+str(config.classifier)+'_bestmodel.pth')
    model.cuda()
    model.eval()
    auc_metric=AUC(average='micro', task_type='binary')
    f1_metric=F1Score(average='micro')
    reference, candidate, source, alignments = [], [], [], []

    with codecs.open(config.sgm.label_dict_file, 'r', 'utf-8') as f:
        label_dict = json.load(f)

    for step, batch in enumerate(testloader):
        src, src_len, src_mask, segment_ids, original_src, tgt, tgt_len, original_tgt = batch
        src = src.to(0)
        tgt = tgt.to(0)
        src_len = src_len.to(0)
        if src_mask is not None:
            src_mask = src_mask.to(0)
        segment_ids = segment_ids.to(0)

        with torch.no_grad():
            if config.sgm.beam_size > 1 and (not config.sgm.global_emb):
                samples, alignment, _ = model.beam_sample(src,src_mask, segment_ids,src_len, beam_size=config.sgm.beam_size, eval_=True)
                candidate += [tgt_vocab.convertToLabels(s, dict_helper.EOS) for s in samples]

            else:
                samples, alignment = model.sample(src, src_mask,segment_ids,src_len)
                candidate += [tgt_vocab.convertToLabels(s.tolist(), dict_helper.EOS) for s in samples]
        source += original_src
        reference += original_tgt
        if alignment is not None:
            alignments += [align for align in alignment]

    f1 = open(config.model_save_path+str(config.classifier)+'_origin.txt', 'a', encoding='utf-8')
    f2 = open(config.model_save_path+str(config.classifier)+'_predict.txt', 'a', encoding='utf-8')
    for i in range(len(reference)):
        f1.write(" ".join(reference[i]) + '\n')
        f2.write(" ".join(candidate[i]) + '\n')
    f1.close()
    f2.close()
    if config.unk and config.attention != 'None':
        cands = []
        for s, c, align in zip(source, candidate, alignments):
            cand = []
            for word, idx in zip(c, align):
                if word == dict_helper.UNK_WORD and idx < len(s):
                    try:
                        cand.append(s[idx])
                    except:
                        cand.append(word)
                        print("%d %d\n" % (len(s), idx))
                else:
                    cand.append(word)
            cands.append(cand)
            if len(cand) == 0:
                print('Error!')
        candidate = cands

    def make_label(l, label_dict):
        length = len(label_dict)
        result = np.zeros(length)
        indices = [label_dict.get(label.strip().lower(), 0) for label in l]
        result[indices] = 1
        return result

    def prepare_label(reference, candidate, label_dict):
        reference = np.array([make_label(y, label_dict) for y in reference])
        candidate = np.array([make_label(y_pre, label_dict) for y_pre in candidate])
        return reference, candidate

    reference, candidate = prepare_label(reference, candidate, label_dict)

    outputs = torch.tensor(reference)
    targets = torch.tensor(candidate)


    f1_metric(logits=outputs, target=targets)
    print('f1:', f1_metric.value())

    auc_metric(logits=outputs, target=targets)
    print('auc:', auc_metric.value())

def main():
    print('config:',config)
    # 加载训练集与验证集

    if config.do_train:
        #训练
        data = load_bert_data(config.train_batch_size, 'train', config)
        config['tgt_vocab_size'] = data['label_size']
        train(data,config)

    if config.do_test:
        #加载测试集

        #测试
        if config.classifier=='BertSGM' or config.classifier=='SGM':
            test_SGM(config)
        if config.classifier=='BertCNN' or config.classifier=='BertRCNN':
            test(config)


main()

