#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/4/29
# @Author  : fangbing

import codecs
from utils import dict_helper as utils
import pickle
import torch
from transformers import *


#创建词袋
def makeVocabulary(filename, vocab, size):

    max_length = 0
    with codecs.open(filename, 'r', 'utf-8') as f:
        for sent in f.readlines():
            tokens = sent.strip().split()

            max_length = max(max_length, len(tokens))

            for word in tokens:
                vocab.add(word)

    print('Max length of %s = %d' % (filename, max_length))

    if size > 0:
        originalSize = vocab.size()
        vocab = vocab.prune(size)
        print('Created dictionary of size %d (pruned from %d)' %
              (vocab.size(), originalSize))
    return vocab



def build_src_tokenize(sentence, seq_length, tokenizer):
    tokens_a = tokenizer.tokenize(sentence)
     # Account for [CLS] and [SEP] with "- 2"
    if len(tokens_a) > seq_length - 2:
        tokens_a = tokens_a[0: (seq_length - 2)]

    tokens = []
    input_type_ids = []
    tokens.append("[CLS]")
    for token in tokens_a:
        tokens.append(token)
    tokens.append("[SEP]")

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_length=len(input_ids)
    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < seq_length:
        input_ids.append(0)
        input_mask.append(0)
    segment_ids = [0] * seq_length

    assert len(input_ids) == seq_length
    assert len(input_mask) == seq_length
    assert len(segment_ids) == seq_length

    return input_ids, input_length,input_mask, segment_ids,tokens_a

def build_tgt_tokenize(filename,max_vocab_size=0):
    vocab= utils.Dict([utils.PAD_WORD, utils.UNK_WORD, utils.BOS_WORD, utils.EOS_WORD])

    with codecs.open(filename, 'r', 'utf-8') as f:
        for sent in f.readlines():
            tokens = sent.strip().split()
            for word in tokens:
                vocab.add(word)
    if max_vocab_size!=0:
        vocab = vocab.prune(max_vocab_size)
    return vocab

def get_vocab(config):
    f = open(config.data_path + "labels.txt")
    lines = f.readlines()
    label_list = []
    for line in lines:
        label_list.append(str(line).replace('\n', ''))
    id2label = {i: label for i, label in enumerate(label_list)}
    label2id = {label: i for i, label in enumerate(label_list)}
    label_size = len(label_list)
    return id2label,label2id,label_size


def pre_data(src_path,tgt_path,vocab,tokenizer,config):
    src_f=codecs.open(src_path, 'r', 'utf-8')
    src_lines=src_f.readlines()

    tgt_f=codecs.open(tgt_path, 'r', 'utf-8')
    tgt_lines=tgt_f.readlines()

    data=[]
    for src,tgt in zip(src_lines,tgt_lines):
        src = src.replace('\n', '')
        src = ''.join(src.split(' '))

        max_seq_length=20
        src_id, src_length, src_mask,segment_ids, src_token = build_src_tokenize(src, config.max_input_length, tokenizer)


        tgt = tgt.replace('\n', '')
        tgt_token = list(tgt.split())

        if config.classifier=='BertCNN' or config.classifier=='BertRCNN':
            tgt_len=len(tgt_token)
            tgt_id=[0]*len(vocab.keys())
            for tgt in tgt_token:
                tgt_id[vocab.get(tgt)]=1
        if config.classifier=='BertSGM' or config.classifier=='BertSeq2Set':
            tgt_len = len(tgt_token)
            tgt_id = vocab.convertToIdx(tgt_token, utils.UNK_WORD, utils.BOS_WORD, utils.EOS_WORD)


        data.append((src_id,src_length,src_mask,segment_ids,src_token,tgt_id,tgt_len,tgt_token))

    return data

def pre_sgm_data(train_src, train_tgt, tgt_vocab ,src_vocab , config):
    src_f = codecs.open(train_src, 'r', 'utf-8')
    src_lines = src_f.readlines()

    tgt_f = codecs.open(train_tgt, 'r', 'utf-8')
    tgt_lines = tgt_f.readlines()

    data = []
    for src, tgt in zip(src_lines, tgt_lines):
        src = src.replace('\n', '')
        tokens_a = src.split(' ')

        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > config.max_input_length - 2:
            tokens_a = tokens_a[0: (config.max_input_length - 2)]

        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        for token in tokens_a:
            tokens.append(token)
        tokens.append("[SEP]")

        input_ids = src_vocab.convertToIdx(tokens, utils.UNK_WORD, utils.BOS_WORD, utils.EOS_WORD)
        input_length = len(input_ids)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < config.max_input_length:
            input_ids.append(0)
            input_mask.append(0)
        segment_ids = [0] * config.max_input_length


        tgt = tgt.replace('\n', '')
        tgt_token = list(tgt.split())

        tgt_len = len(tgt_token)
        tgt_id = tgt_vocab.convertToIdx(tgt_token, utils.UNK_WORD, utils.BOS_WORD, utils.EOS_WORD)

        # return input_ids, input_length,input_mask, segment_ids,tokens_a

        data.append((input_ids, input_length, input_mask, segment_ids, tokens_a, tgt_id, tgt_len, tgt_token))

    return data


class DatasetIterater(object):
    def __init__(self, dataset, batch_size,config):
        self.batch_size = batch_size
        self.dataset = dataset
        self.n_batches = len(dataset) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(dataset) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.classifier=config.classifier

    def _to_tensor(self, datas):
        src_id = torch.LongTensor([_[0] for _ in datas])
        src_len = torch.LongTensor([_[1] for _ in datas])
        src_mask = torch.LongTensor([_[2] for _ in datas])
        segment_ids = torch.LongTensor([_[3] for _ in datas])

        if self.classifier=='BertSGM' or self.classifier=='SGM' or config.classifier=='BertSeq2Set':
            tgt_len = [len(_[5]) for _ in datas]
            tgt_pad = torch.zeros(len(datas), max(tgt_len)).long()
            for i, s in enumerate(datas):
                tgt_id = s[5]
                end = tgt_len[i]
                tgt_pad[i, :end] = torch.LongTensor(tgt_id)[:end]
            tgt_id=tgt_pad
        if self.classifier == 'BertCNN' or self.classifier=='BertRCNN' :
            tgt_id = torch.LongTensor([_[5] for _ in datas])

        tgt_len = torch.LongTensor([_[6] for _ in datas])
        src_token = [_[4] for _ in datas]
        tgt_token = [_[7] for _ in datas]
        # return (src_id,src_len,src_mask,segment_ids,tgt_id)
        return (src_id,src_len,src_mask,segment_ids,src_token,tgt_id,tgt_len,tgt_token)

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.dataset[self.index * self.batch_size: len(self.dataset)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.dataset[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def load_bert_data(batch_size,type,config):
    print('-----------')
    tokenizer = BertTokenizer.from_pretrained(config.pretrian_path)
    train_src, train_tgt = config.data_path + 'ov_train.src' , config.data_path + 'ov_train.tgt'
    valid_src, valid_tgt = config.data_path + 'ov_valid.src' , config.data_path + 'ov_valid.tgt'
    test_src, test_tgt = config.data_path + 'test.src' , config.data_path + 'test.tgt'
    print('config.classifier:',config.classifier)
    if config.classifier=='BertCNN' or config.classifier=='BertRCNN':

        id2label,vocab,label_size=get_vocab(config)
        print(vocab)
        label2id=vocab

    if config.classifier == 'BertSGM' or config.classifier=='BertSeq2Set':
        vocab = build_tgt_tokenize(train_tgt)
        label_size = vocab.size()
        id2label=vocab.idxToLabel
        label2id=vocab.labelToIdx

    if type=='train' and config.pretrain=='Bert':
        train_dataset = pre_data(train_src, train_tgt, vocab, tokenizer,config)
        valid_dataset = pre_data(valid_src, valid_tgt, vocab, tokenizer,config)
        config['train_nbatchs'] = len(train_dataset) // batch_size
        config['valid_nbatchs'] = len(valid_dataset) // batch_size
        trainloader = DatasetIterater(train_dataset, batch_size,config)
        validloader = DatasetIterater(valid_dataset, batch_size,config)
        data = {'trainset': train_dataset, 'validset': valid_dataset,
                'trainloader': trainloader, 'validloader': validloader,
                'label_size': label_size, 'tgt_vocab': vocab,'label2id':label2id,'id2label':id2label}
        pickle.dump(data, open(config.save_data_path +str(config.classifier)+'_save_data.pkl', 'wb'))

    if type == 'test' and config.pretrain == 'Bert':
        test_dataset = pre_data(test_src, test_tgt, vocab, tokenizer,config)
        testloader = DatasetIterater(test_dataset, batch_size,config)
        config['test_nbatchs'] = len(test_dataset) // batch_size
        data={'testset': test_dataset, 'testloader': testloader, 'tgt_vocab': vocab, 'label_size': label_size,'label2id':label2id,'id2label':id2label}


    if config.classifier=='SGM':
        if type=='train':
            src_vocab = build_tgt_tokenize(train_src,config.src_vocab_size)
            tgt_vocab = build_tgt_tokenize(train_tgt,config.label_vocab_size)
            id2label = tgt_vocab.idxToLabel
            label2id = tgt_vocab.labelToIdx
            label_size = tgt_vocab.size()

            config['src_vocab_size']=src_vocab.size()

            train_dataset = pre_sgm_data(train_src, train_tgt, tgt_vocab ,src_vocab , config)
            valid_dataset = pre_sgm_data(valid_src, valid_tgt, tgt_vocab , src_vocab, config)

            trainloader = DatasetIterater(train_dataset, batch_size, config)
            validloader = DatasetIterater(valid_dataset, batch_size, config)
            config['train_nbatchs'] = len(train_dataset) // batch_size
            config['valid_nbatchs'] = len(valid_dataset) // batch_size
            data = {'trainset': train_dataset, 'validset': valid_dataset,
                    'trainloader': trainloader, 'validloader': validloader,
                    'label_size': label_size, 'tgt_vocab': tgt_vocab, 'src_vocab': src_vocab, 'label2id': label2id, 'id2label': id2label}
            pickle.dump(data, open(config.save_data_path +str(config.classifier)+'_save_data.pkl', 'wb'))
        if type=='test':
            data = pickle.load(open(config.save_data_path +str(config.classifier)+'_save_data.pkl', 'rb'))
            src_vocab=data['src_vocab']
            tgt_vocab=data['tgt_vocab']
            label_size= tgt_vocab.size()
            id2label = tgt_vocab.idxToLabel
            label2id = tgt_vocab.labelToIdx
            test_dataset = pre_sgm_data(test_src, test_tgt, tgt_vocab, src_vocab, config)

            testloader = DatasetIterater(test_dataset, batch_size, config)
            config['test_nbatchs'] = len(test_dataset) // batch_size
            data = {'testset': test_dataset, 'testloader': testloader, 'tgt_vocab': tgt_vocab, 'label_size': label_size,
                    'label2id': label2id, 'id2label': id2label}

    return data


