#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/4/29
# @Author  : fangbing

from utils.utils import AttrDict


dct = {
    'data_path':'./data/',   #训练数据存储目录
    'save_data_path':'./data/save_data/', #数据预处理结果存储目录
    'model_save_path:':'./result/',  #模型预测结果存储目录
    'optim': 'bertadam',  # 优化函数
    'max_grad_norm': 10 , #梯度的最大范数，用于梯度裁剪
    'learning_rate_decay': 0.3,  #学习率衰减幅度
    'start_decay_at': 2,  #从第几个epoch开始衰减学习率
    'pretrian_path':'/data/textClassifier/Mutil_label_classification/Seq2Set',
    'max_input_length':20,    #训练文本最大长度

    "src_vocab_size": 20000,  # 输入词袋最多值
    "label_vocab_size": 20,   #label最大词袋长度

    'embedding_size': 256,    #编码维度
    'dropout': 0.5, # 0.5

    'cnn': {
        'num_filters': 256, # 256
        'filter_sizes': (2, 3, 4)
    },

    'rcnn': {
        'rnn_hidden': 256,
        'num_layers': 2,
        'kernel_size': 256,
        'dropout': 0.5
    },

    'dpcnn': {
        "num_filters": 256, # 256
        # "kernel_size": 256
    },


    'sgm':{
        'use_attention':True,
        'cell': 'lstm' ,    #编码单元模型，（可选：lstm、gru）
        'attention': 'luong_gate',  #attention方式，（可选：luong_attention、luong_gate、bahdanau_attention）
        'learning_rate': 2e-5, #学习率 0.0003
        'max_grad_norm': 10 , #梯度的最大范数，用于梯度裁剪
        'learning_rate_decay': 0.5,  #学习率衰减幅度
        'start_decay_at': 2,  #从第几个epoch开始衰减学习率
        'emb_size': 256 ,    #向量embedding长度
        'dec_num_layers': 3 , #encoder层数 3
        'enc_num_layers': 3 , #decoder层数 3
        'bidirectional': True, #是否使用双向lstm或双向grus
        'dropout': 0.1  ,      #dropout值
        'max_time_step': 10,
        'eval_interval': 200 , #每100步验证一次
        'save_interval': 500 , #每200步存储一次模型
        'unk': False,
        'schedule': False,
        'schesamp': False,
        'length_norm': True,
        'metrics': ['micro_f1'] , #度量指标
        'shared_vocab': False , #src 与 label 是否使用同一个词袋
        'beam_size': 3  , #beamsearch的数量，为1则和贪心策略一致，不做beamsearch
        'eval_time': 10,
        'mask': True,
        'global_emb': True , #是否使用前面t-1时刻的标签预测
        'tau': 0.1,
        'pool_size':0,
        'label_dict_file':'./data/label_sorted.json',
        'max_split':0   #为了内存效率，设置最大生成时间步

    },

    'seq2set':{
        'reward':'f1',
        'baseline':'self_critic',
        'max_time_step':3,
        'shared_vocab':False,
        'dropout':0.5,
        'dec_num_layers':2,
        'cell':'lstm',
        'enc_num_layers':2,
        'bidirectional':True,
        'score_fn':'',
        'emb_size': 256 ,
        'beam_size':1
    },

    'n_gpu':'0',  #使用gpu训练，''：表示使用cpu；
    'pretrain':'Bert', #Bert or ''
    'classifier':'BertSGM',
    'do_train':True,
    'do_test':True,
    'epochs':5,
    'resume_path':'',
    'train_batch_size':16,
    'eval_batch_size':16,
    'loss_scale':0,
    'weight_decay':0.01,
    'adam_epsilon':1e-8,
    'grad_clip':1.0,  #裁剪参数可迭代的梯度范数。
    'learning_rate':2e-5,
    'seed':42,
    'hidden_size':256,
    'gradient_accumulation_steps':1  #梯度积累的步骤


}

config = AttrDict(dct)


if __name__ == '__main__':
    print(config)