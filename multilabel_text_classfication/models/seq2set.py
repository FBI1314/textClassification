import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn import metrics
import numpy as np
from utils import dict_helper
from models.seq2set_encoder_decoder import rnn_encoder,rnn_decoder

from transformers import *



class Seq2Set(nn.Module):
    def __init__(self,config, encoder=None, decoder=None):
        super(Seq2Set,self).__init__()

        self.bert_model = None
        if encoder is not None:
            self.encoder = encoder
        else:
            if config.pretrain=='Bert':
                self.bert_model = BertModel.from_pretrained(config.pretrian_path)

            self.encoder = rnn_encoder(config,self.bert_model)
        tgt_embedding = self.encoder.embedding if config.seq2set.shared_vocab else None
        if decoder is not None:
            self.decoder = decoder
        else:
            self.decoder =rnn_decoder(config,embedding=tgt_embedding, score_fn=config.seq2set.score_fn)
        
        self.log_softmax = nn.LogSoftmax(dim=-1)
        if config.n_gpu!='':
            self.use_cuda = True
        self.config = config
        if config.use_cuda:
            self.criterion.cuda()
    def forward(self, src, src_mask,src_len, dec, targets,segment_ids):
        if self.bert_model is None:
            src = src.t()
        dec = dec.t()
        targets = targets.t()

        contexts, state = self.encoder(src, src_mask,segment_ids,src_len.tolist())
        outputs,final_state=self.decoder(dec,state,contexts)
        return outputs,targets

    def sample(self,src, src_mask, segment_ids,src_len):
        contexts,state=self.encoder(src,src_mask,segment_ids,src_len.tolist())

        bos = Variable(torch.ones(src.size(0)).long().fill_(dict_helper.BOS))
        if self.use_cuda:
            bos = bos.cuda()

        sample_ids, final_outputs = self.decoder.sample([bos], state, contexts)
        _, attns_weight = final_outputs
        alignments = attns_weight.max(2)[1]
        #targets = tgt[1:]

        return sample_ids.t(), alignments.t()

    def greedy_sample(self, src, src_mask,src_len, dec, targets,segment_ids):

        contexts,state=self.encoder(src,src_mask,segment_ids,src_len.tolist())

        bos = Variable(torch.ones(src.size(0)).long().fill_(dict_helper.BOS))
        if self.use_cuda:
            bos = bos.cuda()

        sample_ids, _ = self.decoder.sample([bos], state, contexts)
        return sample_ids


    def rl_sample(self,src, src_mask,src_len, dec, targets,segment_ids):
        contexts,state=self.encoder(src,src_mask,segment_ids,src_len.tolist())
        bos=Variable(torch.ones(src.size(0)).long().fill_(dict_helper.BOS))
        if self.use_cuda:
            bos=bos.cuda()
        inputs,sample_ids,probs=[],[],[]
        inputs+=[bos]
        for i in range(self.config.seq2set.max_time_step):
            output,state,attn_weights=self.decoder.sample_one(inputs[i],state,contexts)
            #multinomial（多项分布），作用是对input的每一行做n_samples次取值，输出的张量是每一次取值时input张量对应行的下标。如果有元素为0，那么在其他不为0的元素被取干净之前，这个元素是不会被取到的。

            predicted=F.softmax(output).multinomial(1) #[batch,1]
            one_hot=Variable(torch.zeros(output.size())).cuda()
            one_hot.scatter_(1,predicted.long(),1)
            prob=torch.masked_select(F.log_softmax(output),one_hot.type(torch.ByteTensor).cuda())
            predicted=predicted.squeeze(dim=1)
            inputs+=[predicted]
            sample_ids+=[predicted]
            probs+=[prob]
        sample_ids=torch.stack(sample_ids).squeeze()
        probs=torch.stack(probs).squeeze()
        return sample_ids,targets,probs
        
    def compute_reward(self,src, src_mask,src_len, dec, targets,segment_ids):
        #基于分布p采样的标签序列：sample_ids为预测的标签序列，probs为decoder输出进过mask_select后的结果，tgt为真实标签序列

        sample_ids,tgt,probs=self.rl_sample(src, src_mask,src_len, dec, targets,segment_ids)

        sample_ids=sample_ids.t().data.tolist()
        tgt=tgt.data.tolist()
        probs=probs.t()
        batch_size=probs.size(0)
        rewards=[]

        for y,y_hat in zip(sample_ids,tgt):
            #获取f1奖励值
            rewards.append(self.get_acc(y,y_hat))
        rewards=torch.tensor(rewards).unsqueeze(1).expand_as(probs)
        rewards=Variable(rewards).cuda()

        #self_critic介绍：https://zhuanlan.zhihu.com/p/58832418
        if self.config.seq2set.baseline=='self_critic':
            greedy_pre=self.greedy_sample(src, src_mask,src_len, dec, targets,segment_ids)
            greedy_pre=greedy_pre.t().data.tolist()
            baselines=[]
            for y,y_hat in zip(greedy_pre,tgt):
                baselines.append(self.get_acc(y,y_hat))
            baselines=torch.Tensor(baselines).unsqueeze(1).expand_as(probs)
            baselines=Variable(baselines).cuda()
            rewards=rewards-baselines
        
        loss=-(probs*rewards).sum()/batch_size
        return loss

    #计算f1、hamming_loss
    def get_acc(self,y,y_hat):
        y_true=np.zeros(103)
        y_pre=np.zeros(103)
        for i in y:
            if i==dict_helper.EOS:
                break
            else:
                if i>3:
                    y_true[i-4]=1
        for i in y_hat:
            if i==dict_helper.EOS:
                break
            else:
                if i>3:
                    y_pre[i-4]=1
        if self.config.seq2set.reward=='f1':
            r=metrics.f1_score(np.array([y_true]),np.array([y_pre]),average='micro')
        elif self.config.seq2set.reward=='hacc':
            r=1-metrics.hamming_loss(np.array([y_true]),np.array([y_pre]))
        elif self.config.seq2set.reward=='linear':
            f1=metrics.f1_score(np.array([y_true]),np.array([y_pre]),average='micro')
            hacc=1-metrics.hamming_loss(np.array([y_true]),np.array([y_pre]))
            r = 0.5*f1 + 0.5*hacc
        return r