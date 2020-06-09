#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/4/29
# @Author  : fangbing

import torch
import numpy as np
from utils.utils import (model_device,summary, seed_everything, AverageMeter,progress_bar)
from torch.nn.utils import clip_grad_norm_
from utils import dict_helper
import codecs
import json

class Trainer(object):
    def __init__(self,
                 config,
                 model,
                 logger,
                 criterion,
                 optimizer,
                 early_stopping,
                 epoch_metrics
                 ):
        self.config=config
        self.start_epoch = 1
        self.global_step = 0
        self.n_gpu = config.n_gpu
        self.model = model
        self.epochs = config.epochs
        self.logger = logger
        self.grad_clip = config.grad_clip
        self.criterion = criterion
        self.optimizer = optimizer
        self.early_stopping = early_stopping
        self.epoch_metrics = epoch_metrics
        self.gradient_accumulation_steps = config.gradient_accumulation_steps
        self.model, self.device = model_device(n_gpu=self.n_gpu, model=self.model)



    def save_model(self,path, model, optim):
        model_state_dict = model.state_dict()
        checkpoints = {
            'model': model_state_dict,
             'optim': optim
            }
        torch.save(checkpoints, path)

    # 重置结果、预测值、评估值
    def epoch_reset(self):
        self.outputs = []
        self.targets = []
        self.result = {}
        for metric in self.epoch_metrics:
            metric.reset()


    def save_info(self, epoch, best):
        model_save = self.model.module if hasattr(
            self.model, 'module') else self.model
        state = {"model": model_save,
                 'epoch': epoch,
                 'best': best}
        return state

        # 如果val perf没有改善，或者我们达到start_decay_at极限，则衰减学习率

    def updateLearningRate(self, score, epoch):
        if self.start_decay_at is not None and epoch >= self.start_decay_at:
            self.start_decay = True

        if self.start_decay:
            self.lr = self.lr * self.lr_decay
            print("Decaying learning rate to %g" % self.lr)

        self.last_score = score
        self.optimizer.param_groups[0]['lr'] = self.lr

    def valid_epoch(self, data):
        self.epoch_reset()
        self.model.eval()
        with torch.no_grad():
            for step, batch in enumerate(data):
                input_ids,src_len, input_mask, segment_ids, src_token,label_ids,tgt_len,tgt_token = batch
                input_ids=input_ids.to(self.device)
                input_mask=input_mask.to(self.device)
                segment_ids=segment_ids.to(self.device)
                logits = self.model(input_ids, input_mask, segment_ids)
                self.outputs.append(logits.cpu().detach())
                self.targets.append(label_ids.cpu().detach())
                # pbar.batch_step(step=step, info={}, bar_type='Evaluating')
            self.outputs = torch.cat(self.outputs, dim=0).cpu().detach()
            self.targets = torch.cat(self.targets, dim=0).cpu().detach()
            loss = self.criterion(target=self.targets, output=self.outputs)
            self.result['valid_loss'] = loss.item()
            print("------------- valid result --------------")
            if self.epoch_metrics:
                for metric in self.epoch_metrics:
                    metric(logits=self.outputs, target=self.targets)
                    value = metric.value()
                    if value:
                        self.result[f'valid_{metric.name()}'] = value
            if 'cuda' in str(self.device):
                torch.cuda.empty_cache()
            return self.result

    def train_epoch(self, data):
        tr_loss = AverageMeter()
        self.epoch_reset()
        update=0
        for step,  batch in enumerate(data):
            self.model.train()

            input_ids,src_len, input_mask, segment_ids, src_token,label_ids,tgt_len,tgt_token = batch
            input_ids = input_ids.to(self.device)
            input_mask = input_mask.to(self.device)
            segment_ids = segment_ids.to(self.device)
            label_ids = label_ids.to(self.device)

            logits = self.model(input_ids, input_mask, segment_ids)
            loss = self.criterion(output=logits, target=label_ids)
            if len(self.n_gpu) >= 2:
                loss = loss.mean()
            if self.gradient_accumulation_steps > 1:
                loss = loss / self.gradient_accumulation_steps
            else:
                loss.backward()
                clip_grad_norm_(self.model.parameters(), self.grad_clip)

            if (step + 1) % self.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.global_step += 1
            tr_loss.update(loss.item(), n=1)

            #存储训练过程中的输出和目标值
            self.outputs.append(logits.cpu().detach())
            self.targets.append(label_ids.cpu().detach())
            update+=1
            progress_bar(update, self.config.train_nbatchs)

        print("\n------------- train result --------------")
        # epoch metric
        self.outputs = torch.cat(self.outputs, dim=0).cpu().detach()
        self.targets = torch.cat(self.targets, dim=0).cpu().detach()
        self.result['loss'] = tr_loss.avg
        if self.epoch_metrics:
            for metric in self.epoch_metrics:
                metric(logits=self.outputs, target=self.targets)
                value = metric.value()
                if value:
                    self.result[f'{metric.name()}'] = value
        if "cuda" in str(self.device):
            torch.cuda.empty_cache()
        return self.result

    def valid_bertsgm_epoch(self,validloader,tgt_vocab,label_dict):
        self.model.eval()
        self.epoch_reset()
        reference, candidate, source, alignments = [], [], [], []

        for step, batch in enumerate(validloader):
            src, src_len, src_mask, segment_ids, original_src,tgt, tgt_len, original_tgt = batch
            self.model.zero_grad()
            src = src.to(self.device)
            tgt = tgt.to(self.device)
            src_len = src_len.to(self.device)
            if src_mask is not None:
                src_mask=src_mask.to(self.device)
            segment_ids=segment_ids.to(self.device)

            with torch.no_grad():
                if self.config.sgm.beam_size > 1 and (not self.config.sgm.global_emb):
                    # beamsearch采样预测
                    samples, alignment, _ = self.model.beam_sample(src, src_mask, segment_ids,src_len, beam_size=self.config.sgm.beam_size,
                                                              eval_=True)
                    candidate += [tgt_vocab.convertToLabels(s, dict_helper.EOS) for s in samples]
                else:
                    # 贪心策略采样预测
                    samples, alignment = self.model.sample(src, src_mask, segment_ids,src_len)
                    # 通过id获取label
                    candidate += [tgt_vocab.convertToLabels(s.tolist(), dict_helper.EOS) for s in samples]
            source += original_src
            reference += original_tgt
            if alignment is not None:
                alignments += [align for align in alignment]


        if self.config.sgm.unk and self.config.sgm.attention != 'None':
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


        reference,candidate=prepare_label(reference,candidate,label_dict)


        self.outputs = torch.tensor(reference)
        self.targets = torch.tensor(candidate)
        # print(self.outputs.size(),self.targets.size())
        if self.epoch_metrics:
            for metric in self.epoch_metrics:
                metric(logits=self.outputs, target=self.targets)
                value = metric.value()
                if value:
                    self.result[f'valid_{metric.name()}'] = value
        if 'cuda' in str(self.device):
            torch.cuda.empty_cache()

        return self.result

    def train_bertsgm_epoch(self,data,epoch):
        self.epoch_reset()
        self.model.train()
        update=0
        for step, batch in enumerate(data):
            src, src_len, src_mask, segment_ids, original_src,tgt, tgt_len, original_tgt = batch
            self.model.zero_grad()
            src = src.to(self.device)
            tgt = tgt.to(self.device)
            src_len = src_len.to(self.device)
            if src_mask is not None:
                src_mask=src_mask.to(self.device)
            segment_ids=segment_ids.to(self.device)

            # 根据训练语料长度排序。
            lengths, indices = torch.sort(src_len, dim=0, descending=True)
            src = torch.index_select(src, dim=0, index=indices)
            tgt = torch.index_select(tgt, dim=0, index=indices)
            dec = tgt[:, :-1]
            targets = tgt[:, 1:]

            if self.config.sgm.schesamp:
                if epoch > 5:
                    e = epoch - 5
                    loss, outputs = self.model(src, src_mask, lengths, dec, targets, segment_ids,self.criterion,teacher_ratio=0.9 ** e)
                else:
                    loss, outputs = self.model(src, src_mask, lengths, dec, targets,segment_ids,self.criterion)
            else:

                loss, outputs = self.model(src, src_mask, lengths, dec, targets, segment_ids, self.criterion)

            targets = targets.t()
            # 总共标签数量
            num_total = targets.ne(dict_helper.PAD).sum().item()
            if self.config.sgm.max_split == 0:
                loss = torch.sum(loss) / num_total
                loss.backward()
            self.optimizer.step()

            #打印进度
            update += 1
            progress_bar(update, self.config.train_nbatchs)

        # 更新学习率
        self.optimizer.updateLearningRate(score=0, epoch=epoch)

     def train_seq2set_epoch(self,data,epoch):
        self.epoch_reset()
        self.model.train()
        update=0
        print('epoch==',epoch)
        for step, batch in enumerate(data):
            src, src_len, src_mask, segment_ids, original_src,tgt, tgt_len, original_tgt = batch
            self.model.zero_grad()
            src = src.to(self.device)
            tgt = tgt.to(self.device)
            src_len = src_len.to(self.device)
            if src_mask is not None:
                src_mask=src_mask.to(self.device)
            segment_ids=segment_ids.to(self.device)

            # 根据训练语料长度排序。
 
            lengths, indices = torch.sort(src_len, dim=0, descending=True)
            src = torch.index_select(src, dim=0, index=indices)
            tgt = torch.index_select(tgt, dim=0, index=indices)
            src_len = torch.index_select(src_len, dim=0, index=indices)
            src_mask = torch.index_select(src_mask, dim=0, index=indices)
            segment_ids = torch.index_select(segment_ids, dim=0, index=indices)
 

            dec = tgt[:, :-1]
            targets = tgt[:, 1:]
  
            loss=self.model.compute_reward(src, src_mask,src_len, dec, targets,segment_ids)
            loss.backward()

            self.optimizer.step()

             #打印进度
            update += 1
            progress_bar(update, self.config.train_nbatchs)

         # 更新学习率
        self.optimizer.updateLearningRate(score=0, epoch=epoch)

    def valid_seq2set_epoch(self,validloader,tgt_vocab,label_dict):
        print('start valid seq2set--------------')
        self.model.eval()
        self.epoch_reset()
        reference, candidate, source, alignments = [], [], [], []

        for step, batch in enumerate(validloader):
            src, src_len, src_mask, segment_ids, original_src,tgt, tgt_len, original_tgt = batch
            self.model.zero_grad()
            src = src.to(self.device)
            tgt = tgt.to(self.device)
            src_len = src_len.to(self.device)
            if src_mask is not None:
                src_mask=src_mask.to(self.device)
            segment_ids=segment_ids.to(self.device)

            # 根据训练语料长度排序。
            lengths, indices = torch.sort(src_len, dim=0, descending=True)
            src = torch.index_select(src, dim=0, index=indices)
            tgt = torch.index_select(tgt, dim=0, index=indices)
            src_len = torch.index_select(src_len, dim=0, index=indices)
            src_mask = torch.index_select(src_mask, dim=0, index=indices)
            segment_ids = torch.index_select(segment_ids, dim=0, index=indices)


            with torch.no_grad():
                if self.config.seq2set.beam_size > 1:
                    # beamsearch采样预测
                    samples, alignment, _ = self.model.beam_sample(src, src_mask, segment_ids,src_len, beam_size=self.config.seq2set.beam_size, eval_=True)
                    candidate += [tgt_vocab.convertToLabels(s, dict_helper.EOS) for s in samples]
                else:
                    # 贪心策略采样预测
                    samples, alignment = self.model.sample(src, src_mask, segment_ids,src_len)
                    # 通过id获取label
                    candidate += [tgt_vocab.convertToLabels(s.tolist(), dict_helper.EOS) for s in samples]
            source += original_src
            reference += original_tgt
            if alignment is not None:
                alignments += [align for align in alignment]


        if self.config.sgm.unk and self.config.sgm.attention != 'None':
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

        print('reference:',reference)
        print('candidate:',candidate)
        reference,candidate=prepare_label(reference,candidate,label_dict)


        self.outputs = torch.tensor(reference)
        self.targets = torch.tensor(candidate)
        # print(self.outputs.size(),self.targets.size())
        if self.epoch_metrics:
            for metric in self.epoch_metrics:
                metric(logits=self.outputs, target=self.targets)
                value = metric.value()
                if value:
                    self.result[f'valid_{metric.name()}'] = value
        if 'cuda' in str(self.device):
            torch.cuda.empty_cache()

        return self.result


    def train(self, data, seed):

        train_data = data['trainloader']
        valid_data = data['validloader']
        tgt_vocab = data['tgt_vocab']
        label2id= data['label2id']
        seed_everything(seed)
        if self.config.classifier == 'BertSGM' or self.config.classifier == 'SGM':
            with codecs.open(self.config.sgm.label_dict_file, 'r', 'utf-8') as f:
                label_dict = json.load(f)

        # ***************************************************************
        best=0
        for epoch in range(self.start_epoch, self.start_epoch+self.epochs):
            print(f"Epoch {epoch}/{self.epochs}")

            if self.config.classifier=='BertCNN' or self.config.classifier=='BertRCNN':
                train_log = self.train_epoch(train_data)
                valid_log = self.valid_epoch(valid_data)
                logs = dict(train_log, **valid_log)
                show_info = f'\nEpoch: {epoch} - ' + "-".join([f' {key}: {value:.4f} ' for key, value in logs.items()])
                print(show_info)


            if  self.config.classifier=='BertSGM' or self.config.classifier=='SGM':
                self.train_bertsgm_epoch(train_data,epoch)
                logs = self.valid_bertsgm_epoch(valid_data,tgt_vocab,label_dict)
                print(logs)

            if self.config.classifier=='BertSeq2Set':
                self.train_seq2set_epoch(train_data,epoch)
                logs = self.valid_seq2set_epoch(valid_data,tgt_vocab,label2id)
                print(logs)

            # 存储f1值最好的模型
            if logs['valid_f1'] > best:
                best = logs['valid_f1']
                torch.save(self.model, self.config.model_save_path + str(self.config.classifier)+'_bestmodel.pth')
            print('Epoch:%d  best f1:%s' % (epoch, str(best)))

            # early_stopping
            if self.early_stopping:
                self.early_stopping.epoch_step(
                    epoch=epoch, current=logs[self.early_stopping.monitor])
                if self.early_stopping.stop_training:
                    break