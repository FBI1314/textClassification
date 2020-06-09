import torch
import torch.nn as nn
# import utils
# import models
from models.sgm_encoder_decoder import rnn_encoder,rnn_decoder
from utils import dict_helper
from models.beam import Beam
from transformers import *



class SGM(nn.Module):

    def __init__(self, config, encoder=None, decoder=None):
        super(SGM, self).__init__()
        self.bert_model = None
        if encoder is not None:
            self.encoder = encoder
        else:
            if config.pretrain=='Bert':
                self.bert_model = BertModel.from_pretrained('./bert_pretrain/')

            self.encoder = rnn_encoder(config,self.bert_model)
        tgt_embedding = self.encoder.embedding if config.sgm.shared_vocab else None
        if decoder is not None:
            self.decoder = decoder
        else:
            self.decoder =rnn_decoder(config,embedding=tgt_embedding, use_attention=config.sgm.attention)
        self.log_softmax = nn.LogSoftmax(dim=-1)
        if config.n_gpu!='':
            self.use_cuda = True
        self.config = config
        if config.use_cuda:
            self.criterion.cuda()

    def forward(self, src, src_mask,src_len, dec, targets,segment_ids,criterion):
        """
        Args:
            src: [bs, src_len]
            src_len: [bs]
            dec: [bs, tgt_len] (bos, x1, ..., xn)
            targets: [bs, tgt_len] (x1, ..., xn, eos)
        """
        if self.bert_model is None:
            src = src.t()
        dec = dec.t()
        targets = targets.t()

        contexts, state = self.encoder(src, src_mask,segment_ids,src_len.tolist())


        if self.decoder.attention is not None:
            if self.bert_model is None:
                contexts=contexts.transpose(0, 1)
            self.decoder.attention.init_context(context=contexts)

        outputs = []
        output = None

        for input in dec.split(1):
            output, state, _ = self.decoder(input.squeeze(0), state, output)
            outputs.append(output)
        outputs = torch.stack(outputs)


        scores = outputs.view(-1, outputs.size(2))
        loss = criterion(scores, targets.contiguous().view(-1))
        return loss, outputs

    #贪心策略解码预测
    def sample(self, src, src_mask,segment_ids,src_len):

        lengths, indices = torch.sort(src_len, dim=0, descending=True)
        _, reverse_indices = torch.sort(indices)
        src = torch.index_select(src, dim=0, index=indices)
        bos = torch.ones(src.size(0)).long().fill_(dict_helper.BOS)
        # print('bos:', bos)
        if self.bert_model is None:
            src = src.t()

        if self.use_cuda:
            bos = bos.cuda()

        contexts, state = self.encoder(src, src_mask,segment_ids,lengths.tolist())

        if self.decoder.attention is not None:
            if self.bert_model is None:
                contexts = contexts.transpose(0, 1)
            self.decoder.attention.init_context(context=contexts)

        # print(self.config.max_time_step)
        inputs, outputs, attn_matrix = [bos], [], []
        output = None

        for i in range(self.config.sgm.max_time_step):

            # print('state[0].size:',state[0].size())
            output, state, attn_weights = self.decoder(inputs[i], state, output, outputs)
            predicted = output.max(1)[1]
            inputs += [predicted]
            outputs += [predicted]
            attn_matrix += [attn_weights]

        outputs = torch.stack(outputs)

        sample_ids = torch.index_select(outputs, dim=1, index=reverse_indices).t()

        #取每个位置最大大attention值
        if self.decoder.attention is not None:
            attn_matrix = torch.stack(attn_matrix)
            alignments = attn_matrix.max(2)[1]
            alignments = torch.index_select(alignments, dim=1, index=reverse_indices).t()
        else:
            alignments = None

        return sample_ids, alignments

    #beamsearch策略编码预测
    def beam_sample(self, src, src_mask,segment_ids,src_len, beam_size=1, eval_=False):

        # (1) Run the encoder on the src.

        lengths, indices = torch.sort(src_len, dim=0, descending=True)
        _, ind = torch.sort(indices)
        src = torch.index_select(src, dim=0, index=indices)
        if self.bert_model is None:
            src = src.t()
            batch_size = src.size(1)
        else:
            batch_size=src.size(0)
        contexts, encState = self.encoder(src,src_mask, segment_ids,lengths.tolist())
         #  (1b) Initialize for the decoder.
        def var(a):
            return a.clone().detach().requires_grad_(False)

        def rvar(a):
            return var(a.repeat(1, beam_size, 1))

        def bottle(m):
            return m.view(batch_size * beam_size, -1)

        def unbottle(m):
            return m.view(beam_size, batch_size, -1)



        if self.config.sgm.cell == 'lstm':
            decState = (rvar(encState[0]), rvar(encState[1]))
        else:
            decState = rvar(encState)

        beam = [Beam(beam_size, n_best=1,cuda=self.use_cuda, length_norm=self.config.sgm.length_norm) for __ in range(batch_size)]
        # if self.decoder.attention is not None:
        #     self.decoder.attention.init_context(contexts)
        if self.decoder.attention is not None:
            if self.bert_model is None:
                # Repeat everything beam_size times.
                contexts = rvar(contexts)
                contexts = contexts.transpose(0, 1)
            contexts = var(contexts.repeat(beam_size, 1, 1))
            self.decoder.attention.init_context(context=contexts)

        # (2) run the decoder to generate sentences, using beam search.

        for i in range(self.config.sgm.max_time_step):

            if all((b.done() for b in beam)):
                break

            # Construct batch x beam_size nxt words.
            # Get all the pending current beam words and arrange for forward.
            inp = var(torch.stack([b.getCurrentState() for b in beam])
                      .t().contiguous().view(-1))
            # if self.bert_model is None:
            #     inp = var(torch.stack([b.getCurrentState() for b in beam])
            #               .t().contiguous().view(-1))
            # else:
            #     inp = var(torch.stack([b.getCurrentState() for b in beam]).contiguous().view(-1))
            # Run one step.


            output, decState, attn = self.decoder(inp, decState)
            # decOut: beam x rnn_size

            # (b) Compute a vector of batch*beam word scores.
            output = unbottle(self.log_softmax(output))
            attn = unbottle(attn)
            # beam x tgt_vocab

            # (c) Advance each beam.
            # update state
            for j, b in enumerate(beam):
                b.advance(output[:, j], attn[:, j])
                if self.config.sgm.cell == 'lstm':
                    b.beam_update(decState, j)
                else:
                    b.beam_update_gru(decState, j)

        # (3) Package everything up.
        allHyps, allScores, allAttn = [], [], []
        if eval_:
            allWeight = []

        for j in ind:
            b = beam[j]
            n_best = 1
            scores, ks = b.sortFinished(minimum=n_best)
            hyps, attn = [], []
            if eval_:
                weight = []
            for i, (times, k) in enumerate(ks[:n_best]):
                hyp, att = b.getHyp(times, k)
                hyps.append(hyp)
                attn.append(att.max(1)[1])
                if eval_:
                    weight.append(att)
            allHyps.append(hyps[0])
            allScores.append(scores[0])
            allAttn.append(attn[0])
            if eval_:
                allWeight.append(weight[0])
        
        if eval_:
            return allHyps, allAttn, allWeight

        return allHyps, allAttn