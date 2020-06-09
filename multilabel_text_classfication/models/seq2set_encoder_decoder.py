import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from models.attention import global_attention

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

class rnn_encoder(nn.Module):
    def __init__(self,config,bert,embedding=None):
        super(rnn_encoder,self).__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.pre_model=None
        if bert is not None:
            self.pre_model = bert
            for param in self.pre_model.parameters():
                param.requires_grad = True
            self.lstm = nn.LSTM(1024, config.hidden_size, num_layers=config.seq2set.enc_num_layers,bidirectional=config.seq2set.bidirectional, batch_first=True)
        else:
            self.embedding = embedding if embedding is not None else nn.Embedding(config.src_vocab_size, config.emb_size)
        
        if config.seq2set.cell == 'gru':
            #input_size=config.emb_size
            self.rnn = nn.GRU(input_size=config.hidden_size, hidden_size=config.hidden_size,
                              num_layers=config.seq2set.enc_num_layers, dropout=config.seq2set.dropout,
                              bidirectional=config.seq2set.bidirectional)
        else:
            # input_size=config.emb_size
            self.rnn = nn.LSTM(input_size=config.hidden_size, hidden_size=config.hidden_size,
                               num_layers=config.seq2set.enc_num_layers, dropout=config.seq2set.dropout,
                               bidirectional=config.seq2set.bidirectional)

    def forward(self,inputs,input_mask,segment_ids,lengths):
        if self.pre_model is not None:
            encoder_out, text_cls = self.pre_model(inputs,attention_mask=input_mask,token_type_ids=segment_ids)
            outputs,state=self.lstm(encoder_out)

        else:
            encoder_out = pack(self.embedding(inputs), lengths)
            outputs, state = self.rnn(encoder_out)
            # print('1---state[0] size:', state[0].size())
            outputs = unpack(outputs)[0]
        if self.config.seq2set.bidirectional:
            # outputs: [max_src_len, batch_size, hidden_size]
            #将双向lstm的输出进行相加，使得输出从[max_src_len, batch_size, 2*hidden_size]-》[max_src_len, batch_size, hidden_size]
            outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
            # print('2----outputs size:',outputs.size())
            if self.config.seq2set.cell == 'gru':
                state = state[:self.config.seq2set.dec_num_layers]
            else:
                state = (state[0][::2], state[1][::2])

        return outputs, state

class rnn_decoder(nn.Module):

    def __init__(self, config,embedding=None, score_fn=None):
        super(rnn_decoder, self).__init__()

        self.config = config
        self.hidden_size = config.hidden_size

        self.embedding = embedding if embedding is not None else nn.Embedding(config.label_vocab_size, config.seq2set.emb_size)
        #config.emb_size
        input_size =  config.hidden_size

        if config.cell == 'gru':
            self.rnn = StackedGRU(input_size=input_size, hidden_size=config.hidden_size,
                                  num_layers=config.seq2set.dec_num_layers, dropout=config.seq2set.dropout)
        else:
            self.rnn = StackedLSTM(input_size=input_size, hidden_size=config.hidden_size,
                                   num_layers=config.seq2set.dec_num_layers, dropout=config.seq2set.dropout)

        self.score_fn = score_fn
        if self.score_fn.startswith('general'):
            self.linear = nn.Linear(config.hidden_size, config.seq2set.emb_size)
        elif score_fn.startswith('concat'):
            self.linear_query = nn.Linear(config.hidden_size, config.hidden_size)
            self.linear_weight = nn.Linear(config.seq2set.emb_size, config.hidden_size)
            self.linear_v = nn.Linear(config.hidden_size, 1)
        elif not self.score_fn.startswith('dot'):
            self.linear = nn.Linear(config.hidden_size, config.label_vocab_size)


        if hasattr(config, 'att_act'):
            activation = config.att_act
            print('use attention activation %s' % activation)
        else:
            activation = None

        self.attention=global_attention(config.hidden_size,activation)
        
        self.dropout = nn.Dropout(config.seq2set.dropout)
        


    def forward(self, input, init_state, contextse):

        embs = self.embedding(input)
        outputs,state,attns=[],init_state,[]
        for emb in embs.split(1):
            output,state=self.rnn(emb.sequeeze(0),state)
            output,att_weights=self.attention(output,contexts)
            output=self.dropout(output)
            outputs+=[output]
            attns+=[att_weights]
        outputs=torch.stack(output)
        attns=torch.stack(attns)
        return outputs,state


    def compute_score(self, hiddens):
        if self.score_fn.startswith('general'):
            if self.score_fn.endswith('not'):
                scores = torch.matmul(self.linear(hiddens), Variable(self.embedding.weight.t().data))
            else:
                scores = torch.matmul(self.linear(hiddens), self.embedding.weight.t())
        elif self.score_fn.startswith('concat'):
            if self.score_fn.endswith('not'):
                scores = self.linear_v(torch.tanh(self.linear_query(hiddens).unsqueeze(1) + self.linear_weight(Variable(self.embedding.weight.data)).unsqueeze(0))).squeeze(2)
            else:
                scores = self.linear_v(torch.tanh(self.linear_query(hiddens).unsqueeze(1) + self.linear_weight(self.embedding.weight).unsqueeze(0))).squeeze(2)
        elif self.score_fn.startswith('dot'):
            if self.score_fn.endswith('not'):
                scores = torch.matmul(hiddens, Variable(self.embedding.weight.t().data))
            else:
                scores = torch.matmul(hiddens, self.embedding.weight.t())
        else:
            scores = self.linear(hiddens)
        return scores

    def sample(self,input,init_state,contexts):
        inputs,outputs,sample_ids,state=[],[],[],init_state
        attns=[]
        inputs+=input
        for i in range(self.config.seq2set.max_time_step):
            output,state, attn_weights=self.sample_one(inputs[i],state,contexts)
            predicted=output.max(1)[1]
            inputs+=[predicted]
            sample_ids+=[predicted]
            outputs+=[output]
            attns+=[attn_weights]
        sample_ids=torch.stack(sample_ids)
        attns=torch.stack(attns)
        return sample_ids,(outputs,attns)

    def sample_one(self,input,state,contexts):
        emb=self.embedding(input)
   
        output,state=self.rnn(emb,state)
       
        hidden,att_weights=self.attention(output,contexts)
         
        output=self.compute_score(hidden)
         

        return output,state,att_weights