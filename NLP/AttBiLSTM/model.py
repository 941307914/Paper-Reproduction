import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class BiLSTM_Attention(nn.Module):
    def __init__(self,embedding_layer,embedding_dim,hidden_dim=64,num_classes=2,dropout=0.5):
        super(BiLSTM_Attention,self).__init__()
        # self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        self.embedding_layer = embedding_layer
        self.lstm = nn.LSTM(embedding_dim,self.hidden_dim,batch_first=True,bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_features=self.hidden_dim*2,out_features=self.num_classes)
    
    ### 创建自注意力层,lstm_output : [batch_size, n_step, n_hidden * num_directions(=2)], F matrix
    def attention(self,lstm_output, final_state):
        hidden = final_state.view(-1,self.hidden_dim*2,1) ### [batch_size, n_hidden * num_directions(=2), 1]
        att_weight = torch.bmm(lstm_output,hidden).squeeze(2) ### [batch_size, n_step]
        soft_att_weight = F.softmax(att_weight,dim=1) ### [batch_size, n_step]
        att_output = torch.bmm(lstm_output.transpose(1,2),soft_att_weight.unsqueeze(2)).squeeze(2) ### [batch_size, n_hidden * num_directions(=2)]
        return att_output,soft_att_weight.data.numpy() ### [batch_size, n_hidden * num_directions(=2)]

    def forward(self,inputs):
        embedding_output = self.embedding_layer(inputs) ### [batch_size, n_step, embedding_dim]

        lstm_output,(final_hidden_state,final_cell_state) = self.lstm(embedding_output) ### [batch_size, n_step, n_hidden * num_directions(=2)]

        att_output,att_weight = self.attention(lstm_output,final_hidden_state) ### [batch_size, n_hidden * num_directions(=2)]

        dropout_output = self.dropout(att_output)
        fc_output = self.fc(dropout_output)
        return fc_output