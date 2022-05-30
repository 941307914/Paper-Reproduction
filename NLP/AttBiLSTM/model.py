import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from gensim.models.word2vec import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from typing import Tuple, Dict

### 创建利用word2vc创建embedding层
def create_embedding_layer(model: KeyedVectors, word_map: Dict[str, int],
                            embedding_dim: int, device: torch.device) -> nn.Embedding:
    """
    Create an embedding layer
    Parameters
    ----------
    model : KeyedVectors
        Trained word2vec model
    word_map : Dict[str, int]
        Word2ix map
    embedding_dim : int
        Dimension of the embedding vectors
    device : torch.device
        Device to create the embedding layer on
    Returns
    -------
    embedding_layer : nn.Embedding
        Embedding layer
    """
    embedding_matrix = np.zeros((len(word_map),embedding_dim))

    for word,index in word_map.items():    
        try:
            embedding_matrix[index,:] = model[word]
        except:
            embedding_matrix[index,:] = np.random.rand(embedding_dim)

    embedding = nn.Embedding(embedding_matrix.shape[0],embedding_matrix.shape[1])
    embedding.weight = nn.Parameter(torch.tensor(embedding_matrix,dtype=torch.float32))
    return embedding


class BiLSTM_Attention(nn.Module):
    def __init__(self,embedding_layer,embedding_dim,hidden_dim=64,num_classes=1,dropout=0.5):
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
        return att_output ### [batch_size, n_hidden * num_directions(=2)]

    def forward(self,inputs):
        embedding_output = self.embedding_layer(inputs) ### [batch_size, n_step, embedding_dim]

        lstm_output,(final_hidden_state,final_cell_state) = self.lstm(embedding_output) ### [batch_size, n_step, n_hidden * num_directions(=2)]

        att_output = self.attention(lstm_output,final_hidden_state) ### [batch_size, n_hidden * num_directions(=2)]

        dropout_output = self.dropout(att_output)
        fc_output = self.fc(dropout_output)
        return fc_output