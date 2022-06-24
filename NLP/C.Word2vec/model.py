import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class SkipGramModel(nn.Module):

    def __init__(self,emb_size,emb_dimension):
        super(SkipGramModel,self).__init__()
        self.emb_size = emb_size
        self.emb_dimension = emb_dimension
        ### sparse (bool, optional) – 若为True,则与权重矩阵相关的梯度转变为稀疏张量
        self.u_embeddings = nn.Embedding(emb_size,emb_dimension,sparse=True)
        self.v_embeddings = nn.Embedding(emb_size,emb_dimension,sparse=True)

        initrange = 1.0 / emb_dimension
        init.uniform_(self.u_embeddings.weight.data, -initrange, initrange)
        init.constant_(self.v_embeddings.weight.data, 0)

    def forward(self,pos_u,pos_v,neg_v):
        emb_u = self.u_embeddings(pos_u)
        emb_v = self.v_embeddings(pos_v)
        emb_neg_v = self.v_embeddings(neg_v)

        score = torch.mul(emb_u,emb_v).sum(dim=1)
        score = torch.clamp(score,-1e10,1e10)
        score = -F.logsigmoid(score)

        neg_score = torch.mul(emb_u,emb_neg_v).sum(dim=1)
        neg_score = torch.clamp(neg_score,-1e10,1e10)
        neg_score = -F.logsigmoid(-neg_score)

        return torch.mean(score + neg_score)

    def save_embedding(self,id2word,file_name):
        embedding = self.u_embeddings.weight.data.cpu().numpy()
        with open(file_name,'w') as f:
            f.write('%d %d\n'%(len(id2word),self.emb_dimension))
            for wid,w in id2word.items():
                e = ' '.join(map(lambda x:str(x),embedding[wid]))
                f.write('%s %s\n'%(w,e))