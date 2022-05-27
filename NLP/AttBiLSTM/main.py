from Preprocess import *
from model import *
from utils import *
from train import *
import os,sys
import torch
from torch import nn, optim
from torch.nn import functional as F

os.chdir(sys.path[0])



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
word_freq_file_path = '..\data\online_shopping_10_cats\word_freq.txt'
file_path = '..\data\online_shopping_10_cats\online_shopping_10_cats.csv'


if os.path.exists(word_freq_file_path):
    labels,reviews_cut,word2idx,idx2word = read_word_freq_file(file_path,file_path)
else:
    labels,reviews_cut,word2idx,idx2word = read_csv(file_path,word_freq_file_path)


### 查看是否存在w2v文件，如果有则直接读取，如果没有则训练w2v
w2v_model_file_path = '..\data\online_shopping_10_cats\w2v_model.bin'
embedding_dim = 100

if os.path.exists(w2v_model_file_path):
    w2v_model,w2v = load_word2vec_format(w2v_model_file_path)
else:
    w2v_model,w2v = train_word2vec(reviews_cut,embedding_dim=embedding_dim, model_save_path=w2v_model_file_path)

### 利用w2v构建embedding层
embedding_layer = create_embedding_layer(w2v_model, word2idx, embedding_dim, device)

### 创建AttBiLSTM模型
model = BiLSTM_Attention(embedding_layer, embedding_dim).to(device)


### 创建训练器通过train.py文件下的Train类
