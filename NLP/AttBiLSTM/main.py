from Preprocess import *
from model import *
from utils import *
from train import *
import os,sys
import torch
from torch import nn, optim
from torch.nn import functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader


os.chdir(sys.path[0])



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
word_freq_file_path = '..\data\online_shopping_10_cats\word_freq.txt'
file_path = '..\data\online_shopping_10_cats\online_shopping_10_cats.csv'
checkpoint_path = '..\data\online_shopping_10_cats'


if os.path.exists(word_freq_file_path):
    labels,reviews_cut,word2idx,idx2word = read_word_freq_file(file_path,word_freq_file_path)
else:
    labels,reviews_cut,word2idx,idx2word = read_csv(file_path,word_freq_file_path)


### 查看是否存在w2v文件，如果有则直接读取，如果没有则训练w2v
w2v_model_file_path = '..\data\online_shopping_10_cats\w2v_model.pkl'
embedding_dim = 100





if os.path.exists(w2v_model_file_path):
    w2v_model = load_word2vec_format(w2v_model_file_path)
else:
    w2v_model = train_word2vec(reviews_cut,embedding_dim=embedding_dim, model_save_path=w2v_model_file_path)

### 利用w2v构建embedding层
embedding_layer = create_embedding_layer(w2v_model, word2idx, embedding_dim, device)

### 创建AttBiLSTM模型
model = BiLSTM_Attention(embedding_layer, embedding_dim).to(device)


### 读取数据
labels,reviews_cut,word2idx,idx2word = read_csv(file_path,word_freq_file_path)
### 使用sklearn分割数据
X_train, X_test, y_train, y_test = train_test_split(reviews_cut, labels, test_size=0.2, random_state=42)
train_dataset = AttBiLSTM_Dataset(X_train,y_train)
test_dataset = AttBiLSTM_Dataset(X_test,y_test)

train_loader = DataLoader(train_dataset,batch_size=1024,shuffle=True)
test_loader = DataLoader(test_dataset,batch_size=1024,shuffle=True)
optimizer = optim.Adam(model.parameters(), lr=0.001)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

trainer = Trainer(num_epochs=10,train_loader=train_loader,model=model,model_name='BiLSTM',loss_function=nn.BCELoss(),
                    optimizer=optimizer,device=device,lr_decay=0.5,grad_clip=5.0,
                    checkpoint_path=checkpoint_path,checkpoint_basename='checkpoint',tensorboard=True)

trainer.run_training(10)

tester = Tester(model=model,val_loader=test_loader,loss_function=nn.BCELoss(),device=device)

tester.run_testing(10)
