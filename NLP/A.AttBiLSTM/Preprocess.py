### 导入numpy
import numpy as np
import pandas as pd
import jieba
from collections import Counter 
from torch.utils.data import Dataset
import torch



### 创建dataset
class AttBiLSTM_Dataset(Dataset):
    def __init__(self,inputs,labels):
        self.labels = [int(i) for i in labels]
        self.inputs = inputs
        self.len = len(labels)
    
    def __getitem__(self,index):
        return torch.tensor(self.inputs[index],dtype=torch.long),torch.tensor(self.labels[index],dtype=torch.float32)
    
    def __len__(self):
        return self.len


def tokenizer(reviews_cut,word2idx,pad_word,sequence_length=110):   # 分词
    inputs = []  
    # 将输入文本进行padding
    try:
        pad_id = word2idx[pad_word]
    except:
        pad_id = word2idx['的']

    for index,item in enumerate(reviews_cut):
        ### 若查找不存在，则返回第二个参数设置的默认值
        temp=[word2idx.get(item_item,pad_id) for item_item in item]#表示如果词表中没有这个稀有词，无法获得，那么就默认返回pad_id。
        if(len(item)<sequence_length):
            for _ in range(sequence_length-len(item)):
                temp.append(pad_id)
        else:
            temp = temp[:sequence_length]
        inputs.append(temp)
    return inputs


### 读取csv文件，返回labels,和分词后的句子，单词库word2idx和dix2word
def read_csv(file_path,word_freq_file_path):
    df = pd.read_csv(file_path,encoding='utf-8')
    ### 读取df中的label和review，返回列表格式
    labels = list(df.loc[:,'label'].astype(str))
    reviews = list(df.loc[:,'review'].astype(str))
    
    ### 对reviews分词
    reviews_cut = []
    for review in reviews:
        reviews_cut.append(list(jieba.cut(review)))
    
    ### 对reviews分词，统计所有的词汇，过滤掉频率低于15次的字,将剩余的词汇写入word_freq_file_path文件，并且转换为数字，返回word2idx和dix2word
    words_counter = Counter()
    for review in reviews_cut:
        words_counter.update(review)
    words_counter = words_counter.most_common()
    words_counter = [word_freq[0] for word_freq in words_counter if word_freq[1] >= 15]
    words_counter = list(set(words_counter))
    ### 将words_counter写入word_freq_file_path文件
    with open(word_freq_file_path,'w',encoding='utf-8') as f:
        for word in words_counter:
            f.write(word+'\n')

    ### 将剩余的词转换为word2idx和idx2word
    word2idx = {}
    idx2word = {}
    for idx,word in enumerate(words_counter):
        word2idx[word] = idx
        idx2word[idx] = word

    reviews_cut = tokenizer(reviews_cut,word2idx,'的',sequence_length=110)
    
    
    return labels,reviews_cut,word2idx,idx2word


### 读取文件中的词，并且返回word2idx和idx2word
def read_word_freq_file(file_path,word_freq_file_path):

    df = pd.read_csv(file_path,encoding='utf-8')
    ### 读取df中的label和review，返回列表格式
    labels = list(df.loc[:,'label'].astype(str))
    reviews = list(df.loc[:,'review'].astype(str))

    ### 对reviews分词
    reviews_cut = []
    for review in reviews:
        reviews_cut.append(list(jieba.cut(review)))


    words_list = []
    with open(word_freq_file_path,'r',encoding='utf-8') as f:
        for line in f:
            words_list.append(line.strip())
            
    word2idx = {}
    idx2word = {}
    for idx,word in enumerate(words_list):
        word2idx[word] = idx
        idx2word[idx] = word

    reviews_cut = tokenizer(reviews_cut,word2idx,'的',sequence_length=110)

    return labels,reviews_cut,word2idx,idx2word