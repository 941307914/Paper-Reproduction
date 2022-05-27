import numpy as np
import pandas as pd
import jieba
from collections import Counter 



### 读取csv文件，返回labels,和分词后的句子，单词库word2idx和dix2word
def read_csv(file_path,word_freq_file_path):
    df = pd.read_csv(file_path,encoding='utf-8')
    ### 读取df中的label和review，返回列表格式
    labels = df['label'].tolist()
    reviews = df['review'].tolist()
    
    ### 对reviews分词
    reviews_cut = []
    for review in reviews:
        reviews_cut.append(jieba.cut(review))
    
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
    
    return labels,reviews_cut,word2idx,idx2word


### 读取文件中的词，并且返回word2idx和idx2word
def read_word_freq_file(file_path,word_freq_file_path):

    df = pd.read_csv(file_path,encoding='utf-8')
    ### 读取df中的label和review，返回列表格式
    labels = df['label'].tolist()
    reviews = df['review'].tolist()

    ### 对reviews分词
    reviews_cut = []
    for review in reviews:
        reviews_cut.append(jieba.cut(review))


    words_counter = []
    with open(word_freq_file_path,'r',encoding='utf-8') as f:
        for line in f:
            words_counter.append(line.strip())
    word2idx = {}
    idx2word = {}
    for idx,word in enumerate(words_counter):
        word2idx[word] = idx
        idx2word[idx] = word
    return labels,reviews_cut,word2idx,idx2word