import os
from typing import Tuple, Dict
import torch
from torch import nn, optim
import random
import numpy as np

from gensim.models.word2vec import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
import matplotlib.pyplot as plt
import itertools
import jieba    

### 创建一个关于sentences的迭代对象，返回的是一个生成器
class genrate_sentence(object):
    def __init__(self,sentences):
        self.sentences = sentences

    def __iter__(self):
        for item in self.sentences:
            yield item


### 训练word2vec模型
def train_word2vec(sentences: list, embedding_dim: int, model_save_path: str,
                     window_size: int = 5, min_count: int = 15,
                        workers: int = 4, sg: int = 1, iter: int = 5,
                        negative: int = 5, cbow_mean: bool = 1,
                        hs: bool = 1, sample: float = 1e-3,
                        seed: int = 42) -> Tuple[Word2Vec, Dict[str, int]]:
    """
    Train a word2vec model
    Parameters
    ----------
    sentences : list
        List of sentences to be trained on
    embedding_dim : int
        Dimension of the embedding vectors
    save_path : str
        Path to save the trained model
    window_size : int
        Context window size
    min_count : int
        Minimum word count threshold
    workers : int
        Number of worker threads to train the model
    sg : int
        1 for skip-gram, 0 for CBOW
    iter : int
        Number of iterations to train the model
    negative : int
        Number of negative examples to sample
    cbow_mean : bool
        If true, use the mean of the context word vectors
        when computing the context word vectors for a word,
        otherwise use the concatenation of the vectors
    hs : bool
        If true, hierarchical softmax will be used for model training
        and prediction
    sample : float
        Threshold for configuring which higher-frequency words are randomly downsampled
        when training the model
    seed : int
        Random seed for initializing and reproducing the experiments
    Returns
    -------
    trained_model : Word2Vec
        Trained word2vec model
    word_map : Dict[str, int]   
        Word2id map
    """
    print("\nTraining Word2Vec model...")
    sentencess = genrate_sentence(sentences)
    model = Word2Vec(sentencess, vector_size=embedding_dim, window=window_size,
                        min_count=min_count)
    model.save(model_save_path)

    return model




### 加载模型
def load_word2vec_format(path: str):
    print("\nLoading Word2Vec model...")
    model = KeyedVectors.load(path)

    return model



#对输入数据进行预处理,主要是对句子用索引表示且对句子进行截断与padding，将填充使用”pad_word“来。
def tokenizer(pd_all,pad_word,word2idx,sequence_length):   # 分词
    pad_id = word2idx[pad_word]
    inputs = []
    sentence_char = [list(jieba.cut(item)) for item in pd_all["review"].astype(str)]    
    # 将输入文本进行padding
    for index,item in enumerate(sentence_char):
        ### 若查找不存在，则返回第二个参数设置的默认值
        temp=[word2idx.get(item_item,pad_id) for item_item in item]#表示如果词表中没有这个稀有词，无法获得，那么就默认返回pad_id。
        if(len(item)<sequence_length):
            for _ in range(sequence_length-len(item)):
                temp.append(pad_id)
        else:
            temp = temp[:sequence_length]
        inputs.append(temp)
    return inputs







def save_checkpoint(
    epoch: int,
    model: nn.Module,
    model_name: str,
    optimizer: optim.Optimizer,
    checkpoint_path: str,
    best_loss: float = None,
    best_acc: float = None,
    checkpoint_basename: str = 'checkpoint'
) -> None:
    """
    Save a model checkpoint
    Parameters
    ----------
    epoch : int
        Epoch number the current checkpoint have been trained for
    model : nn.Module
        Model
    model_name : str
        Name of the model
    optimizer : optim.Optimizer
        Optimizer to update the model's weights
    dataset_name : str
        Name of the dataset
    word_map : Dict[str, int]
        Word2ix map
    checkpoint_path : str
        Path to save the checkpoint
    checkpoint_basename : str
        Basename of the checkpoint
    """
    state = {
        'epoch': epoch,
        'model': model,
        'model_name': model_name,
        'optimizer': optimizer,
        'best_loss': best_loss,
        'best_acc': best_acc,
    }
    save_path = os.path.join(checkpoint_path, checkpoint_basename + '.pth.tar')
    torch.save(state, save_path)

def load_checkpoint(
    checkpoint_path: str, device: torch.device
) -> Tuple[nn.Module, str, optim.Optimizer, str, Dict[str, int], int]:
    """
    Load a checkpoint, so that we can continue to train on it
    Parameters
    ----------
    checkpoint_path : str
        Path to the checkpoint to be loaded
    device : torch.device
        Remap the model to which device
    Returns
    -------
    model : nn.Module
        Model
    model_name : str
        Name of the model
    optimizer : optim.Optimizer
        Optimizer to update the model's weights
    dataset_name : str
        Name of the dataset
    word_map : Dict[str, int]
        Word2ix map
    start_epoch : int
        We should start training the model from __th epoch
    """
    checkpoint = torch.load(checkpoint_path, map_location=str(device))

    model = checkpoint['model']
    model_name = checkpoint['model_name']
    optimizer = checkpoint['optimizer']
    dataset_name = checkpoint['dataset_name']
    word_map = checkpoint['word_map']
    start_epoch = checkpoint['epoch'] + 1

    return model, model_name, optimizer, dataset_name, word_map, start_epoch

def clip_gradient(optimizer: optim.Optimizer, grad_clip: float) -> None:
    """
    Clip gradients computed during backpropagation to avoid explosion of gradients.
    Parameters
    ----------
    optimizer : optim.Optimizer
        Optimizer with the gradients to be clipped
    grad_clip : float
        Gradient clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

class AverageMeter:
    """
    Keep track of most recent, average, sum, and count of a metric
    """
    def __init__(self, tag = None, writer = None):
        self.writer = writer
        self.tag = tag
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

        # tensorboard
        if self.writer is not None:
            self.writer.add_scalar(self.tag, val)

def adjust_learning_rate(optimizer: optim.Optimizer, scale_factor: float) -> None:
    """
    Shrink learning rate by a specified factor.
    Parameters
    ----------
    optimizer : optim.Optimizer
        Optimizer whose learning rate must be shrunk
    shrink_factor : float
        Factor in interval (0, 1) to multiply learning rate with
    """
    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * scale_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def seed_everything(seed=42):
    '''
    设置整个开发环境的seed
    :param seed:
    :param device:
    :return:
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True