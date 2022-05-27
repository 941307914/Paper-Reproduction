import os
from typing import Tuple, Dict
import torch
from torch import nn, optim
import random
import numpy as np

from gensim.models.word2vec import Word2Vec
from gensim.models.keyedvectors import KeyedVectors


### 训练word2vec模型
def train_word2vec(sentences: list, embedding_dim: int, model_save_path: str,
                     window_size: int = 5, min_count: int = 5,
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
    model = Word2Vec(sentences, size=embedding_dim, window=window_size,
                        min_count=min_count, workers=workers, sg=sg,
                        iter=iter, negative=negative, cbow_mean=cbow_mean,
                        hs=hs, sample=sample, seed=seed)
    model.save(model_save_path)
    word_map = {word: i for i, word in enumerate(model.wv.index2word)}
    return model, word_map


### genism读取w2v数据
def load_word2vec_format(path: str, binary: bool = False) -> Tuple[KeyedVectors, Dict[str, int]]:
    """
    Load a word2vec model in word2vec format
    Parameters
    ----------
    path : str
        Path to the word2vec model
    binary : bool
        If true, the model is in binary format
    Returns
    -------
    model : KeyedVectors
        Trained word2vec model
    word_map : Dict[str, int]
        Word2id map
    """
    print("\nLoading Word2Vec model...")
    model = KeyedVectors.load_word2vec_format(path, binary=binary)
    word_map = {word: i for i, word in enumerate(model.index2word)}
    return model, word_map

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
    embedding_layer = nn.Embedding(len(word_map), embedding_dim)
    embedding_layer.weight.data.copy_(torch.from_numpy(model.vectors))
    embedding_layer.weight.requires_grad = False
    embedding_layer = embedding_layer.to(device)
    return embedding_layer


def save_checkpoint(
    epoch: int,
    model: nn.Module,
    model_name: str,
    optimizer: optim.Optimizer,
    dataset_name: str,
    word_map: Dict[str, int],
    checkpoint_path: str,
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
        'dataset_name': dataset_name,
        'word_map': word_map
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