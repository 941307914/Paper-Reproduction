import time
from tkinter import Variable
from typing import Optional, Dict
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tqdm import tqdm
from sklearn.metrics import *

from utils import AverageMeter, save_checkpoint, \
    clip_gradient, adjust_learning_rate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer:
    """
    Training pipeline
    Parameters
    ----------
    num_epochs : int
        We should train the model for __ epochs
    start_epoch : int
        We should start training the model from __th epoch
    train_loader : DataLoader
        DataLoader for training data
    model : nn.Module
        Model
    model_name : str
        Name of the model
    loss_function : nn.Module
        Loss function (cross entropy)
    optimizer : optim.Optimizer
        Optimizer (Adam)
    lr_decay : float
        A factor in interval (0, 1) to multiply the learning rate with
    dataset_name : str
        Name of the dataset
    word_map : Dict[str, int]
        Word2id map
    grad_clip : float, optional
        Gradient threshold in clip gradients
    print_freq : int
        Print training status every __ batches
    checkpoint_path : str, optional
        Path to the folder to save checkpoints
    checkpoint_basename : str, optional, default='checkpoint'
        Basename of the checkpoint
    tensorboard : bool, optional, default=False
        Enable tensorboard or not?
    log_dir : str, optional
        Path to the folder to save logs for tensorboard
    """
    def __init__(
        self,
        num_epochs: int,
        train_loader: DataLoader,
        model: nn.Module,
        model_name: str,
        loss_function: nn.Module,
        optimizer,
        device,
        lr_decay: float,
        grad_clip = Optional[None],
        checkpoint_path: Optional[str] = None,
        checkpoint_basename: str = 'checkpoint'
    ) -> None:
        self.num_epochs = num_epochs
        self.train_loader = train_loader

        self.model = model.to(device)
        self.model_name = model_name
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.lr_decay = lr_decay
        self.grad_clip = grad_clip

        self.checkpoint_path = checkpoint_path
        self.checkpoint_basename = checkpoint_basename
        self.device = device

        # setup visualization writer instance
        # self.writer = TensorboardWriter(log_dir, tensorboard)
        self.len_epoch = len(self.train_loader)

        # initialize the learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=self.lr_decay,
            patience=1,
            verbose=True,
            threshold=0.0001,
            threshold_mode='rel',
            cooldown=0,
            min_lr=0,
            eps=1e-08
        )
        
    

    def train(self) -> None:
        self.model.train()  # training mode enables dropout

        batch_time = AverageMeter()  # forward prop. + back prop. time per batch
        data_time = AverageMeter()  # data loading time per batch
        losses = AverageMeter(tag='loss')  # cross entropy loss
        accs = AverageMeter(tag='acc')  # accuracies
        start = time.time()


        best_loss = float('inf')
        best_acc = 0.0

        # batches
        for i,(batch_x,batch_y) in tqdm(enumerate(self.train_loader)):
            data_time.update(time.time() - start)
            self.optimizer.zero_grad()
            batch_x = Variable(batch_x).to(self.device)
            batch_y = Variable(batch_y).to(self.device)
            outputs = self.model(batch_x)
            outputs = outputs.squeeze(1)
            outputs = torch.sigmoid(outputs)
            
            loss = self.loss_function(outputs, batch_y)

            loss.backward()
            self.optimizer.step()
            self.scheduler.step(loss)
            
            # clip gradients
            if self.grad_clip is not None:
                clip_gradient(self.optimizer, self.grad_clip)

            losses.update(loss.item(), batch_x.size(0))
            accs.update(torch.sum(outputs.round() == batch_y).item() / batch_x.size(0), batch_x.size(0))

            ### 比较loss和acc和最大的loss和acc，如果比最大的loss和acc小，则更新最大的loss和acc
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_acc = accs.avg

        return best_loss,best_acc

    def run_training(self,epoch) -> None:
        """
        Run training
        """
        # initialize the best_loss and best_acc
        best_loss = float('inf')
        best_acc = 0.0

        # epochs
        for _ in range(epoch):
            # train for one epoch
            train_loss,train_acc = self.train()

            # save the model if the validation loss is the best we've seen so far
            if train_loss < best_loss:
                best_loss = train_loss
                best_acc = train_acc
                save_checkpoint(epoch,self.model, self.model_name, self.optimizer, self.checkpoint_path,
                best_loss, best_acc,self.checkpoint_basename)

        # print the best validation loss and accuracy
            print(
                '\nTrain Loss: {}\t'
                'Train Accuracy: {}'.format(
                    train_loss,
                    train_acc
                )
            )


### 在测试集上验证数据
class Tester:
    def __init__(self, model,val_loader,loss_function,device):
        self.model = model
        self.test_loader = val_loader
        self.device = device
        self.model.to(self.device)
        self.loss_function = loss_function
        self.model.eval()
        # setup visualization writer instance
 

    def test(self) -> None:
        data_time = AverageMeter()  # data loading time per batch
        losses = AverageMeter(tag='loss')  # cross entropy loss
        start = time.time()


        true_y = []
        pred_y = []

        # batches
        for i,(batch_x,batch_y) in tqdm(enumerate(self.test_loader)):
            data_time.update(time.time() - start)

            batch_x = Variable(batch_x).to(self.device)
            batch_y = Variable(batch_y).to(self.device)
            outputs = self.model(batch_x)
            outputs = torch.sigmoid(outputs.squeeze(1))

            loss = self.loss_function(outputs, batch_y)
            losses.update(loss.item(), batch_x.size(0))

            outputs = outputs.round()
            true_y.extend(batch_y.cpu().data.numpy())
            pred_y.extend(outputs.cpu().data.numpy())

        return loss,true_y,pred_y

    def run_testing(self,epoch) -> None:
        """
        Run training
        """
        # initialize the best_loss and best_acc
        best_loss = float('inf')
        ### 准确率、召回率、F1值、精确率
        best_acc = 0.0
        best_recall = 0.0
        best_precision = 0.0
        best_f1 = 0.0


        # epochs
        for _ in range(epoch):
            # train for one epoch
            loss,true_y,pred_y = self.test()

            ### 根据true_y,pred_y计算
            acc = accuracy_score(true_y,pred_y)
            recall = recall_score(true_y,pred_y)
            precision = precision_score(true_y,pred_y)
            f1 = f1_score(true_y,pred_y)

            if(loss<best_loss):
                best_loss = loss
                best_acc = acc
                best_recall = recall
                best_precision = precision
                best_f1 = f1

                print('\nTest Loss: {}\t'
                'Test Accuracy: {}\t'
                'Test Recall: {}\t'
                'Test Precision: {}\t'
                'Test F1: {}'.format(
                    loss,
                    best_acc,
                    best_recall,
                    best_precision,
                    best_f1
                ))