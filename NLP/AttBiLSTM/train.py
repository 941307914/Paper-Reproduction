import time
from typing import Optional, Dict
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from utils import TensorboardWriter, AverageMeter, save_checkpoint, \
    clip_gradient, adjust_learning_rate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


### 创建训练类，用于训练模型
class Train:
    def __init__(self, model, model_name, optimizer, dataset_name, word_map, start_epoch,
                 epochs, batch_size, learning_rate, grad_clip, log_step,
                 log_dir, checkpoint_dir, teacher_forcing_ratio, teacher_forcing_ratio_decay):
        self.model = model
        self.model_name = model_name
        self.optimizer = optimizer
        self.dataset_name = dataset_name
        self.word_map = word_map
        self.start_epoch = start_epoch
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.grad_clip = grad_clip
        self.log_step = log_step
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.teacher_forcing_ratio_decay = teacher_forcing_ratio_decay
        self.writer = TensorboardWriter(self.log_dir)
        self.loss_criterion = nn.CrossEntropyLoss(reduction='none')
        self.train_losses = []
        self.valid_losses = []
        self.train_accuracies = []
        self.valid_accuracies = []

    def train(self):
        """
        Train the model for a certain number of epochs
        """
        for epoch in range(self.start_epoch, self.epochs + 1):
            self.adjust_learning_rate(epoch)
            self.train_epoch(epoch)
            self.validate_epoch(epoch)
            self.save_checkpoint(epoch)
            self.writer.add_scalar('train_loss', self.train_losses[-1], epoch)
            self.writer.add_scalar('valid_loss', self.valid_losses[-1], epoch)
            self.writer.add_scalar('train_acc', self.train_accuracies[-1], epoch)
            self.writer.add_scalar('valid_acc', self.valid_accuracies[-1], epoch)
            self.writer.add_scalar('teacher_forcing_ratio', self.teacher_forcing_ratio, epoch)

### 写一个测试函数用于检验模型的性能
    def validate_epoch(self, epoch):
        """
        Validate the model on a single epoch
        """
        self.model.eval()
        valid_loss = 0.0
        valid_num_correct = 0
        valid_num_total = 0
        with torch.no_grad():
            for batch in self.valid_loader:
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                logits = self.model(input_ids, segment_ids, input_mask)
                loss = self.loss_criterion(logits.view(-1, self.num_labels), label_ids.view(-1))
                valid_loss += loss.item()
                valid_num_correct += (torch.argmax(logits, dim=2) == label_ids).sum().item()
                valid_num_total += input_ids.size(0)
        valid_loss /= valid_num_total
        valid_accuracy = valid_num_correct / valid_num_total
        self.valid_losses.append(valid_loss)
        self.valid_accuracies.append(valid_accuracy)
        print('Validation loss: %.4f, accuracy: %.4f' % (valid_loss, valid_accuracy))
        self.writer.add_scalar('valid_loss', valid_loss, epoch)
        self.writer.add_scalar('valid_acc', valid_accuracy, epoch)