from __future__ import annotations
import os
import shutil

import torch
from torch import optim
from torch.nn import Module
from torch.nn.modules import loss
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau


class Trainer:

    def __init__(self,
                 model: Module,
                 train_loader: DataLoader,
                 valid_loader: DataLoader,
                 checkpoint_dir: os.PathLike,
                 criterion: loss.Loss,
                 scheduler: ReduceLROnPlateau,
                 optimizer: optim.Optimizer,
                 writer = None):

        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.num_train = len(train_loader.dataset)
        self.checkpoint_dir = checkpoint_dir
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        self.criterion = criterion
        self.optimizer = optimizer
        self.sheduler = scheduler

        self.best_acc = 0
        self.writer = writer

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        else:
            shutil.rmtree(self.checkpoint_dir)
            os.makedirs(self.checkpoint_dir)

    def train(self, epoch: int):
        self.model.train()
        for i, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            if (i + 1) % 10 == 0:
                outputs = outputs > 0.5
                acc = (outputs == labels).float().mean().item()

                print(
                    f'Epoch: {epoch} Iteration: {i + 1}, Loss: {loss.item():.4f}, Accuracy: {acc:.4f}'
                )
        
        if self.writer is not None:
            self.writer.add_scalar('Loss/train', loss.item(), epoch + 1)
            self.writer.add_scalar('Accuracy/train', acc, epoch + 1)
        
        save_path = os.path.join(self.checkpoint_dir, f'{epoch}.pth')
        last_path = os.path.join(self.checkpoint_dir, 'last.pth')
        torch.save(self.model.state_dict(), save_path)
        shutil.copyfile(save_path, last_path)

    def validation(self, epoch: int):
        self.model.eval()
        with torch.no_grad():
            val_loss = 0
            val_acc = 0
            for i, (images, labels) in enumerate(self.valid_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()

                outputs = outputs > 0.5
                acc = (outputs == labels).float().mean().item()
                val_acc += acc
            val_loss /= len(self.valid_loader)
            val_acc /= len(self.valid_loader)

            print(f'Epoch: {epoch} Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}')

            if val_acc > self.best_acc:
                self.best_acc = val_acc
                save_path = os.path.join(self.checkpoint_dir, 'best.pth')
                torch.save(self.model.state_dict(), save_path)
        
        if self.writer is not None:
            self.writer.add_scalar('Loss/valid', val_loss, epoch + 1)
            self.writer.add_scalar('Accuracy/valid', val_acc, epoch + 1)
        
        self.sheduler.step(val_loss)