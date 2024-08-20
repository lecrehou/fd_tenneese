#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch
import logging
import os
import time
import warnings
import math
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, Dataset, Sampler
import pandas as pd
from torch_geometric.data import DataLoader
import models2
import datasets
from sklearn.metrics import classification_report,accuracy_score

from utils.save import Save_Tool
from utils.freeze import set_freeze_by_id
from utils.metrics import Accuracy_Score


import numpy as np

import random
df_loss = pd.DataFrame(columns=['epoch','accury', 'epoch loss'])
df_loss.to_csv("train_loss1.csv",index=False)
df1_loss = pd.DataFrame(columns=['epoch', 'accury','epoch loss'])
df1_loss.to_csv("test_loss1.csv",index=False)


class train_utils(object):
    def __init__(self, args, save_dir):
        self.args = args
        self.save_dir = save_dir

    def setup(self):
        """
        Initialize the datasets, model, loss and optimizer
        :param args:
        :return:
        """
        args = self.args

        # Consider the gpu or cpu condition
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            logging.info('using {} gpus'.format(self.device_count))
            assert args.batch_size % self.device_count == 0, "batch size should be divided by device count"
        else:
            warnings.warn("gpu is not available")
            self.device = torch.device("cpu")
            self.device_count = 1
            logging.info('using {} cpu'.format(self.device_count))

        torch.manual_seed(args.rand_seed)
        # Load the datasets
        Dataset = getattr(datasets, args.data_name)
        self.datasets = {}

        self.datasets['train'],self.datasets['val'] = Dataset(args.data_dir, args.data_file).data_preprare()
        
        self.dataloaders = {x: DataLoader(self.datasets[x],
                                          batch_size=args.batch_size,
                                          shuffle= True,
                                          num_workers=args.num_workers,
                                          pin_memory=(True if self.device == 'cuda' else False),
                                          drop_last=True)
                            for x in ['train', 'val']}
        self.model = getattr(models2, args.model_name)(feature=Dataset.feature, out_channel=Dataset.num_classes,
                                                       pooltype=args.pooltype, drop_rate =args.drop_rate,
                                                       backbone = args.backbone,dropping_method = args.dropping_method,
                                                       heads = args.heads, K = args.K, alpha = args.alpha,
                                                       first_layer_dimension = args.first_layer_dimension,
                                                       batch_size = args.batch_size)
        if args.layer_num_last != 0:
            set_freeze_by_id(self.model, args.layer_num_last)
        if self.device_count > 1:
            self.model = torch.nn.DataParallel(self.model)

        # Define the optimizer
        if args.opt == 'sgd':
            self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr,
                                       momentum=args.momentum, weight_decay=args.weight_decay)
        elif args.opt == 'adam':
            self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr,
                                        weight_decay=args.weight_decay)
        else:
            raise Exception("optimizer not implement")

        # Define the learning rate decay
        if args.lr_scheduler == 'step':
            steps = [int(step) for step in args.steps.split(',')]
            self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, steps, gamma=args.gamma)
        elif args.lr_scheduler == 'exp':
            self.lr_scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, args.gamma)
        elif args.lr_scheduler == 'stepLR':
            self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, args.steps, args.gamma)
        elif args.lr_scheduler == 'fix':
            self.lr_scheduler = None
        else:
            raise Exception("lr schedule not implement")

        # Define the monitoring accuracy
        if args.monitor_acc == 'Detection':
            self.cal_acc = Accuracy_Score
        else:
            raise Exception("monitor_acc is not implement")

        # Load the checkpoint
        self.start_epoch = 0
        if args.resume:
            suffix = args.resume.rsplit('.', 1)[-1]
            if suffix == 'tar':
                checkpoint = torch.load(args.resume)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.start_epoch = checkpoint['epoch'] + 1
            elif suffix == 'pth':
                self.model.load_state_dict(torch.load(args.resume, map_location=self.device))

        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss(reduction='none')

    def train(self):
        """
        Training process
        :return:
        """
        args = self.args
        step = 0
        best_acc = 0.80
        batch_count = 0
        batch_loss = 0.0
        step_start = time.time()
        acc_df = pd.DataFrame(columns=('epoch','loss','accuracy'))
        save_list = Save_Tool(max_num=args.max_model_num)

        torch.manual_seed(args.rand_seed)

        for epoch in range(self.start_epoch, args.max_epoch):
            logging.info('-'*5 + 'Epoch {}/{}'.format(epoch, args.max_epoch - 1) + '-'*5)
            # Update the learning rate
            if self.lr_scheduler is not None:
                # self.lr_scheduler.step(epoch)
                logging.info('current lr: {}'.format(self.lr_scheduler.get_last_lr()))
            else:
                logging.info('current lr: {}'.format(args.lr))

            # Each epoch has a training and val phase
            for phase in ['train', 'val']:
                # Define the temp variable
                epoch_start = time.time()
                epoch_loss = 0.0
                y_labels = np.zeros((0,))
                y_pre = np.zeros((0,))
                y_labels_list = []
                y_pre_list = []
                # Set model to train mode or test mode
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()
                if epoch >=0:
                    
                    for data in self.dataloaders[phase]:
                        inputs = data.to(self.device)
                        labels = inputs.y
                        # Do the learning process, in val, we do not care about the gradient for relaxing
                        with torch.set_grad_enabled(phase == 'train'):
                             # Calculate the training information
                            if phase == 'train':
                                logits, logits1 = self.model(inputs, args.drop_rate)

                                y_labels = np.concatenate((y_labels, labels.view(-1).cpu().detach().numpy()), axis=0)
                                y_pre = np.concatenate((y_pre, logits1.view(-1).cpu().detach().numpy()), axis=0)
                                # labels = torch.where(labels == 3, torch.tensor(0), torch.where(labels == 9, torch.tensor(1), labels))
                                loss = self.criterion(logits, labels)
                                loss = torch.mean(loss)
                                epoch_loss = loss.item()
                                loss.requires_grad_(True)

                                self.optimizer.zero_grad()
                                loss.backward()
                                self.optimizer.step()
                            if phase == 'val':

                                logits, logits1 = self.model(inputs, args.drop_rate)
                                y_labels = np.concatenate((y_labels, labels.view(-1).cpu().detach().numpy()), axis=0)
                                y_pre = np.concatenate((y_pre, logits1.view(-1).cpu().detach().numpy()), axis=0)

                                loss = self.criterion(logits, labels)
                                loss = torch.mean(loss)
                                epoch_loss = loss.item()

                    classification = classification_report(y_labels,y_pre,output_dict=True)
                    accuracy = accuracy_score(y_labels, y_pre, normalize=True)
                    df = pd.DataFrame(classification).transpose()
#                 online learning
#                 if epoch >=0:
#                     if phase =='train':
#
#                         random.shuffle(self.datasets['train'])
#                         data_loader_train = DataLoader(self.datasets['train'], batch_size=64, shuffle=False)
#                         top_k = 675
#                         topk_indices_all = []
# #                         topk_loss_all = []
#                         topk_loss_all = torch.tensor([], device=self.device)
#                         indices_offset = 0
#                         # 遍历数据批次
#                         for batch in data_loader_train:
#                             inputs = batch.to(self.device)
#                             labels = inputs.y
#                             with torch.set_grad_enabled(False):
#                                 logits, logits1 = self.model(inputs, args.drop_rate)
#                                 loss = self.criterion(logits, labels)
#                                 # 将损失添加到 topk_loss_all
#                                 topk_loss_all = torch.cat([topk_loss_all, loss])
#
# #                                 if len(loss) == 64:
# #                                     topk_losses, topk_indices = torch.topk(loss, top_k)
# #                                     # 计算每个Top-K损失值在原始数据集中的索引位置
# #                                     original_indices = topk_indices + indices_offset
# #                                     # 添加Top-K损失值和索引到相应的列表
# #                                     topk_indices_all = np.concatenate((topk_indices_all, original_indices.cpu().detach().nump,y()), axis=0)
# #                                     del topk_losses, topk_indices
# #                             indices_offset += len(batch)
#                         topk_losses, topk_indices = torch.topk(topk_loss_all, top_k)
# #                         topk_losses_1, topk_indices_1 = torch.topk(-topk_loss_all, top_k)
#                         print(len(topk_indices))
# #
#                         topk_samples = [self.datasets['train'][int(i)] for i in topk_indices]
#
#                         # 将每个二维特征数组展平成一维数组，并记录原始形状
#                         flattened_features = []
#                         original_labels = []
#
#                         for data_ori in topk_samples:
#                             flattened_features.append(data_ori['x'].flatten())
#                             original_labels.append(data_ori['y'])
#
#                         topk_samples = self.datasets['train'] + topk_samples
#
#                         data_loader_train1 = DataLoader(topk_samples, batch_size=64, shuffle=True)
#                         del topk_samples
#                         for batch in data_loader_train1:
#                             inputs = batch.to(self.device)
#                             inputs.requires_grad = True
#                             labels = inputs.y
#                             with torch.set_grad_enabled(True):
#                                 logits, logits1 = self.model(inputs, args.drop_rate)
#                                 y_labels_list.append(labels.view(-1).cpu().detach().numpy())
#                                 y_pre_list.append(logits1.view(-1).cpu().detach().numpy())
#
#                                 loss = self.criterion(logits, labels)
#                                 loss = torch.mean(loss)
#
#                                 epoch_loss = loss.item()
#                                 self.optimizer.zero_grad()
#                                 loss.backward()
#                                 self.optimizer.step()
#                         y_labels = np.concatenate(y_labels_list, axis=0)
#                         y_pre = np.concatenate(y_pre_list, axis=0)
#                         classification = classification_report(y_labels, y_pre, output_dict=True)
#                         accuracy = accuracy_score(y_labels, y_pre, normalize=True)
#                         df = pd.DataFrame(classification).transpose()
#
#
#                     if phase == 'val':
#                         data_loader_val = DataLoader(self.datasets['val'], batch_size=64, shuffle=True)
#                         for data in data_loader_val:
#                             inputs = data.to(self.device)
#                             labels = inputs.y
#                             with torch.set_grad_enabled(phase == 'train'):
#                                 logits, logits1 = self.model(inputs, args.drop_rate)
#                                 y_labels_list.append(labels.view(-1).cpu().detach().numpy())
#                                 y_pre_list.append(logits1.view(-1).cpu().detach().numpy())
#                             loss = self.criterion(logits, labels)
#                             loss = torch.mean(loss)
#                             epoch_loss = loss.item()
#                         y_labels = np.concatenate(y_labels_list, axis=0)
#                         y_pre = np.concatenate(y_pre_list, axis=0)
#                         classification = classification_report(y_labels, y_pre, output_dict=True)
#                         accuracy = accuracy_score(y_labels, y_pre, normalize=True)
#                         df = pd.DataFrame(classification).transpose()
#                     torch.cuda.empty_cache()
#

                if phase == 'val':
                    # 将val_epoch_accuracy保存到excel中
                    loss_list = [epoch,accuracy,epoch_loss]
                    loss_data = pd.DataFrame([loss_list])
                    loss_data.to_csv('test_loss1.csv', mode='a', header=False,index=False)
                    filename = 'result1/val_epoch_{}.csv'.format(epoch)
                    df.to_csv(filename, index=True )
                    logging.info('Epoch: {} ,{}-Loss: {:.4f} {},Cost {:.1f} sec'.format( epoch, phase, epoch_loss, phase, time.time() - epoch_start))
                    # save the model
                    if epoch >= args.max_epoch - 10:  # take the average of the last 5 epochs

                        acc_df = acc_df.append(
                            pd.DataFrame({'epoch': [epoch],
                                          'loss': [loss],
                                          'accuracy': [accuracy]}), ignore_index=True)

                    # save the checkpoint for other learning
                    model_state_dic = self.model.module.state_dict() if self.device_count > 1 else self.model.state_dict()
                    save_path = os.path.join(self.save_dir, '{}_ckpt.tar'.format(epoch))
                    torch.save({
                        'epoch': epoch,
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'model_state_dict': model_state_dic
                    }, save_path)
                    save_list.update(save_path)
                    # save the best model according to the val accuracy
                    if accuracy > best_acc or epoch == args.max_epoch - 1:
                        best_acc = accuracy
                        logging.info("save best model epoch {}, acc {:.4f}".format(epoch, best_acc))
                        torch.save(model_state_dic,
                                   os.path.join(self.save_dir, '{}-{:.4f}-best_model.pth'.format(epoch, best_acc)))

                    if epoch == args.max_epoch - 1:
                        # acc_df.to_csv('Results_'+args.model_name+'_'+args.pooltype+ '_'+args.data_file+'.csv', sep=",", index=False)
                        acc_means = acc_df.mean()
                        logging.info(
                            "loss {:.4f}, acc {:.4f}".format(acc_means['loss'], acc_means['accuraacy'], ))


                elif phase == 'train':
                    # 将train_epoch_accuracy存储在excel中
                    loss_list = [epoch,accuracy,epoch_loss]
                    loss_data = pd.DataFrame([loss_list])
                    loss_data.to_csv('train_loss1.csv', mode='a', header=False, index=False)
                    filename = 'result1/train_epoch_{}.csv'.format(epoch)
                    df.to_csv(filename, index=True)
                    logging.info('Epoch: {} ,{}-Loss: {:.4f} {},Cost {:.1f} sec'.format(
                            epoch, phase, epoch_loss,phase,time.time()-epoch_start))


            if self.lr_scheduler is not None:
                self.lr_scheduler.step()














