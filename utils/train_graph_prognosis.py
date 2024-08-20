#!/usr/bin/python
# -*- coding:utf-8 -*-

import argparse
import os
from datetime import datetime
from utils.logger import setlogger
import logging
from utils.train_utils_graph import train_utils
import warnings
import torch
warnings.filterwarnings("ignore")
args = None

def parse_args():

    parser = argparse.ArgumentParser(description='train')

    # basic parameters
    parser.add_argument('--model_name', type=str, default='Resnet', help='the name of the model')
    parser.add_argument('--data_name', type=str, default='TE_process_graph', help='the name of the data')
    parser.add_argument('--data_file', type=str, default='FD00hou1_52_1', help='the file of the data')

    parser.add_argument('--data_dir', type=str, default='./data/data_pkl/', help='the directory of the data')
    parser.add_argument('--monitor_acc', type=str, default='Detection', help='the performance score')
    parser.add_argument('--cuda_device', type=str, default='1', help='assign device')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint',
                        help='the directory to save the model')
    parser.add_argument('--batch_size', type=int, default=64, help='batchsize of the training process')
    parser.add_argument('--num_workers', type=int, default=0, help='the number of training process')

    # Define the tasks
    parser.add_argument('--task', choices=[ 'Graph'], type=str,
                        default='Graph', help='Graph regression only.')
    parser.add_argument('--pooltype', choices=['TopKPool', 'EdgePool', 'ASAPool', 'SAGPool'],type=str,
                        default='SAGPool', help='For the Graph classification task')
    parser.add_argument('--hidden_channels', type=int, default=1024)


    # optimization information
    parser.add_argument('--layer_num_last', type=int, default=0, help='the number of last layers which unfreeze')
    parser.add_argument('--opt', type =str, choices=['sgd', 'adam'], default='adam', help='the optimizer')
    parser.add_argument('--lr', type=float, default=0.001, help='the initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='the momentum for sgd')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='the weight decay')
    parser.add_argument('--lr_scheduler', type=str, choices=['step', 'exp', 'stepLR', 'fix', 'cos'], default='step',
                        help='the learning rate schedule')
    parser.add_argument('--gamma', type=float, default=0.1, help='learning rate scheduler parameter for step and exp')
    parser.add_argument('--steps', type=str, default='20, 20', help='the learning rate decay for step and stepLR')

    # save, load and display information
    parser.add_argument('--resume', type=str, default='', help='the directory of the resume training model')
    parser.add_argument('--max_model_num', type=int, default=1, help='the number of most recent models to save')
    parser.add_argument('--max_epoch', type=int, default=50, help='max number of epoch')
    parser.add_argument('--print_step', type=int, default=200, help='the interval of log training information')

    # 新增
    parser.add_argument('--drop_rate', type=float, default= 0.2, help='The chosen dropping rate (default: 0).')
    parser.add_argument('--backbone', type=str, default='GAT', help='The backbone model [GCN, GAT, APPNP].')
    parser.add_argument('--dropping_method', type=str, default='DropMessage',
                        help='The chosen dropping method [Dropout, DropEdge, DropNode, DropMessage].')
    parser.add_argument('--heads', type=int, default=1, help='The head number for GAT (default: 1).')
    parser.add_argument('--K', type=int, default=10, help='The K value for APPNP (default: 10).')
    parser.add_argument('--alpha', type=float, default=0.1, help='The alpha value for APPNP (default: 0.1).')
    parser.add_argument('--first_layer_dimension', type=int, default=64,
                        help='The hidden dimension number for two-layer GNNs (default: 16).')
    parser.add_argument('--num_filter', type = int, help='The hyperparameter m in the paper', default=11)
    parser.add_argument('--max_nodes', type=int, help='', default=52)
    parser.add_argument('-r', '--rand_seed', type=int, default=0, help='The random seed (default: 0).')

    args = parser.parse_args()
    # random seed setting
    random_seed = args.rand_seed
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    return args




if __name__ == '__main__':

    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device.strip()
    # Prepare the saving path for the model
    sub_dir = args.model_name + '_' + args.pooltype+ '_'+ datetime.strftime(datetime.now(), '%m%d-%H%M%S')
    save_dir = os.path.join(args.checkpoint_dir, sub_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # set the logger
    setlogger(os.path.join(save_dir, 'train.log'))

    # save the args
    for k, v in args.__dict__.items():
        logging.info("{}: {}".format(k, v))

    trainer = train_utils(args, save_dir)
    trainer.setup()
    trainer.train()
