from datasets.TE_process_graph_Datasets import datasets2
import numpy as np
import pandas as pd


class TE_process_graph(object):
    num_classes = 21
    feature = 500
    feature1 = 500

    def __init__(self, data_dir, data_file):
        self.data_dir = data_dir
        self.data_file = data_file

    def data_preprare(self, test=True):
        if test:
            path = self.data_dir + 'val_' + self.data_file + '.pkl'
            edge_index = np.load(self.data_dir + 'edge_index_' + self.data_file + '.npy')
            test_dataset = datasets2(root=path, edge_index = edge_index)
            return test_dataset

        else:
            train_path = self.data_dir + 'train_' + self.data_file + '.pkl'
            test_path = self.data_dir + 'test_' + self.data_file + '.pkl'

            edge_index = np.load(self.data_dir + 'edge_index_' + self.data_file + '.npy')
            train_dataset = datasets2(root=train_path, edge_index = edge_index)
            val_dataset = datasets2(root=test_path, edge_index = edge_index)
            return train_dataset,val_dataset
