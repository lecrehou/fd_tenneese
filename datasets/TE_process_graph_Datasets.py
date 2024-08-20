import pandas as pd
from torch_geometric.data import Data
import torch
from tqdm import tqdm
# from imblearn.over_sampling import SMOTE
import numpy as np
def datasets2(root, edge_index):
    anno_pd = pd.read_pickle(root)
    data_list = []
    features = anno_pd['data'].tolist()
    label = anno_pd['label'].tolist()
    # simulationRun = anno_pd['simulationRun'].tolist()
    for i in tqdm(range(len(features))):
        x = features[i].T
        node_features = torch.tensor(x, dtype=torch.float)
        graph_label = torch.tensor([label[i]], dtype=torch.long)
        edge = torch.tensor(edge_index, dtype=torch.long)
        data = Data(x=node_features, y=graph_label, edge_index=edge)
        data_list.append(data)
    return data_list

#     if root == './data/data_pkl/train_FD00hou.pkl':
#         anno_pd= pd.read_pickle(root)
#         data_list = []
#         features = anno_pd['data'].tolist()
#         label = anno_pd['label'].tolist()


#         # 将每个二维特征数组展平成一维数组，并记录原始形状
#         flattened_features = []
#         original_shapes = []

#         for feature in features:
#             flattened_features.append(feature.flatten())
#             original_shapes.append(feature.shape)
    
#         smote = SMOTE(sampling_strategy={i: 148 for i in range(1, 21)}, random_state=0)
#         X_resampled, y_minority_resampled = smote.fit_resample(flattened_features,label)
#         resampled_data = []
#         for feature in X_resampled:
#             original_shape = (500, 52)  # 从原始形状恢复
#             feature = np.array(feature)
#             resampled_feature = feature.reshape(original_shape)
#             resampled_data.append(resampled_feature)

#         # simulationRun = anno_pd['simulationRun'].tolist()
#         for i in tqdm(range(len(resampled_data))):
#             x = resampled_data[i].T
#             node_features = torch.tensor(x, dtype=torch.float)
#             graph_label = torch.tensor([y_minority_resampled[i]], dtype=torch.long)
#             edge = torch.tensor(edge_index, dtype=torch.long)
#             data = Data(x=node_features, y=graph_label, edge_index=edge)
#             data_list.append(data)
#     else :
#         anno_pd = pd.read_pickle(root)
#         data_list = []
#         features = anno_pd['data'].tolist()
#         label = anno_pd['label'].tolist()
#         # simulationRun = anno_pd['simulationRun'].tolist()
#         for i in tqdm(range(len(features))):
#             x = features[i].T
#             node_features = torch.tensor(x, dtype=torch.float)
#             graph_label = torch.tensor([label[i]], dtype=torch.long)
#             edge = torch.tensor(edge_index, dtype=torch.long)
#             data = Data(x=node_features, y=graph_label, edge_index=edge)
#             data_list.append(data)
#     return data_list


