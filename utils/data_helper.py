import pandas as pd
import os
import time
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import random
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
import torch

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    
    
class Decoupling_CCL_Dataset(Dataset):

    def __init__(self, users, pos_items, num_item, N, train_u_dict, train_u_num, config_dict):

        self.users = users
        self.pos_items = pos_items
        self.num_item = num_item
        self.N = int(N)
            
        self.train_u_dict = train_u_dict
        self.train_u_num = train_u_num
        self.config_dict = config_dict

    def __len__(self):

        return len(self.users)

    def __getitem__(self, idx):
        """
            One user each time
        """
        
        user = self.users[idx]
        
        pos_item = self.pos_items[idx]
        
        neg_item = np.random.choice(self.num_item, self.N, replace = True)  # sample N neg
        

        sample = {'user':user, 'pos_item':pos_item, 'neg_item':neg_item}

        return sample 
      


def get_Decoupling_CCL_trainloader(data_reader, config_dict, g):
    
    train_dataset = Decoupling_CCL_Dataset(data_reader.train_df['user'].values, 
                               data_reader.train_df['item'].values,
                               data_reader.num_item,
                               config_dict['num_neg'],
                               data_reader.train_u_dict,
                               data_reader.train_u_num,
                               config_dict,
                                        )

    train_dataloader = DataLoader(train_dataset,
                                  batch_size = config_dict['train_batch_size'], 
                                  shuffle = True,
                                  num_workers = 4,
                                  pin_memory = True,
                                  worker_init_fn=seed_worker,
                                  generator=g
                                 )
    
    return train_dataloader




class Decoupling_CCL_Dataset_Debiased(Dataset):

    def __init__(self, users, pos_items, num_item, N, num_extra, train_u_dict, train_u_num, config_dict):

        self.users = users
        self.pos_items = pos_items
        self.num_item = num_item
        self.N = int(N)
        if num_extra <= 1: # for extra = 0, we still sample 1 postive (remove it from loss)
            self.num_extra = 1
        else:
            self.num_extra = int(num_extra)
            
        self.train_u_dict = train_u_dict
        self.train_u_num = train_u_num
        self.config_dict = config_dict
        self.K = config_dict['K'] # (1.25)

    def __len__(self):

        return len(self.users)

    def __getitem__(self, idx):
        """
            One user each time
        """
        
        user = self.users[idx]
        
        pos_item = self.pos_items[idx]
        
        neg_item = np.random.choice(self.num_item, self.N, replace = True)  # sample N neg
        
        
        extra_item = np.random.choice(self.train_u_dict[user], self.num_extra, replace = True)

        tu = (self.train_u_num[user] * self.K)/self.num_item
        

        sample = {'user':user, 'pos_item':pos_item, 'neg_item':neg_item, 'extra_item': extra_item, 'tu':tu}

        return sample 



def get_Decoupling_CCL_trainloader_Debiased(data_reader, config_dict, g):
    
    train_dataset = Decoupling_CCL_Dataset_Debiased(data_reader.train_df['user'].values, 
                               data_reader.train_df['item'].values,
                               data_reader.num_item,
                               config_dict['num_neg'],
                               config_dict['num_extra_pos'],
                               data_reader.train_u_dict,
                               data_reader.train_u_num,
                               config_dict,
                                        )

    train_dataloader = DataLoader(train_dataset,
                                  batch_size = config_dict['train_batch_size'], 
                                  shuffle = True,
                                  num_workers = 4,
                                  pin_memory = True,
                                  worker_init_fn=seed_worker,
                                  generator=g
                                 )
    
    return train_dataloader







class Data_Reader(object):
    
    def __init__(self, config_dict):
        
        self.config_dict = config_dict
        dataset = config_dict['dataset']
        path_dict = config_dict['dataset_config'][dataset]
        train_path, test_path, valid_path = path_dict['train_data'], path_dict['test_data'], path_dict['valid_data']

        self.train_df = self.process_df(pd.read_csv(train_path))
        
        self.test_df = self.process_df(pd.read_csv(test_path))

        self.valid_df = self.process_df(pd.read_csv(valid_path))
        
        self.train_df.drop_duplicates(subset = ['user', 'item'], inplace = True)

        
        self.train_u_dict = self.train_df.groupby('user')['item'].apply(list).to_dict()
        
        self.train_u_num = self.train_df.groupby('user')['item'].size() # shows for each user how many pos item this use has
        
        self.test_u_dict = self.test_df.groupby('user')['item'].apply(list).to_dict()

        self.valid_u_dict = self.test_df.groupby('user')['item'].apply(list).to_dict()
        
        self.num_user = max(self.train_df['user'].values.max(), self.test_df['user'].values.max(), self.valid_df['user'].values.max()) + 1
        
        self.num_item = max(self.train_df['item'].values.max(), self.test_df['item'].values.max(), self.valid_df['item'].values.max()) + 1
        
        
        # lightgcn configs
        self.n_users, self.m_items = self.num_user, self.num_item
        self.split = config_dict['split']
        self.folds = config_dict['folds']
        self.path = 'adj_' + dataset
        self.Graph = None
        
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        self.UserItemNet = csr_matrix((np.ones(len(self.train_df)), 
                                       (self.train_df['user'].values, self.train_df['item'].values)),
                                      shape=(self.n_users, self.m_items))
        
        self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()   # user degree
        self.users_D[self.users_D == 0.] = 1
        self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()   # item degree
        self.items_D[self.items_D == 0.] = 1.

        
    def process_df(self, raw_total_df):
    
        total_df = raw_total_df.copy()
        total_df.drop(['label', 'user_history', 'user_id'], axis=1, inplace = True)
        total_df.rename(columns={"query_index": "user", "corpus_index": "item"}, inplace = True)

        return total_df
        

    def get_train(self):
        
        return self.train_df['user'].values, self.train_df['item'].values
    
    def get_num(self):
        
        return self.num_user, self.num_item
    
    
    
    def _split_A_hat(self, A):
        A_fold = []
        fold_len = (self.n_users + self.m_items) // self.folds
        for i_fold in range(self.folds):
            start = i_fold*fold_len
            if i_fold == self.folds - 1:
                end = self.n_users + self.m_items
            else:
                end = (i_fold + 1) * fold_len
            A_fold.append(self._convert_sp_mat_to_sp_tensor(A[start:end]).coalesce().to(world.device))
        return A_fold

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))
        
    def getSparseGraph(self, device):
        print("loading adjacency matrix")
        try:
            pre_adj_mat = sp.load_npz(self.path + '/s_pre_adj_mat.npz')
            print("successfully loaded...")
            norm_adj = pre_adj_mat
        except :
            print("generating adjacency matrix")
            s = time.time()
            adj_mat = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
            adj_mat = adj_mat.tolil()
            R = self.UserItemNet.tolil()
            adj_mat[:self.n_users, self.n_users:] = R
            adj_mat[self.n_users:, :self.n_users] = R.T
            adj_mat = adj_mat.todok()

            rowsum = np.array(adj_mat.sum(axis=1))
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat = sp.diags(d_inv)

            norm_adj = d_mat.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat)
            norm_adj = norm_adj.tocsr()
            end = time.time()
            print(f"costing {end-s}s, saved norm_mat...")
            sp.save_npz(self.path + '/s_pre_adj_mat.npz', norm_adj)

        if self.split == True:
            self.Graph = self._split_A_hat(norm_adj)
            print("done split matrix")
        else:
            self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
            self.Graph = self.Graph.coalesce().to(device)
            print("don't split the matrix")
        return self.Graph
    
    
