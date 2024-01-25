import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
    

class MF(nn.Module):
    
    def __init__(self, user_size, item_size, config_dict):
        
        super().__init__()
        
        self.user_size = user_size
        
        self.item_size = item_size
       
                
        self.embed_size = config_dict['embedding_dim']
        
        self.cosine = config_dict['cosine_flag_train']
        
        self.dropout = nn.Dropout(config_dict['embedding_dropout'])
        
    
        self.U = nn.Embedding(self.user_size, self.embed_size)
        nn.init.normal_(self.U.weight, std = 0.1 / np.sqrt(self.embed_size))

        
        self.I = nn.Embedding(self.item_size, self.embed_size)
        nn.init.normal_(self.I.weight, std = 0.1 / np.sqrt(self.embed_size))
        
    def get_embed(self):
        
        return self.U, self.I
                
        
    def compute(self, user, pos_item, neg_item):
        '''
        user: shape = (B)
        pos_item: shape = (B)
        neg_item: shape = (B, num_neg)
        pos_extra: shape = (B, num_extra)
        '''
        
        user_vec = self.U(user)
        
        item_pos_vec = self.I(pos_item)
        
        item_neg_vec = self.I(neg_item)
                
        if self.cosine:
            
            user_vec = F.normalize(user_vec, dim = -1)
            user_vec = self.dropout(user_vec)
            
            item_pos_vec = F.normalize(item_pos_vec, dim = -1)
            item_neg_vec = F.normalize(item_neg_vec, dim = -1)

        pos_scores = (user_vec * item_pos_vec).sum(1, keepdim = True) # (B, 1)
        
        neg_scores = (user_vec.unsqueeze(1) * item_neg_vec).sum(-1) # (B, num_neg)
        
        reg = torch.norm(self.U.weight, 2) ** 2 + torch.norm(self.I.weight, 2) ** 2
        
        return pos_scores, neg_scores, reg
    
    
    
class LightGCN(nn.Module):
    
    def __init__(self, user_size, item_size, config_dict, Graph):
        
        super().__init__()
        
        self.config_dict = config_dict
        
        self.user_size = user_size
        
        self.item_size = item_size
       
                
        self.embed_size = config_dict['embedding_dim']
        
        self.cosine = config_dict['cosine_flag_train']
        
        self.dropout = nn.Dropout(config_dict['embedding_dropout'])
        
        
        # LightGCN
        self.node_drop = config_dict['node_drop']
        self.keep_prob = config_dict['keep_prob']
        self.Graph = Graph
        self.A_split = config_dict['split']
        self.n_layers = config_dict['n_layers']
        
        
        self.U = nn.Embedding(self.user_size, self.embed_size)

        self.I = nn.Embedding(self.item_size, self.embed_size)

        nn.init.xavier_uniform_(self.U.weight)
        nn.init.xavier_uniform_(self.I.weight)
        
        
    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index]/keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g
    
    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph
    
    def get_embed(self):
        """
        propagate methods for lightGCN
        """       
        users_emb = self.U.weight
        items_emb = self.I.weight
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]
        if self.node_drop:
            if self.training:         
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph        
        else:
            g_droped = self.Graph    
        
        for layer in range(int(self.n_layers)):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.user_size, self.item_size])
        return users, items 
                
        
    def compute(self, user, pos_item, neg_item):
        '''
        user: shape = (B)
        pos_item: shape = (B)
        neg_item: shape = (B, num_neg)
        pos_extra: shape = (B, num_extra)
        '''
        
        if self.config_dict['sampling_method'] == 'uniform':
            UU, II = self.get_embed()
            
            
            user_vec = UU[user]
            
            item_pos_vec = II[pos_item]
            
            item_neg_vec = II[neg_item]
            
            
            if self.cosine:
                
                user_vec = F.normalize(user_vec, dim = -1)
                user_vec = self.dropout(user_vec)
                
                item_pos_vec = F.normalize(item_pos_vec, dim = -1)
                item_neg_vec = F.normalize(item_neg_vec, dim = -1)

            pos_scores = (user_vec * item_pos_vec).sum(1, keepdim = True) # (B, 1)
            
            neg_scores = (user_vec.unsqueeze(1) * item_neg_vec).sum(-1) # (B, num_neg)
            
            reg = torch.norm(self.U(user), 2) ** 2 + torch.norm(self.I(pos_item), 2) ** 2 + torch.norm(self.I(neg_item), 2) ** 2
            
            reg = reg/len(user)
        
        elif self.config_dict['sampling_method'] == 'in-batch':
            UU, II = self.get_embed()
            
            
            user_vec = UU[user]
            
            item_pos_vec = II[pos_item]
            
                            
            if self.cosine:
                
                user_vec = F.normalize(user_vec, dim = -1)
                user_vec = self.dropout(user_vec)
                
                item_pos_vec = F.normalize(item_pos_vec, dim = -1)

            pos_scores = (user_vec * item_pos_vec).sum(1, keepdim = True) 
            
            
            all_scores = torch.mm(user_vec, item_pos_vec.t().contiguous())
            
            batch_size = user_vec.shape[0] 
            index_tensor = torch.cat([torch.arange(batch_size).long(), torch.arange(batch_size).long()]).to(self.config_dict['device'])
            column_indices_prior = torch.cat([torch.arange(batch_size).long(), torch.zeros(batch_size).long()]).to(self.config_dict['device'])
            column_indices_post = torch.cat([torch.zeros(batch_size).long(), torch.arange(batch_size).long()]).to(self.config_dict['device'])
            all_scores[index_tensor, column_indices_prior] = all_scores[index_tensor, column_indices_post]



            neg_scores = all_scores[:, 1:]
            
            reg = torch.norm(self.U(user), 2) ** 2 + torch.norm(self.I(pos_item), 2) ** 2
            reg = reg/len(user)

        
        return pos_scores, neg_scores, reg







class MF_Debiased(MF):
    
    def __init__(self, user_size, item_size, config_dict):
        
        super().__init__(user_size, item_size, config_dict)

        
    def compute(self, user, pos_item, neg_item, pos_extra):
        '''
        user: shape = (B)
        pos_item: shape = (B)
        neg_item: shape = (B, num_neg)
        pos_extra: shape = (B, num_extra)
        '''
        
        user_vec = self.U(user)
        
        item_pos_vec = self.I(pos_item)
        
        item_neg_vec = self.I(neg_item)
        
        item_ext_vec = self.I(pos_extra)
        
        if self.cosine:
            
            user_vec = F.normalize(user_vec, dim = -1)
            user_vec = self.dropout(user_vec)
            
            item_pos_vec = F.normalize(item_pos_vec, dim = -1)
            item_neg_vec = F.normalize(item_neg_vec, dim = -1)
            item_ext_vec = F.normalize(item_ext_vec, dim = -1)

        pos_scores = (user_vec * item_pos_vec).sum(1, keepdim = True) # (B, 1)
        
        neg_scores = (user_vec.unsqueeze(1) * item_neg_vec).sum(-1) # (B, num_neg)
        ext_scores = (user_vec.unsqueeze(1) * item_ext_vec).sum(-1) # (B, num_extra)
        
        reg = torch.norm(self.U.weight, 2) ** 2 + torch.norm(self.I.weight, 2) ** 2
        
        return pos_scores, neg_scores, ext_scores, reg
    



class LightGCN_Debiased(LightGCN):
    
    def __init__(self, user_size, item_size, config_dict, Graph):
        
        super().__init__(user_size, item_size, config_dict, Graph)

    
        
    def compute(self, user, pos_item, neg_item, pos_extra):
        '''
        user: shape = (B)
        pos_item: shape = (B)
        neg_item: shape = (B, num_neg)
        pos_extra: shape = (B, num_extra)
        '''
        
        if self.config_dict['sampling_method'] == 'uniform':
            UU, II = self.get_embed()
            
            
            user_vec = UU[user]
            
            item_pos_vec = II[pos_item]
            
            item_neg_vec = II[neg_item]
            
            item_ext_vec = II[pos_extra]
            
            if self.cosine:
                
                user_vec = F.normalize(user_vec, dim = -1)
                user_vec = self.dropout(user_vec)
                
                item_pos_vec = F.normalize(item_pos_vec, dim = -1)
                item_neg_vec = F.normalize(item_neg_vec, dim = -1)
                item_ext_vec = F.normalize(item_ext_vec, dim = -1)

            pos_scores = (user_vec * item_pos_vec).sum(1, keepdim = True) # (B, 1)
            
            neg_scores = (user_vec.unsqueeze(1) * item_neg_vec).sum(-1) # (B, num_neg)
            ext_scores = (user_vec.unsqueeze(1) * item_ext_vec).sum(-1) 
            
            reg = torch.norm(self.U(user), 2) ** 2 + torch.norm(self.I(pos_item), 2) ** 2 + torch.norm(self.I(neg_item), 2) ** 2 + torch.norm(self.I(pos_extra), 2) ** 2
            


            reg = reg/len(user)
        
        elif self.config_dict['sampling_method'] == 'in-batch':
            UU, II = self.get_embed()
            
            
            user_vec = UU[user]
            
            item_pos_vec = II[pos_item]
            
            
            
            item_ext_vec = II[pos_extra]
            
            if self.cosine:
                
                user_vec = F.normalize(user_vec, dim = -1)
                user_vec = self.dropout(user_vec)
                
                item_pos_vec = F.normalize(item_pos_vec, dim = -1)
                item_ext_vec = F.normalize(item_ext_vec, dim = -1)

            pos_scores = (user_vec * item_pos_vec).sum(1, keepdim = True) 
            
            
            all_scores = torch.mm(user_vec, item_pos_vec.t().contiguous())
            
            batch_size = user_vec.shape[0] 
            index_tensor = torch.cat([torch.arange(batch_size).long(), torch.arange(batch_size).long()]).to(self.config_dict['device'])
            column_indices_prior = torch.cat([torch.arange(batch_size).long(), torch.zeros(batch_size).long()]).to(self.config_dict['device'])
            column_indices_post = torch.cat([torch.zeros(batch_size).long(), torch.arange(batch_size).long()]).to(self.config_dict['device'])
            all_scores[index_tensor, column_indices_prior] = all_scores[index_tensor, column_indices_post]           

            neg_scores = all_scores[:, 1:]
            
            
            
            ext_scores = (user_vec.unsqueeze(1) * item_ext_vec).sum(-1) 
            reg = torch.norm(self.U(user), 2) ** 2 + torch.norm(self.I(pos_item), 2) ** 2 + torch.norm(self.I(pos_extra), 2) ** 2
            reg = reg/len(user)

        
        return pos_scores, neg_scores, ext_scores, reg
    





