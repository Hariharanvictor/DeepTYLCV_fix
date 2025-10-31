from torch.utils.data import Dataset
import torch
import os
from torch.nn import functional as F
from lmdb_embeddings.reader import LmdbEmbeddingsReader
from lmdb_embeddings.exceptions import MissingWordError
# from utils import extract_headers_labels

class IL6_Dataset(Dataset):
    def __init__(self, dataset_config,headers,labels, conv_features):
        self.feat_paths = []
        # self.neg_feat_paths = []
        self.dataset_config = dataset_config
        self.embedding_types = 'mean_representations' if dataset_config.mean else 'representations'
        self.max_length = dataset_config.max_length
        self.headers = headers
        self.labels = labels
        self.conv_features = conv_features
        for feature_name in dataset_config.feature_list:
            self.feat_paths.append(
                os.path.join(dataset_config.data_root, feature_name)
            )
        self._setup_keys()
        self._preload_data()
    
    def get_pep_keys(self):
        return self.pep_keys
        
    def _setup_keys(self):
        self.pep_keys = [i for i in self.headers]
        
    def _preload_data(self):
        self.data = {}
        self.data_conv = {}
        
        for key in self.pep_keys:    
            self.data[key] = [
                torch.load(os.path.join(path, f'{key}.pt'))
                for path in self.feat_paths
            ]
            self.data_conv[key] = self.conv_features[key]
            
            
        
        
    
    def __len__(self):
        return len(self.pep_keys)
    
    def __getitem__(self, idx):
        pep_key = self.pep_keys[idx]
        X = self.data[pep_key]
        X_conv = self.data_conv[pep_key]
        
        len_tokens = [x.shape[0] if not self.dataset_config.mean else 1 for x in X]
        
        masks_X = [None for l in len_tokens]
        
        X = [x.detach() for x in X]
        
        if not self.dataset_config.mean:
            for i in range(len(X)):
                X[i] = F.pad(X[i], (0, 0, 0, self.max_length - X[i].size(0)), value=0)
                masks_X[i] = torch.ones(self.max_length, dtype=torch.bool)
                masks_X[i][:len_tokens[i]] = False
        
        return X, X_conv, masks_X, self.labels[idx]
    
    
class IL6_Dataset_LMDB_embedding(Dataset):
    def __init__(self, dataset_config,headers,labels):
        self.feat_paths = []
        # self.neg_feat_paths = []
        self.dataset_config = dataset_config
        self.embedding_types = 'mean_representations' if dataset_config.mean else 'representations'
        self.max_length = dataset_config.max_length
        self.headers = headers
        self.labels = labels
        # self.embeddings=LmdbEmbeddingsReader(dataset_config.plm_lmdb_path)
        self.conv_features=LmdbEmbeddingsReader(dataset_config.conv_lmdb_path)
        for feature_name in dataset_config.feature_list:
            self.feat_paths.append(
                os.path.join(dataset_config.data_root, feature_name)
            )
        self.embedding_list=[LmdbEmbeddingsReader(feat_path) for feat_path in self.feat_paths]
        # for feature_name in dataset_config.feature_list:
        #     self.feat_paths.append(
        #         os.path.join(dataset_config.data_root, feature_name)
        #     )
        self._setup_keys()
        # self._preload_data()
    
    def get_pep_keys(self):
        return self.pep_keys
        
    def _setup_keys(self):
        self.pep_keys = [i for i in self.headers]
        
    # def _preload_data(self):
    #     self.data = {}
    #     self.data_conv = {}
        
    #     for key in self.pep_keys:    
    #         self.data[key] = [
    #             torch.from_numpy(self.embeddings.get_word_vector(key))
    #         ]
    #         self.data_conv[key] = self.conv_features[key]
            
            
        
        
    
    def __len__(self):
        return len(self.pep_keys)
    
    def __getitem__(self, idx):
        pep_key = self.pep_keys[idx]
        X = [torch.from_numpy(embedding.get_word_vector(pep_key)) for embedding in self.embedding_list]
        # print(X.shape)
        X_conv = self.conv_features.get_word_vector(pep_key)
        
        len_tokens = [x.shape[0] if not self.dataset_config.mean else 1 for x in X]
        
        masks_X = [None for l in len_tokens]
        
        X = [x.detach() for x in X]
        
        if not self.dataset_config.mean:
            for i in range(len(X)):
                X[i] = F.pad(X[i], (0, 0, 0, self.max_length - X[i].size(0)), value=0)
                masks_X[i] = torch.ones(self.max_length, dtype=torch.bool)
                masks_X[i][:len_tokens[i]] = False
        
        return X, X_conv, masks_X, self.labels[idx]