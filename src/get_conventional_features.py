from argparse import ArgumentParser
import pickle

import pandas as pd
# from src.utils import get_config
# from src.lightning_module import CONTRA_IL6_Module, CONTRA_IL6_CONV
# from src.dataset import IL6_Dataset,IL6_Dataset_LMDB_embedding
# from src.metrics import calculate_metrics
import torch
from torch.utils.data import DataLoader, Subset
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor,Callback

from sklearn.model_selection import StratifiedKFold
import os
import copy
import yaml
import numpy as np
from tqdm import tqdm
from lightning.pytorch.loggers import TensorBoardLogger
import glob
from Bio import SeqIO
import warnings
from sklearn.preprocessing import StandardScaler
from pytorch_lightning.callbacks import Callback
from typing import Any, Dict
warnings.filterwarnings("ignore")

def get_conventional_features(dir, scaler_dir, Mode='Train'):
    conventional_features_dict = {}
    feature_dfs = []
    formatted_header_list = []

    file_list = sorted(glob.glob(os.path.join(dir, '*', '*.txt')))
    print(file_list)
    print(f"Found {len(file_list)} descriptor files.")

    for i, file in enumerate(file_list):
        df = pd.read_csv(file, sep='\s+', header=None)

        if i == 0:
            header_list = df.iloc[:, 0].tolist()
            for header in header_list:
                if Mode in ['Train', 'Test']:
                    header_formatted = header.split('>')[1].replace('.', '_')
                else:
                    header_formatted = header.replace('>', '')
                formatted_header_list.append(header_formatted)

        if isinstance(df.iloc[0, 0], str):
            df = df.drop(columns=[0, df.columns[-1]])  # drop ID and label
        feature_dfs.append(df)

    # Combine and rename columns
    combined_features = pd.concat(feature_dfs, axis=1)
    combined_features.columns = [f"feature_{i}" for i in range(combined_features.shape[1])]

    # Load selected feature list
    with open('/data/Vinoth_SDATA/TYLCV_Proj/3-model/conventional_model_code/selected_features_final_with_new_dataset.pkl', 'rb') as f:
        optimal_features_list = pickle.load(f)

    optimal_features = combined_features[optimal_features_list]
    feature_list = optimal_features.values  # Keep as NumPy array for sklearn compatibility

    # Scale features
    scaler_path = os.path.join(scaler_dir, 'scaler.pkl')
    if Mode == 'Train':
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(feature_list)
        os.makedirs(scaler_dir, exist_ok=True)
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"Scaler saved to {scaler_path}")
    else:
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler file not found at {scaler_path}. Please run training first.")
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        scaled_features = scaler.transform(feature_list)

    # Convert to tensor and store in dictionary
    for header, features_scaled in zip(formatted_header_list, scaled_features):
        conventional_features_dict[header] = torch.tensor(features_scaled, dtype=torch.float)
    # print(len(conventional_features_dict['Severe_C1_PP179265_1_C1']),'frist_elemme)nt'
    print(f"Loaded {len(conventional_features_dict)} conventional features (after scaling)")
    return conventional_features_dict

dir='/data/Vinoth_SDATA/TYLCV_Proj/2-features/3-external_validation/2-conventional-descriptors/'
scaler_dir='/data/Vinoth_SDATA/TYLCV_Proj/7-consistent_model_code/IL-6_Architecture_ms_cnn/CONTRA-IL6-main/results_19_7/config/Top-2-final/sweeper_0/'
your_dict=get_conventional_features(dir, scaler_dir, Mode='Test')
# print(conv_dict)

import os
import torch
from lmdb_embeddings.writer import LmdbEmbeddingsWriter

# Your dictionary with tensors
# your_dict = {
#     '11_Mild_C4': tensor([-0.1670, -1.9180, -1.8507, ...]),  # your tensor data
#     # ... other entries
# }

# Setup LMDB output folder
OUTPUT_DATABASE_FOLDER = '/data/Vinoth_SDATA/TYLCV_Proj/survey_dataset/lmdb_files/conventional'
os.makedirs(OUTPUT_DATABASE_FOLDER, exist_ok=True)

# Extract words (keys) and vectors (values) from your dictionary
words = list(your_dict.keys())
vectors = list(your_dict.values())

def iter_embeddings():
    for word, vector in zip(words, vectors):
        # Convert tensor to numpy array on CPU
        yield word, vector.detach().cpu().numpy()

print('Writing vectors to a LMDB database...')

# Create the LMDB writer and save
writer = LmdbEmbeddingsWriter(iter_embeddings()).write(OUTPUT_DATABASE_FOLDER)

