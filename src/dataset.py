from torch.utils.data import Dataset
import torch, os
from torch.nn import functional as F
import pandas as pd
import numpy as np
import gzip

class DeepTYLCV_Dataset(Dataset):
    def __init__(self, dataset_config, headers, labels):
        self.dataset_config = dataset_config
        self.max_length = int(dataset_config.max_length)
        self.use_mean   = bool(dataset_config.mean)
        self.headers    = list(headers)
        self.labels     = list(labels)

        # --- conventional features from Parquet ---
        df = pd.read_parquet(dataset_config.conv_lmdb_path)
        if df.index.name is None:
            if "header" in df.columns:
                df = df.set_index("header")
            else:
                raise ValueError("Parquet must have an index or a 'header' column.")
        df = df.astype(np.float32).sort_index()

        # stash as tensors for only the needed headers
        self.conv = {}
        for k in self.headers:
            if k not in df.index:
                raise KeyError(f"Conventional features missing for key: {k}")
            self.conv[k] = torch.from_numpy(df.loc[k].to_numpy(copy=False)).float().contiguous()

        # --- PLM feature paths (per-key .pt.gz files) ---
        self.feat_paths = [os.path.join(dataset_config.data_root, name)
                           for name in dataset_config.feature_list]

        # preload all PLM tensors into memory
        self.data = {}
        for key in self.headers:
            self.data[key] = []
            for path in self.feat_paths:
                # Now looking for .pt.gz files instead of .pt
                f_gz = os.path.join(path, f"{key}.pt.gz")
                f_pt = os.path.join(path, f"{key}.pt")
                
                # Try compressed first, fallback to uncompressed
                if os.path.exists(f_gz):
                    with gzip.open(f_gz, 'rb') as gz_file:
                        t = torch.load(gz_file, map_location="cpu")
                elif os.path.exists(f_pt):
                    t = torch.load(f_pt, map_location="cpu")
                else:
                    raise FileNotFoundError(f"Missing tensor file: {f_gz} or {f_pt}")
                
                if not isinstance(t, torch.Tensor):
                    raise TypeError(f"Not a Tensor: {f_gz or f_pt}")
                self.data[key].append(t.to(torch.float32).contiguous())

    def __len__(self):
        return len(self.headers)

    def __getitem__(self, idx):
        key    = self.headers[idx]
        X_list = [x.detach() for x in self.data[key]]  # list of [T,D] tensors
        X_conv = self.conv[key]                        # [D]

        if self.use_mean:
            # mean over time, keep time dim = 1 for consistency
            X_list = [x.mean(dim=0, keepdim=True) for x in X_list]
            masks  = [None] * len(X_list)
        else:
            padded, masks = [], []
            for x in X_list:
                T = x.size(0)
                if T > self.max_length:
                    x = x[:self.max_length]
                    T = self.max_length
                pad = self.max_length - T
                if pad > 0:
                    x = F.pad(x, (0, 0, 0, pad), value=0.0)
                m = torch.ones(self.max_length, dtype=torch.bool)
                m[:T] = False
                padded.append(x)
                masks.append(m)
            X_list = padded

        y = self.labels[idx]
        return X_list, X_conv, masks, y
