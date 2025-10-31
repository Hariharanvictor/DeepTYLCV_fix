import os
import sys
import torch
from tqdm import tqdm
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import pandas as pd
import tempfile

# Your embedders
from bio_embeddings.embed import ESMEmbedder, ProtTransBertBFDEmbedder, ProtTransAlbertBFDEmbedder
import iFeatureOmegaCLI as ifo
from .ccd_feature_order import CCD_INFO
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "iFeatureOmegaCLI"))
import atexit, os
import pickle



class ESM:
    def __init__(self, model_name='esm1_t34_670M_UR50S', device='cpu'):
        self.embedder = ESMEmbedder(device=device)
    
    def get_features_batch(self, sequences, batch_size=4):
        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i:i+batch_size]
            embeddings = [self.embedder.embed(seq) for seq in batch_seqs]
            yield embeddings

class PTAB:
    def __init__(self, model_name="PTAB", device="cpu"):
        self.embedder = ProtTransAlbertBFDEmbedder(device=device)

    def get_features_batch(self, sequences, batch_size=4):
        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i:i+batch_size]
            embeddings = [self.embedder.embed(seq) for seq in batch_seqs]
            yield embeddings

class PTBB:
    def __init__(self, model_name="PTBB", device="cpu"):
        self.embedder = ProtTransBertBFDEmbedder(device=device)

    def get_features_batch(self, sequences, batch_size=4):
        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i:i+batch_size]
            embeddings = [self.embedder.embed(seq) for seq in batch_seqs]
            yield embeddings

# class CCD:
#     def __init__(self, data_dict=None, file_path=None):
#         self.file_path = file_path
#         self.ifo = ifo.iProtein(file=file_path)
#         self.init_feature()

#     def init_feature(self):
#         # feature_name_list = list(CCD_INFO.keys())
#         features=[]
#         feature_name_list = [
#             "AAC",
#             "APAAC",
#             "CKSAAGP type 1",
#             "CKSAAP type 1",
#             "CTDC",
#             "CTDD",
#             "CTDT",
#             "DDE",
#             "DPC type 1",
#             "GAAC",
#             "GDPC type 1",
#             "GTPC type 1",
#             "Geary",
#             "KSCTriad",
#             "Moran",
#             "QSOrder",
#             "SOCNumber"
#         ]

#         # CCD_FEATURES = torch.tensor(
#         #     [[-1 for _ in range(394)] for _ in range(self.ifo.sequence_number)],
#         #     dtype=torch.float
#         # )

#         for feature_name in feature_name_list:
#             print(feature_name,'namesss')
#             self.ifo.get_descriptor(feature_name)
#             encodings = self.ifo.encodings
#             # append to features dataframe
#             # print(encodings.shape)
#             # features.append(encodings)
#             indices = CCD_INFO[feature_name]["indices"]
#             order = CCD_INFO[feature_name]["order"]
#         # print(features.columns,'columns')
#         # change columns to feataure_0, feature_1, ...
#         # pd_features = pd.concat(features, axis=1)
#         # print(pd_features.columns,'all columns')
#         # features = pd_features.reset_index(drop=True)
#         # print(features.columns,'feature columns')
#         # features.columns = [f"feature_{i}" for i in range(features.shape[1])]

#         # print(features.columns,'new columns')
#         # read pickle file list of selected features
#         # with open('/home/vinoth/Hari_proj/TYLCV/webserver/Github_code/DeepTYLCV/src/selected_features_final_with_new_dataset.pkl', 'rb') as f:
#         #     optimal_features_list = pickle.load(f)
#         # print(optimal_features_list,'optimal_features_list')
#         # CCD_FEATURES = features[optimal_features_list]
#         # print(optimal_features,'optimal_features') 
#         # print(optimal_features.shape,'optimal_features shape')       
#     #         CCD_FEATURES[:, order] = torch.tensor(encodings.iloc[:, indices].values, dtype=torch.float)
  
#         # self.CCD_FEATURES_DF = selected_df  # (n_seq x n_feat)
#         self.CCD_FEATURES = torch.tensor(CCD_FEATURES.values, dtype=torch.float32)
    
#     def get_features_batch(self, batch_size=4):
#         for i in range(0, len(self.CCD_FEATURES), batch_size):
#             batch = [self.CCD_FEATURES[i].unsqueeze(0) for i in range(i, min(i + batch_size, len(self.CCD_FEATURES)))]
#             yield batch

#     def get_features_all(self):
#         return [self.CCD_FEATURES[i].unsqueeze(0) for i in range(len(self.CCD_FEATURES))]


class CCD:
    def __init__(self, data_dict=None, file_path=None):
        if not file_path and not data_dict:
                    raise ValueError("Provide either file_path (FASTA) or data_dict={id: seq}.")

                # If only dict provided, dump to a temp FASTA that iProtein understands
        if file_path:
            fasta_to_use = file_path
            self._tmp_fasta = None
        else:
            records = [SeqRecord(Seq(seq), id=str(sid), description="") for sid, seq in data_dict.items()]
            tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False)
            SeqIO.write(records, tmp, "fasta")
            tmp.close()
            fasta_to_use = tmp.name
            self._tmp_fasta = fasta_to_use  # for cleanup if you want
        if self._tmp_fasta:
            atexit.register(lambda p=self._tmp_fasta: os.path.exists(p) and os.remove(p))
        self.file_path = fasta_to_use
        # IMPORTANT: no data_dict here — your installed iProtein only accepts file=
        self.ifo = ifo.iProtein(file=self.file_path)
        self.init_feature()

    def init_feature(self):
        feature_name_list = list(CCD_INFO.keys())

        CCD_FEATURES = torch.tensor(
            [[-1 for _ in range(394)] for _ in range(self.ifo.sequence_number)],
            dtype=torch.float
        )

        for feature_name in feature_name_list:
            self.ifo.get_descriptor(feature_name)
            encodings = self.ifo.encodings
            indices = CCD_INFO[feature_name]["indices"]
            order = CCD_INFO[feature_name]["order"]

            CCD_FEATURES[:, order] = torch.tensor(encodings.iloc[:, indices].values, dtype=torch.float)
  
        self.CCD_FEATURES = CCD_FEATURES
    
    def get_features_batch(self, batch_size=4):
        for i in range(0, len(self.CCD_FEATURES), batch_size):
            batch = [self.CCD_FEATURES[i].unsqueeze(0) for i in range(i, min(i + batch_size, len(self.CCD_FEATURES)))]
            yield batch

    def get_features_all(self):
        return [self.CCD_FEATURES[i].unsqueeze(0) for i in range(len(self.CCD_FEATURES))]