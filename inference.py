from Bio import SeqIO
# from src.feature_extractor import ESM, ProtTrans, CCD
from src.feature_extractor import ESM, PTAB,PTBB,CCD
from tqdm import tqdm
import os
import torch
import pickle

class Inferencer:
    def __init__(self,predictor,device='cpu'):
        self.predictor = predictor
        self.esm=  ESM(device=device)
        self.ptab = PTAB(device=device)
        self.ptbb = PTBB(device=device)
    
    @staticmethod 
    def read_fasta_file(fasta_file):
        data_dict = {}
        for record in SeqIO.parse(fasta_file, 'fasta'):
            assert record.id not in data_dict, f'Duplicated ID: {record.id}'
            data_dict[record.id] = str(record.seq)
        return data_dict
    
    def scale_ccd_features(self,ccd_features, scaler_path):
        # scaler_path = os.path.join(scaler_dir, 'scaler.pkl')
        # if not os.path.exists(scaler_path):
        #     raise FileNotFoundError(f"Scaler file not found at {scaler_path}")
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        scaled = scaler.transform(ccd_features.numpy())
        return torch.tensor(scaled, dtype=torch.float)
        
    def predict_fasta_file(self, fasta_file, threshold=0.5, batch_size=4):
        data_dict = self.read_fasta_file(fasta_file)
        keys = list(data_dict.keys())
        seqs = list(data_dict.values())
        scaler_path = '/home/vinoth/Hari_proj/TYLCV/webserver/Github_code/DeepTYLCV/DeepTYLCV_webserver_data/scaler.pkl'
        total_batch_len = (len(seqs) // batch_size) + int(len(seqs) % batch_size == 0)
        esm_generator=self.esm.get_features_batch(seqs, batch_size=batch_size)
        ptab_generator=self.ptab.get_features_batch(seqs, batch_size=batch_size)
        ptbb_generator=self.ptbb.get_features_batch(seqs, batch_size=batch_size)
        ccd_generator=CCD(file_path=fasta_file).get_features_batch(batch_size=batch_size)
        ALL_LABELS = []
        ALL_PROBS = []

        # print(seqs)
        for esm_features, ptab_features, ptbb_features, ccd_features in tqdm(zip(esm_generator, ptab_generator, ptbb_generator, ccd_generator), total=total_batch_len):
            # print(esm_features)
            # print(ptab_features)
            # print(ptbb_features)
            # print(ccd_features)
            # exit()
            ccd_tensor = torch.cat(ccd_features, dim=0)
            if scaler_path is not None:
                ccd_tensor = self.scale_ccd_features(ccd_tensor, scaler_path)
            # print(ccd_tensor,'ccd_tensor')
            # exit()
            labels, probs = self.predictor(
                esm_features,ptab_features,ptbb_features, ccd_tensor, threshold=threshold
            )
            ALL_LABELS.extend(labels)
            ALL_PROBS.extend(probs)
        
        return {key: [label, prob] for key, label, prob in zip(keys, ALL_LABELS, ALL_PROBS)}
    
    def predict_sequences(self, data_dict, threshold=0.5, batch_size=4):
        keys = list(data_dict.keys())
        seqs = list(data_dict.values())
        total_batch_len = (len(seqs) // batch_size) + int(len(seqs) % batch_size == 0)
        scaler_path = '/home/vinoth/Hari_proj/TYLCV/webserver/Github_code/DeepTYLCV/DeepTYLCV_webserver_data/scaler.pkl'
        esm_generator=self.esm.get_features_batch(seqs, batch_size=batch_size)
        ptab_generator=self.ptab.get_features_batch(seqs, batch_size=batch_size)
        ptbb_generator=self.ptbb.get_features_batch(seqs, batch_size=batch_size)
        ccd_generator=CCD(data_dict=data_dict).get_features_batch(batch_size=batch_size)
        ALL_LABELS = []
        ALL_PROBS = []

        # print(seqs)
        for esm_features, ptab_features, ptbb_features, ccd_features in tqdm(zip(esm_generator, ptab_generator, ptbb_generator, ccd_generator), total=total_batch_len):
            # print(esm_features)
            # print(ptab_features)
            # print(ptbb_features)
            # print(ccd_features)
            # exit()
            ccd_tensor = torch.cat(ccd_features, dim=0)
            if scaler_path is not None:
                ccd_tensor = self.scale_ccd_features(ccd_tensor, scaler_path)
            # print(ccd_tensor,'ccd_tensor')
            # exit()
            labels, probs = self.predictor(
                esm_features,ptab_features,ptbb_features, ccd_tensor, threshold=threshold
            )
            ALL_LABELS.extend(labels)
            ALL_PROBS.extend(probs)
        
        return {key: [label, prob] for key, label, prob in zip(keys, ALL_LABELS, ALL_PROBS)}

    @staticmethod
    def save_csv_file(outputs, output_path):
        with open(output_path, 'w') as f:
            f.write('ID,severity,Probability\n')
            for key, value in outputs.items():
                f.write(f'{key},{value[0]},{value[1]}\n')