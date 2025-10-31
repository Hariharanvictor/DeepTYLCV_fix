from argparse import ArgumentParser
import pickle

import pandas as pd
from src.utils import get_config
from src.lightning_module import DeepTYLCV_Module
from src.dataset import DeepTYLCV_Dataset
from src.metrics import calculate_metrics
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



def extract_headers_labels(fasta_file):
    headers= []
    labels = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        tem_header = record.description.replace('.', '_')
        headers.append(tem_header)
        if 'Severe' in tem_header:
            labels.append(1)
        else:
            labels.append(0)
    return headers, labels

def test_kfold_models(model, ckpt_dir, test_dataloader, device):
    MODELS_LIST = []
    for ckpt_file in os.listdir(ckpt_dir):
        if not ckpt_file.endswith('.ckpt'):
            continue
        
        pt_file_path = os.path.join(ckpt_dir, ckpt_file)
        state_dict = torch.load(pt_file_path)['state_dict']
        model.load_state_dict(state_dict)
        model.eval()
        model.to(device)
        
        MODELS_LIST.append(copy.deepcopy(model))
    
    
    ALL_Y = []
    ALL_PROBS = []
    for batch in tqdm(test_dataloader, desc='Testing', total=len(test_dataloader)):
        total_prob_batch = []
        X,X_conv, masks_X, y = batch
        X = [x.to(device) for x in X]
        X_conv =X_conv.to(device)
        masks_X = [mask.to(device) for mask in masks_X]
        y = y.to(device)
        
        for model in MODELS_LIST:
            # Copy masks_X to avoid in-place operation
            masks_X_tmp = [mask.clone() for mask in masks_X]
            with torch.no_grad():
                logits = model(X, masks_X_tmp,X_conv)
                
            total_prob_batch.append(logits.softmax(dim=-1))
        
        avg_prob_batch = torch.sum(torch.stack(total_prob_batch), dim=0) / len(MODELS_LIST)
        
        max_probs = avg_prob_batch.max(dim=-1)
        max_probs = torch.abs((max_probs.indices + 1) % 2 - max_probs.values) # Reverse probability when class is 0
        
        ALL_PROBS.append(max_probs)
        ALL_Y.append(y)
    
    ALL_Y = torch.concat(ALL_Y)
    ALL_PROBS = torch.concat(ALL_PROBS)
    
    avgprob_output_metrics = calculate_metrics(
        ALL_PROBS.detach().cpu().numpy(),
        ALL_Y.detach().cpu().numpy()
    )
    
    with open(os.path.join(ckpt_dir, 'test_avg_result.txt'), 'w') as f:
        for metric_name, value in avgprob_output_metrics.items():
            f.write(f'{metric_name}: {value}\n')
            
def evaluate_model(model, ckpt_dir, fold, dataloader, device):
    ckpt_files = [f for f in os.listdir(ckpt_dir) if f'fold_{fold}' in f and f.endswith('.ckpt')]
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoint file found for fold {fold} in directory {ckpt_dir}")
    
    pt_file_path = os.path.join(ckpt_dir, ckpt_files[0])
    state_dict = torch.load(pt_file_path)['state_dict']
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    
    ALL_Y = []
    ALL_PROBS = []
    for batch in dataloader:
        X, X_conv, masks_X, y = batch
        X = [x.to(device) for x in X]
        X_conv = X_conv.to(device)
        masks_X = [mask.to(device) for mask in masks_X]
        y = y.to(device)
        
        with torch.no_grad():
            logits = model(X, masks_X, X_conv)
            
        max_probs = logits.softmax(dim=-1).max(dim=-1)
        max_probs = torch.abs((max_probs.indices + 1) % 2 - max_probs.values) # Reverse probability when class is 0
        
        ALL_PROBS.append(max_probs)
        ALL_Y.append(y)
    
    ALL_Y = torch.concat(ALL_Y)
    ALL_PROBS = torch.concat(ALL_PROBS)
    
    with open(os.path.join(ckpt_dir, f'fold_thres_{fold}_result.txt'), 'w') as f:
        
        output_metrics = calculate_metrics(
                ALL_PROBS.detach().cpu().numpy(),
                ALL_Y.detach().cpu().numpy(),
                threshold=0.5
            )
        f.write(f'Threshold: 0.5\n')
        for metric_name, value in output_metrics.items():
            f.write(f'{metric_name}: {value}\n')
        f.write('-'*20 + '\n')

def train_nlp_conv(config):
    pl.seed_everything(config.seed)
    device = torch.device(config.trainer.device)
    test_file_path=config.dataset.test_fasta_path
    header_test,label_test=extract_headers_labels(test_file_path)
    test_dataset = DeepTYLCV_Dataset(config.dataset,header_test, label_test)
    
    # Create folder if it's not existed and save config file to output path
    if config.save_config:
        if not os.path.exists(config.trainer.output_path):
            os.makedirs(config.trainer.output_path)
        with open(os.path.join(config.trainer.output_path, 'config.yaml'), 'w') as f:
            yaml.dump(config, f)
    
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=config.trainer.num_workers, pin_memory=True)
    
    headers_all_fold=[]
    labels_all_fold=[]
    fasta_files=sorted(glob.glob(f'{config.dataset.train_dir}/*.fasta'))
    for fasta_file in fasta_files: 
        header,label=extract_headers_labels(fasta_file)
        headers_all_fold.append(header)
        labels_all_fold.append(label)
        

    for idx in range(len(headers_all_fold)):
        header_val=headers_all_fold[idx]
        label_val=labels_all_fold[idx]
        header_train=[]
        label_train=[]
        for jdx in range(len(headers_all_fold)):
            if idx==jdx:
                continue
            else:
                header_train.extend(headers_all_fold[jdx])
                label_train.extend(labels_all_fold[jdx])
        
        tb_logger = TensorBoardLogger(
            save_dir=config.trainer.output_path,
            name=f'fold_{idx+1}'
        )
        
        # metrics_logger = MetricsLogger(config.trainer.output_path, idx+1)
        train_dataset = DeepTYLCV_Dataset(config.dataset,header_train, label_train)
        val_dataset = DeepTYLCV_Dataset(config.dataset,header_val, label_val)
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=config.trainer.batch_size, shuffle=True, num_workers=config.trainer.num_workers, pin_memory=True)
        val_dataloader = DataLoader(dataset=val_dataset, batch_size=config.trainer.batch_size, shuffle=False, num_workers=config.trainer.num_workers, pin_memory=True)

        model_module = DeepTYLCV_Module(config.model, config.trainer, config.loss)

        trainer = pl.Trainer(
            accelerator=config.trainer.device,
            max_epochs=config.trainer.epochs,
            val_check_interval=0.5,
            gradient_clip_val=10.,
            callbacks=[
                ModelCheckpoint(
                    monitor='bacc',
                    dirpath=config.trainer.output_path,
                    filename=f'fold_{idx+1}_{{epoch}}_{{sn:.5f}}_{{sp:.5f}}_{{bacc:.5f}}_{{threshold:.2f}}',
                    save_top_k=1,
                    mode='max'
                ),
                LearningRateMonitor(logging_interval='epoch')
            ],
            logger=tb_logger
        )
        
        trainer.fit(model_module, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
        
        evaluate_model(model_module, config.trainer.output_path, idx+1, val_dataloader, device)
        
    
    test_kfold_models(model_module, config.trainer.output_path, test_dataloader, device)


    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--config_path', type=str, default='configs/default_config.yaml')
    parser.add_argument('--save_config', action='store_true')
    args = parser.parse_args()
    config = get_config(args.config_path)
    
    # Set args to config
    config.save_config = args.save_config
    # main(config)
    train_nlp_conv(config)