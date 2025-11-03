# DeepTYLCV_fix
<h1 align="center">DeepTYLCV</h1>
<p align="center"><a href="https://balalab-skku.org/DeepTYLCV/">🌐 Webserver (CBBL-SKKU)</a> | <a href="https://1drv.ms/f/some_onedrive_link">🚩 Model & Dataset</a></p>

The official implementation of **DeepTYLCV: An interpretable and experimentally validated AI model for predicting virulence in tomato yellow leaf curl virus**

## TOC

This project is summarized into:
- [Installing environment](#installing-environment)
- [Preparing datasets](#preparing-datasets)
- [Configurations](#configurations)
- [Training models](#training-models)
- [Inferencing models](#inferencing-models)

## Installing environment
First, install [Miniconda](https://docs.anaconda.com/miniconda/) or [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html).  
Then create and activate the environment:

```bash
conda create -n deeptylcv python=3.10
conda activate deeptylcv
```
Next, install the required dependencies:
```bash
cd DeepTYLCV/
python -m pip install -r requirements.txt --no-cache-dir
```
**Preparing datasets**

DeepTYLCV uses Open Reading Frames(ORF) originated from Genomes sequences. ORF can be found in the data/ directory. In this project, we used 3 Protein Language models(PLM): ESM-1 (ESM), ProtTrans-ALBERT-BFD (PTAB), ProtTrans-BERT-BFD (PTBB). Which are provided in the Zenodo and optimal concatenated conventional descriptors features(optCCDS) are provided in the opt_CCD_features/ directory.

**Configurations**

You can find configuration files in the configs/ directory:


Main parameters include:
```bash
dataset:
  conv_lmdb_path: /opt_CCD_features
  data_root: /PLM
  feature_list:
  - ESMEmbedder
  - ProtTransAlbertBFDEmbedder
  - ProtTransBertBFDEmbedder
  max_length: 363
  mean: false
  test_fasta_path: /data/test.fasta
  train_dir: /data/5-fold-data
  tran_nlp: true
....
model:
  d_model: 128
  dilation: 1
  dp_size: 0.3
  fc_1: 128
  feature_dim_list:
  - 1280
  - 4096
  - 1024
  kernel_size:
  - 5
  - 7
  - 9
  max_concatenated_len: 363
  n_classes: 2
  n_head: 8
  norm_type: batch
  num_transformer_layers: 6
  stride: 3
...
trainer:
  batch_size: 16
  device: cpu
  epochs: 100
  lr: 0.0001
  num_workers: 4
  output_path: results/
```
**Training models**
To reconstruct the results from the paper, you can run two following commands:
```bash
python train.py --config_path /configs/config_DeepTYLCV.yaml --save_config
```
**Inferencing models**
You can easily predict directly from FASTA files or sequence dictionaries using the Inferencer:
```bash
from predictor import DeepTYLCV_Predictor
from inference import Inferencer
import yaml

config = yaml.safe_load(open('configs/config_DeepTYLCV.yaml'))

predictor = DeepTYLCV_Predictor(
    model_config=config['model'],
    ckpt_dir='/path/to/ckpt/dir',
    nfold=5,
    device='cuda'
)

infer = Inferencer(predictor, scaler_path='/path/to/scaler/model' device='cuda')

# Predict from FASTA file
outputs = infer.predict_fasta_file(
    fasta_file='/path/to/input.fasta',
    threshold=0.5,
    batch_size=4
)

# Predict from sequences directly
outputs_seq = infer.predict_sequences(
    data_dict={'TYLCV_seq_1': 'MSSSHIFIGETIGT', 'TYLCV_seq_2': 'CFGG'},
    threshold=0.5,
    batch_size=4
)

# Save results
infer.save_csv_file(outputs, 'DeepTYLCV_predictions.csv')
```
DeepTYLCV provides an end-to-end deep learning pipeline — from feature extraction and model fusion to sequence-level prediction — to classify the severity of TYLCV infection using multi-scale PLM fusion and biological descriptors.
